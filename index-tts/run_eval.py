"""
TTS synthesis for the Open TTS Leaderboard (IndexTTS-2.5 backend, stage 1).

IndexTTS-2.5 is run from a source clone of https://github.com/index-tts/index-tts via
`indextts.infer_v2_5.IndexTTS2` (22.05 kHz output). `infer()` takes the language as a REQUIRED
`lang` argument, so `--language` is passed through from the dataset config; submit_jobs.sh skips
the languages the model does not support.

No native batching: upstream fixes the batch at one utterance (CFM caches allocated at
max_batch_size=1, GPT `num_return_sequences=1`, `infer()` takes one string). `--batch_size` only
controls manifest chunking / resume; per-sample time is batch time / batch size.

There is no reference-free mode (`infer()` requires `spk_audio_prompt`). The two modes differ in
WHICH reference is used:

  --voice_clone: each sample clones its own `prompt_audio`, so SIM can be scored.
      Outputs get a `_voice_clone` suffix.
  --no-voice_clone (default): ONE fixed reference for the whole split (default: the split's sample
      `--fixed_prompt_index`, so it is language-matched), for WER/CER comparison against the
      fixed-voice backends. SIM is skipped.
      NOTE fixed-voice RTFx is not strictly comparable with cloning RTFx: upstream caches the
      speaker conditioning keyed on the prompt path, so it is encoded once for the whole split.
"""

import argparse
import contextlib
import logging
import os
import warnings

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, wav_rel_path, warm_up, write_entry,
)

# index-tts pulls in chatty deps (modelscope, transformers, numba); keep warnings/info out of the
# way so the outer "Generating" tqdm is the only progress output.
warnings.filterwarnings("ignore")
for _name in ("modelscope", "numba", "matplotlib"):
    logging.getLogger(_name).setLevel(logging.WARNING)
try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
except Exception:
    pass

# `indextts.infer_v2_5` sets HF_HUB_CACHE to a relative path at import time, which would redirect
# `load_dataset`; restore the pre-import value. The model loads its weights from explicit paths.
_HF_HUB_CACHE_BEFORE = os.environ.get("HF_HUB_CACHE")

from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

if _HF_HUB_CACHE_BEFORE is None:
    os.environ.pop("HF_HUB_CACHE", None)
else:
    os.environ["HF_HUB_CACHE"] = _HF_HUB_CACHE_BEFORE

torch.set_float32_matmul_precision("high")

# `infer()` returns int16-valued samples (upstream scales by 32767 for Gradio); soundfile wants
# float32 in [-1, 1].
INT16_SCALE = 32768.0


def main(args):
    # Set seed for reproducibility (the GPT samples mel codes and the s2mel head is a CFM sampler).
    seed = args.seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True

    checkpoint_path = _resolve_checkpoint(args)
    cfg_path = os.path.join(checkpoint_path, "config.yaml")
    model = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=checkpoint_path,
        device=args.device,
        use_bf16=args.use_bf16,
        # Off by default: the fused BigVGAN kernel needs nvcc, and plain torch keeps RTFx
        # like-for-like with the other backends.
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=False,   # not installed (upstream ships it as an extra)
        use_accel=False,       # needs flash-attn (upstream `accel` extra), not installed
        use_torch_compile=False,
        # QwenEmotion is only needed for infer(use_emo_text=True), which this eval never uses.
        use_qwen_emo=False,
    )
    # 22.05 kHz: the s2mel/BigVGAN output rate, read from the config rather than hardcoded.
    sampling_rate = int(model.cfg.s2mel["preprocess_params"]["sr"])
    print(
        f"Loaded IndexTTS-2.5 ({args.model_id}, sr={sampling_rate}, bf16={args.use_bf16}, "
        f"lang={args.language}, max_text_tokens_per_segment={args.max_text_tokens_per_segment})"
    )

    # Parameter count: IndexTTS2 is not an nn.Module, so sum unique params over its nn.Module attributes.
    try:
        print_model_size(model)
    except Exception as e:
        print(f"Could not determine model size: {e}")

    # Flat, bucket-friendly layout, per model:
    #   results/<model_safe>/  MODEL_<safe>_DATASET_<dsid>.jsonl  +  <dsid>/output_<id>.wav
    # Manifest paths are relative to model_dir, so downstream stages resolve them wherever it is copied.
    # Keep the two modes' audio separate so voice-clone and fixed-voice runs don't overwrite.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir

    # `infer()` takes the reference as a file path; prompts live under `results/` for stage 3 SIM.
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
    # Fixed-voice mode: one reference for the whole split, filled in after the dataset is loaded.
    fixed_prompt = {"full": None}
    n_empty = 0  # samples that produced no audio (reported at the end)
    # infer_generator() prints several progress lines per call; redirect stdout only (tqdm and
    # tracebacks go to stderr).
    devnull = None if args.verbose_model else open(os.devnull, "w")

    def quiet_model():
        return contextlib.nullcontext() if devnull is None else contextlib.redirect_stdout(devnull)

    def write_prompt(path, prompt_audio):
        """Write one reference clip to `path` as a wav (upstream truncates it to 15 s itself)."""
        array = np.asarray(prompt_audio["array"], dtype=np.float32).reshape(-1)
        sf.write(path, array, int(prompt_audio["sampling_rate"]))

    def synth_one(text, prompt_path):
        """Synthesize one utterance and return it as a float32 waveform.

        `output_path=None` makes `infer()` return the samples, keeping file I/O out of the timed region.
        """
        nonlocal n_empty
        # Re-seed per sample so generation is reproducible regardless of batch layout.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        out = model.infer(
            spk_audio_prompt=prompt_path,
            text=text,
            output_path=None,
            lang=args.language,
            # emo_audio_prompt=None: the speaker reference doubles as the emotion reference
            # (upstream's plain zero-shot cloning path).
            emo_audio_prompt=None,
            interval_silence=args.interval_silence,
            max_text_tokens_per_segment=args.max_text_tokens_per_segment,
            duration_factor=args.duration_factor,
            text_normalization=not args.no_text_normalization,
            verbose=False,
        )
        # infer() returns None when its generator yielded nothing (it swallows the IndexError).
        if out is None:
            n_empty += 1
            print(f"Warning: no audio generated for text: {text[:60]}...")
            return np.zeros(int(0.1 * sampling_rate), dtype=np.float32)
        _, wav_int16 = out
        return np.asarray(wav_int16, dtype=np.float32).reshape(-1) / INT16_SCALE

    def generate_tts(batch):
        """Synthesize a minibatch of target texts (per-sample loop); time the batch for RTFx."""
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)

        # Write the prompts before timing (I/O is not generation). The relative path goes into the
        # manifest for stage 3 (SIM).
        if args.voice_clone:
            prompt_rel_paths, prompt_full_paths = [], []
            for sample_id, prompt_audio in zip(batch["id"], batch["prompt_audio"]):
                prel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
                ppath = os.path.join(model_dir, prel)
                if not os.path.exists(ppath):
                    write_prompt(ppath, prompt_audio)
                prompt_rel_paths.append(prel)
                prompt_full_paths.append(ppath)
        else:
            # Fixed-voice mode: the same reference for every sample; no SIM.
            prompt_rel_paths = [None] * minibatch_size
            prompt_full_paths = [fixed_prompt["full"]] * minibatch_size

        # START TIMING (TTS generation for the whole minibatch)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # No batched API (see the module docstring): loop per sample.
        with quiet_model():
            wavs = [
                synth_one(text, prompt_path)
                for text, prompt_path in zip(texts_to_generate, prompt_full_paths)
            ]

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(wavs, batch["id"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        batch["gen_audio_filepath"] = gen_paths
        batch["prompt_audio_filepath"] = prompt_rel_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    # `prompt_audio` is kept in both modes (fixed-voice mode may take its one reference from the
    # split); it is dropped below when not cloning. `prompt_text` is unused: infer() takes no transcript.
    dataset = load_tts_dataset(args, ("prompt_audio",))

    # A reference is mandatory in both modes, so the dataset must carry one unless fixed-voice mode
    # was given an external reference.
    needs_dataset_prompt = args.voice_clone or not args.prompt_audio_path
    if "prompt_audio" not in dataset.column_names and needs_dataset_prompt:
        raise ValueError(
            f"Dataset {args.dataset_path}/{args.dataset}/{args.split} has no `prompt_audio` column. "
            "IndexTTS-2.5 has no built-in speaker, so a reference audio is required in every mode. "
            "Pass --prompt_audio_path to supply one externally."
        )

    # ── Fixed-voice mode: resolve the ONE reference used for the whole split ──
    if not args.voice_clone:
        fixed_rel = os.path.join(dataset_dir_name, "prompts", "fixed_prompt.wav")
        fixed_full = os.path.join(model_dir, fixed_rel)
        if args.prompt_audio_path:
            audio, sr = sf.read(args.prompt_audio_path, dtype="float32", always_2d=True)
            write_prompt(fixed_full, {"array": audio.mean(axis=1), "sampling_rate": sr})
            src = args.prompt_audio_path
        else:
            # Default: the split's own sample #--fixed_prompt_index, which keeps the reference
            # language-matched.
            if not 0 <= args.fixed_prompt_index < len(dataset):
                raise ValueError(
                    f"--fixed_prompt_index={args.fixed_prompt_index} is out of range for a "
                    f"{len(dataset)}-sample split."
                )
            row = dataset[args.fixed_prompt_index]
            write_prompt(fixed_full, row["prompt_audio"])
            src = f"{args.dataset}/{args.split} sample #{args.fixed_prompt_index} (id={row.get('id')})"
        fixed_prompt["full"] = fixed_full
        print(
            f"Fixed-voice mode: one reference for all {len(dataset)} samples, from {src}. "
            f"SIM is skipped (no per-sample reference)."
        )

    # Fixed-voice mode: drop the reference column so nothing else triggers audio decoding.
    if not args.voice_clone and "prompt_audio" in dataset.column_names:
        dataset = dataset.remove_columns(["prompt_audio"])

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Drives generate_tts() with one-row batches; `model_dir`/`dataset_dir_name` are rebound to a
    # temp dir (generate_tts closes over them) so nothing lands in the results tree. TTFA is the
    # whole-utterance time: infer(stream_return=True) yields per 120-token text segment, so it would
    # emit a single chunk for eval-length utterances anyway. See scripts/ttfa_probe.py.
    if args.ttfa_probe != 0:
        # Imported here: the probe module is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        def _gen(job):
            out = generate_tts(dict(job))
            return float(out["audio_length_s"][0]), None, None

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device,
                   "note": "driven at batch size 1, blocking infer() (see the probe note in run_eval.py)"},
        )
        return

    # ── Manifest (written incrementally; resume skips rows already in it) ───
    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(generate_tts, dataset, starts, args.batch_size, args.warmup_steps)
    for batch_start in tqdm(starts, desc="Generating"):
        batch = generate_tts(dataset[batch_start : batch_start + args.batch_size])

        # Append each sample to the JSONL immediately. Only cloning runs carry the prompt path (its
        # absence tells the pipeline to skip SIM).
        for afp, ppath, alen, gtime, ref in zip(
            batch["gen_audio_filepath"], batch["prompt_audio_filepath"], batch["audio_length_s"],
            batch["generation_time_s"], batch["references"],
        ):
            entry = manifest_entry(afp, alen, gtime, ref, ppath if args.voice_clone else None)
            write_entry(manifest_file, entry)
        manifest_file.flush()

    manifest_file.close()
    if devnull is not None:
        devnull.close()
    if n_empty:
        print(f"NOTE: {n_empty} sample(s) generated no audio and were written as 0.1 s of silence.")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


def _resolve_checkpoint(args):
    """Return a writable dir holding the IndexTTS-2.5 weights + its auxiliary models.

    Unless --checkpoint_path points at a pre-populated dir, the weights are downloaded to a
    writable `local_dir` (IndexTTS2 stores auxiliary models under `<model_dir>/hf_cache/`).
    `qwen0.6bemo4-merge/` is skipped since main() passes use_qwen_emo=False.
    """
    from huggingface_hub import snapshot_download

    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        if not os.path.isfile(os.path.join(checkpoint_path, "config.yaml")):
            raise FileNotFoundError(
                f"No config.yaml under --checkpoint_path={checkpoint_path}. Point it at a dir "
                f"populated by `hf download {args.model_id} --local-dir <dir>`, or drop the flag "
                "to let this script download the weights itself."
            )
        print(f"Using existing checkpoint dir {os.path.abspath(checkpoint_path)}")
    else:
        print(f"Downloading {args.model_id} to {os.path.abspath(args.download_dir)} ...")
        checkpoint_path = snapshot_download(
            args.model_id,
            local_dir=args.download_dir,
            ignore_patterns=["qwen0.6bemo4-merge/*"],
        )

    # Fetch the auxiliary models (w2v-BERT, CAMPPlus, BigVGAN, ...) up front with upstream's helper,
    # so the download is a visible step instead of a stall inside __init__.
    from indextts.utils.model_download import ensure_models_available

    ensure_models_available(checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="IndexTeam/IndexTTS-2.5",
        help="Model id, used for the results folder / manifest name.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Local dir already holding the weights + config.yaml. Default: download --model_id "
             "into --download_dir.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="checkpoints",
        help="Where to download the weights when --checkpoint_path is not given. Must be writable: "
             "IndexTTS2 stores its auxiliary models under <dir>/hf_cache/.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language tag passed to infer(lang=...). IndexTTS-2.5 covers zh, en, ja, es, ar (plus "
             "the mixed 'zhen'); it does NOT infer the language from the text. An unknown tag falls "
             "back to upstream's 'common' token, which is why submit_jobs.sh gates the combos instead.",
    )
    # --batch_size is only the manifest chunk size (no batched API; generation is per sample).
    add_common_args(parser, batch_size=32, voice_clone=False)
    parser.add_argument(
        "--use_bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the GPT backbone in bfloat16 (the model card's own example passes use_bf16=True).",
    )
    parser.add_argument(
        "--use_cuda_kernel",
        action="store_true",
        help="Compile BigVGAN's fused anti-alias activation CUDA kernel. Needs nvcc (a -devel base "
             "image); off by default so RTFx is measured on the same plain-torch vocoder path as "
             "the other backends.",
    )
    parser.add_argument(
        "--max_text_tokens_per_segment",
        type=int,
        default=120,
        help="Upstream default. Longer texts are split into segments, synthesized separately and "
             "concatenated with --interval_silence between them.",
    )
    parser.add_argument(
        "--interval_silence",
        type=int,
        default=200,
        help="Milliseconds of silence inserted between segments of one utterance (upstream default).",
    )
    parser.add_argument(
        "--duration_factor", type=float, default=1.0, help="Speech-rate control (0.5-2.0; 1.0 = model default)."
    )
    parser.add_argument(
        "--no_text_normalization",
        action="store_true",
        help="Disable IndexTTS's own text normalization (on by default, as upstream ships it).",
    )
    parser.add_argument(
        "--verbose_model",
        action="store_true",
        help="Let infer()'s per-sample progress/timing prints through (suppressed by default).",
    )
    # --voice_clone / --no-voice_clone (default): see the module docstring. A reference is required
    # either way: IndexTTS-2.5 has no built-in speaker.
    parser.add_argument(
        "--fixed_prompt_index",
        type=int,
        default=0,
        help="--no-voice_clone only: which sample of the split supplies the fixed reference "
             "(default 0), keeping the voice language-matched to the split.",
    )
    parser.add_argument(
        "--prompt_audio_path",
        type=str,
        default=None,
        help="--no-voice_clone only: use this wav as the fixed reference instead of a dataset sample "
             "(e.g. one shared voice across languages).",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
