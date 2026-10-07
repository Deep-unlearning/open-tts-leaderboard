"""
TTS synthesis for the Open TTS Leaderboard (Fun-CosyVoice3 backend, stage 1).

There is no reference-free mode: the HF weights ship no `spk2info.pt`, and every other entry point
takes a `prompt_wav`. The two modes differ in WHICH reference is used:

  --voice_clone: each sample clones its own `prompt_audio`/`prompt_text`, so SIM can be
      scored. Outputs get a `_voice_clone` suffix.
  --no-voice_clone (default): ONE fixed reference for the whole split (default: the split's sample
      `--fixed_prompt_index`, so it is language-matched), for WER/CER comparison against the
      fixed-voice backends. SIM is skipped.

Upstream behaviours worked around here:

1. `inference_zero_shot()` yields one waveform PER SENTENCE; `synth_one` concatenates them.
2. The reference is passed as a file path, so prompts are written to wavs (under `results/` for
   stage 3), clipped to <= 30 s and upsampled to >= 16 kHz to satisfy upstream asserts.
3. CosyVoice3 expects an instruct prefix on the PROMPT text (see upstream `cosyvoice3_example()`).
4. The text frontend routes all non-Chinese text through the English wetext normalizer, which can
   crash on some inputs (e.g. "1970년", "#5", doubled spaces); `synth_one` retries those with
   `text_frontend=False`. It also spells digits out in English for non-en/zh text, so prefer
   `--no_text_frontend` there.

No native batching: texts are synthesized one at a time and `--batch_size` only controls manifest
chunking / resume; per-sample time is batch time / batch size. `--device` is used only for timing,
since `CosyVoiceModel` hardcodes `torch.device('cuda')`.
"""

import argparse
import logging
import os
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, wav_rel_path, warm_up, write_entry,
)

# `cosyvoice` is used from a clone of the upstream repo; third_party/Matcha-TTS must be importable
# too (cosyvoice3.yaml instantiates a matcha module at load time).
COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice")
sys.path.insert(0, COSYVOICE_ROOT)
sys.path.insert(0, os.path.join(COSYVOICE_ROOT, "third_party/Matcha-TTS"))

# Silence cosyvoice's per-chunk INFO logs and per-sample warnings.
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

torch.set_float32_matmul_precision("high")

# Instruct-style prefix prepended to the PROMPT text. Overridable with --instruct_prefix.
INSTRUCT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
# frontend._extract_speech_token() asserts prompt duration <= 30 s.
PROMPT_MAX_SECONDS = 30.0
# file_utils.load_wav() asserts sample_rate >= 16000 whenever it has to resample.
PROMPT_MIN_SR = 16000


def _resolve_model_dir(args):
    """Return the model dir to load, honouring --llm_checkpoint.

    `CosyVoice3Model.load()` hardcodes `llm.pt`, so to select `llm.rl.pt` without mutating the
    weights dir, build a shadow dir of symlinks whose `llm.pt` points at the requested checkpoint.
    """
    model_dir = args.checkpoint_path if os.path.isdir(args.checkpoint_path) else args.model_id
    if args.llm_checkpoint == "llm.pt":
        return model_dir
    if not os.path.isdir(model_dir):
        raise ValueError(
            f"--llm_checkpoint={args.llm_checkpoint} needs the weights on disk to build a shadow "
            f"dir, but {args.checkpoint_path} is not a directory. Bake or download the weights first."
        )
    # Symlink targets must be absolute (relative ones resolve against the shadow dir).
    abs_model_dir = os.path.abspath(model_dir)
    src = os.path.join(abs_model_dir, args.llm_checkpoint)
    if not os.path.exists(src):
        raise ValueError(f"LM checkpoint {src} not found.")
    shadow = os.path.join(args.shadow_dir, os.path.basename(abs_model_dir) + "_" + args.llm_checkpoint)
    os.makedirs(shadow, exist_ok=True)
    for name in os.listdir(abs_model_dir):
        # Everything is symlinked through as-is except the LM, which is aliased to `llm.pt`.
        if name in ("llm.pt", args.llm_checkpoint):
            continue
        link = os.path.join(shadow, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(abs_model_dir, name), link)
    # Re-link unconditionally so a stale alias is never reused.
    link = os.path.join(shadow, "llm.pt")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(src, link)
    print(f"Using LM checkpoint '{args.llm_checkpoint}' via shadow dir {shadow}")
    return shadow


def main(args):
    # Set seed for reproducibility (the LM uses RAS sampling and the flow head is a CFM sampler).
    seed = args.seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True

    model = AutoModel(model_dir=_resolve_model_dir(args), fp16=args.fp16)
    sampling_rate = int(model.sample_rate)  # Fun-CosyVoice3: 24 kHz
    print(
        f"Loaded {args.model_id} (llm={args.llm_checkpoint}, sr={sampling_rate}, fp16={args.fp16}, "
        f"speed={args.speed}, text_frontend={not args.no_text_frontend})"
    )

    # Parameter count: nn.Modules hang off `model.model` (.llm/.flow/.hift), a plain class, so walk
    # both levels.
    try:
        print_model_size(model, getattr(model, "model", None))
    except Exception as e:
        print(f"Could not determine model size: {e}")

    # Flat, bucket-friendly layout, per model:
    #   results/<model_safe>/  MODEL_<safe>_DATASET_<dsid>.jsonl  +  <dsid>/output_<id>.wav
    # Manifest paths are relative to model_dir, so downstream stages resolve them wherever it is copied.
    # The base and RL LMs share one model id, so the RL run gets its own suffix (as do voice modes).
    # Must match resolve_mode() in submit_jobs.sh.
    variant_suffix = "" if args.llm_checkpoint == "llm.pt" else "_rl"
    mode_suffix = f"{variant_suffix}{'_voice_clone' if args.voice_clone else ''}"
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir

    # The frontend opens the reference as a file, so prompts are written here (under `results/`
    # so stage 3 can read them back to score speaker similarity).
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
    n_clipped = 0  # prompts truncated to PROMPT_MAX_SECONDS (reported at the end)
    n_frontend_fallback = 0  # samples re-synthesized without the text frontend (see synth_one)
    # Fixed-voice mode: one reference for the whole split, filled in after the dataset is loaded.
    fixed_prompt = {"full": None, "text": None}

    def write_prompt(path, prompt_audio):
        """Write one cloning reference, enforcing the frontend's duration/sample-rate asserts."""
        nonlocal n_clipped
        array = np.asarray(prompt_audio["array"], dtype=np.float32).reshape(-1)
        sr = int(prompt_audio["sampling_rate"])
        # load_wav() asserts sr >= 16 kHz whenever it resamples.
        if sr < PROMPT_MIN_SR:
            import librosa  # a cosyvoice dependency; imported lazily since this is the rare path

            array = librosa.resample(array, orig_sr=sr, target_sr=PROMPT_MIN_SR)
            sr = PROMPT_MIN_SR
        # _extract_speech_token() asserts <= 30 s; some in-the-wild references exceed it.
        max_samples = int(PROMPT_MAX_SECONDS * sr)
        if len(array) > max_samples:
            array = array[:max_samples]
            n_clipped += 1
        sf.write(path, array, sr)

    def synth_one(text, prompt_text, prompt_path):
        """Synthesize one utterance, concatenating the per-sentence chunks the generator yields."""
        nonlocal n_frontend_fallback
        # Re-seed per sample so sampling is reproducible regardless of batch layout.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        def infer(text_frontend):
            return [
                out["tts_speech"]
                for out in model.inference_zero_shot(
                    text,
                    prompt_text,
                    prompt_path,
                    stream=False,
                    speed=args.speed,
                    text_frontend=text_frontend,
                )
            ]

        try:
            chunks = infer(not args.no_text_frontend)
        except Exception as e:
            # The English wetext normalizer crashes on some non-English inputs (module docstring,
            # item 4). It runs before the first yield, so retrying with the frontend off is clean.
            n_frontend_fallback += 1
            print(
                f"Warning: text frontend raised {type(e).__name__}; retrying with "
                f"text_frontend=False for text: {text[:80]}"
            )
            chunks = infer(False)
        if not chunks:
            print(f"Warning: no audio generated for text: {text[:60]}...")
            return np.zeros(int(0.1 * sampling_rate), dtype=np.float32)
        # One chunk per sentence.
        audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        return np.asarray(audio.detach().to(torch.float32).cpu()).reshape(-1)

    def generate_tts(batch):
        """Synthesize a minibatch of target texts (per-sample loop); time the batch for RTFx."""
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)

        if args.voice_clone:
            # Write the per-sample prompts before timing (I/O is not generation). The relative path
            # goes into the manifest for stage 3 (SIM).
            prompt_rel_paths, prompt_full_paths = [], []
            for sample_id, prompt_audio in zip(batch["id"], batch["prompt_audio"]):
                prel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
                ppath = os.path.join(model_dir, prel)
                if not os.path.exists(ppath):
                    write_prompt(ppath, prompt_audio)
                prompt_rel_paths.append(prel)
                prompt_full_paths.append(ppath)
            raw_prompt_texts = list(batch["prompt_text"])
        else:
            # Fixed-voice mode: the same reference for every sample; no SIM.
            prompt_rel_paths = [None] * minibatch_size
            prompt_full_paths = [fixed_prompt["full"]] * minibatch_size
            raw_prompt_texts = [fixed_prompt["text"]] * minibatch_size
        # CosyVoice3 wants the instruct prefix on the PROMPT text.
        prompt_texts = [f"{args.instruct_prefix}{t}" for t in raw_prompt_texts]

        # START TIMING (TTS generation for the whole minibatch)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # No batched API (inference_zero_shot takes one string): loop per sample.
        wavs = [
            synth_one(text, prompt_text, prompt_path)
            for text, prompt_text, prompt_path in zip(texts_to_generate, prompt_texts, prompt_full_paths)
        ]

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(wavs, batch["id"]):
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        batch["gen_audio_filepath"] = gen_paths
        batch["prompt_audio_filepath"] = prompt_rel_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    # The reference columns are kept in both modes (fixed-voice mode may take its one reference
    # from the split); they are dropped below when not cloning.
    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio"))

    # A reference is mandatory in both modes, so the dataset must carry one unless fixed-voice mode
    # was given an external reference.
    needs_dataset_prompt = args.voice_clone or not args.prompt_audio_path
    missing = [c for c in ("prompt_text", "prompt_audio") if c not in dataset.column_names]
    if missing and needs_dataset_prompt:
        raise ValueError(
            f"Dataset {args.dataset_path}/{args.dataset}/{args.split} is missing {missing}. "
            "Fun-CosyVoice3 has no built-in speaker (no spk2info.pt ships with the weights), so a "
            "reference audio is required in every mode. Pass --prompt_audio_path/--prompt_text to "
            "supply one externally."
        )

    # ── Fixed-voice mode: resolve the ONE reference used for the whole split ──
    if not args.voice_clone:
        fixed_rel = os.path.join(dataset_dir_name, "prompts", "fixed_prompt.wav")
        fixed_full = os.path.join(model_dir, fixed_rel)
        if args.prompt_audio_path:
            if not args.prompt_text:
                raise ValueError("--prompt_audio_path requires --prompt_text (the reference's transcript).")
            audio, sr = sf.read(args.prompt_audio_path, dtype="float32", always_2d=True)
            write_prompt(fixed_full, {"array": audio.mean(axis=1), "sampling_rate": sr})
            fixed_prompt["text"] = args.prompt_text
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
            fixed_prompt["text"] = row["prompt_text"]
            src = f"{args.dataset}/{args.split} sample #{args.fixed_prompt_index} (id={row.get('id')})"
        fixed_prompt["full"] = fixed_full
        print(
            f"Fixed-voice mode: one reference for all {len(dataset)} samples, from {src}. "
            f"SIM is skipped (no per-sample reference)."
        )

    # Fixed-voice mode: drop the reference columns so nothing else triggers audio decoding.
    if not args.voice_clone:
        dataset = dataset.remove_columns([c for c in ("prompt_text", "prompt_audio") if c in dataset.column_names])

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # One request at a time with stream=True, so the first-chunk timestamp is real. Returns before
    # the main loop, so it never affects WER/RTFx. See scripts/ttfa_probe.py.
    if args.ttfa_probe != 0:
        # Imported here: the probe module is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        chunk_counts = []  # chunks per utterance
        # Pristine value, read before anything has generated (see the reset in _gen).
        token_hop_len_init = getattr(model.model, "token_hop_len", None)

        def _gen(job, early_stop):
            """One request with stream=True, timestamping the first chunk."""
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # CosyVoice2Model.tts() grows self.token_hop_len on every yield and never restores it,
            # so later requests would stream nothing. Reset it so each request starts fresh.
            if token_hop_len_init is not None:
                model.model.token_hop_len = token_hop_len_init
            first, chunks = None, []
            for out in model.inference_zero_shot(
                job[args.text_column][0],
                f"{args.instruct_prefix}{job['_prompt_text']}",
                job["_prompt_path"],
                stream=True,
                speed=args.speed,
                text_frontend=not args.no_text_frontend,
            ):
                chunk = out["tts_speech"].reshape(-1).detach().to(torch.float32).cpu()
                if first is None:
                    first = time.perf_counter()
                chunks.append(chunk)
                if early_stop:
                    break
            chunk_counts.append(len(chunks))
            if not chunks:
                return 0.0, sampling_rate, None
            return torch.cat(chunks).numpy(), sampling_rate, first

        columns = dataset.column_names
        jobs = []
        for i in sample_indices(args.ttfa_probe, len(dataset)):
            job = {c: [dataset[i][c]] for c in columns}
            if args.voice_clone:
                ref = os.path.join(output_dir, "prompts", f"prompt_{dataset[i]['id']}.wav")
                write_prompt(ref, dataset[i]["prompt_audio"])
                job["_prompt_path"], job["_prompt_text"] = ref, dataset[i]["prompt_text"]
            else:
                # Fixed-voice mode: the one reference the eval uses for the whole split.
                job["_prompt_path"], job["_prompt_text"] = fixed_prompt["full"], fixed_prompt["text"]
            jobs.append(job)

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            # token_hop_len sets the streaming granularity, i.e. the floor under TTFA.
            extra={"device": args.device, "note": "driven at batch size 1, stream=True",
                   "chunks_per_utterance": chunk_counts,
                   "token_hop_len": token_hop_len_init},
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
    if n_clipped:
        print(f"NOTE: {n_clipped} prompt(s) were longer than {PROMPT_MAX_SECONDS:.0f}s and were clipped "
              "(the frontend hard-asserts this limit).")
    if n_frontend_fallback:
        print(f"NOTE: {n_frontend_fallback} sample(s) crashed the (English) text frontend and were "
              "synthesized from raw text instead — see the warnings above.")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        help="Model id, used for the results folder / manifest name (and as the download fallback).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/opt/models/Fun-CosyVoice3-0.5B-2512",
        help="Local dir with the weights (baked at image build); falls back to --model_id on the Hub.",
    )
    parser.add_argument(
        "--llm_checkpoint",
        type=str,
        default="llm.pt",
        choices=["llm.pt", "llm.rl.pt"],
        help="Which LM checkpoint to load: 'llm.pt' (base) or 'llm.rl.pt' (the RL model, a separate "
        "row in the model card's results table). The RL run gets an '_rl' manifest suffix.",
    )
    parser.add_argument(
        "--shadow_dir",
        type=str,
        default="/tmp",
        help="Where to build the symlink shadow dir used to select a non-default --llm_checkpoint.",
    )
    # --device is for our timing calls only (CosyVoiceModel hardcodes 'cuda'), so it must name the
    # first visible GPU. --batch_size is only the manifest chunk size (generation is per sample).
    add_common_args(parser, batch_size=32, voice_clone=False)
    parser.add_argument("--fp16", action="store_true", help="Run the flow/hift stack in fp16 (default: fp32).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking-rate multiplier (1.0 = unchanged).")
    parser.add_argument(
        "--no_text_frontend",
        action="store_true",
        help="Disable cosyvoice's text frontend (wetext normalization + sentence splitting). Left ON "
        "by default to match the library default.",
    )
    parser.add_argument(
        "--instruct_prefix",
        type=str,
        default=INSTRUCT_PREFIX,
        help="Instruct-style prefix prepended to the PROMPT text, as CosyVoice3 expects.",
    )
    # --voice_clone / --no-voice_clone (default): see the module docstring. A reference is required
    # either way: CosyVoice3 has no reference-free entry point.
    parser.add_argument(
        "--fixed_prompt_index",
        type=int,
        default=0,
        help="--no-voice_clone only: which sample of the split supplies the single reference. Taking "
        "it from the split itself keeps the voice language-matched.",
    )
    parser.add_argument(
        "--prompt_audio_path",
        type=str,
        default=None,
        help="--no-voice_clone only: use this wav as the fixed reference instead of a dataset sample "
        "(requires --prompt_text). Lets one voice be held constant across languages.",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="Transcript of --prompt_audio_path (CosyVoice3 conditions on the reference's text).",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
