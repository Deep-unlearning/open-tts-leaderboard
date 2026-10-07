"""
TTS synthesis for the Open TTS Leaderboard (VoxCPM2 backend, stage 1).

VoxCPM2 (openbmb/VoxCPM2): ~2B tokenizer-free TTS, 48 kHz output. With `--voice_clone` each
sample is cloned from the dataset's `prompt_audio`/`prompt_text` (written to wavs, since
`generate()` takes a path); by default (`--no-voice_clone`) VoxCPM2 picks its own speaker (no SIM).

`VoxCPM.generate()` takes a single text, so texts are synthesized one at a time inside
`generate_tts(batch)`; `--batch_size` only sets manifest-writing / resume granularity, and
per-sample time = batch time / batch size.
"""

import argparse
import inspect
import logging
import os
import time
import warnings

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args,
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    open_manifest,
    output_paths,
    pending_batches,
    print_model_size,
    print_next_steps,
    set_seed,
    warm_up,
    write_entry,
)

# voxcpm pulls in chatty deps (modelscope, funasr, transformers); keep warnings/info out
# of the way so the outer "Generating" tqdm is the only progress output.
warnings.filterwarnings("ignore")
for _name in ("modelscope", "funasr"):
    logging.getLogger(_name).setLevel(logging.WARNING)
try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
except Exception:
    pass

from voxcpm import VoxCPM  # noqa: E402

torch.set_float32_matmul_precision("high")


def main(args):
    # Set seed for reproducibility (VoxCPM samples internally: LM sampling + diffusion).
    seed = args.seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Load the checkpoint baked into the image, else pull from the Hub. The ZipEnhancer prompt
    # denoiser is skipped (`enable_denoiser/load_denoiser=False`).
    if os.path.isdir(args.checkpoint_path):
        init_kwargs = {"voxcpm_model_path": args.checkpoint_path, "enable_denoiser": False}
        if "device" in inspect.signature(VoxCPM.__init__).parameters:
            init_kwargs["device"] = args.device
        model = VoxCPM(**init_kwargs)
    else:
        print(f"Checkpoint dir {args.checkpoint_path} not found; downloading {args.model_id} from the Hub.")
        fp_kwargs = {"load_denoiser": False}
        if "device" in inspect.signature(VoxCPM.from_pretrained).parameters:
            fp_kwargs["device"] = args.device
        model = VoxCPM.from_pretrained(args.model_id, **fp_kwargs)
    sampling_rate = int(getattr(model.tts_model, "sample_rate", 48000))  # VoxCPM2: 48 kHz
    print(f"Loaded VoxCPM2 model {args.model_id} (sr={sampling_rate}, cfg_value={args.cfg_value}, "
          f"inference_timesteps={args.inference_timesteps})")

    # VoxCPM is a wrapper around nn.Module components: count their unique params.
    try:
        print_model_size(model)
    except Exception as e:
        print(f"Could not determine model size: {e}")

    # Layout: results/<model_safe>/MODEL_<safe>_DATASET_<dsid>.jsonl + <dsid>/output_<id>.wav.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    # Keep the two modes' audio separate so voice-clone and auto-voice runs don't overwrite.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir

    # `generate()` takes the cloning reference as a file path; prompts are written here (under
    # `results/` so stage 3 can read them back to score speaker similarity).
    prompt_dir = os.path.join(output_dir, "prompts")
    if args.voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    def generate_tts(batch):
        """Synthesize a minibatch one utterance at a time; time the whole loop for RTFx."""
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)

        # Write the cloning prompts to wavs BEFORE timing (I/O is not generation).
        if args.voice_clone:
            prompt_rel_paths, prompt_full_paths, prompt_texts = [], [], list(batch["prompt_text"])
            for sample_id, prompt_audio in zip(batch["id"], batch["prompt_audio"]):
                prel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
                ppath = os.path.join(model_dir, prel)
                if not os.path.exists(ppath):
                    sf.write(ppath, np.asarray(prompt_audio["array"], dtype=np.float32), prompt_audio["sampling_rate"])
                prompt_rel_paths.append(prel)
                prompt_full_paths.append(ppath)
        else:
            prompt_rel_paths = [None] * minibatch_size
            prompt_full_paths = [None] * minibatch_size
            prompt_texts = [None] * minibatch_size

        # START TIMING (TTS generation for the whole minibatch)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        wavs = []
        for text, prompt_text, prompt_path in zip(texts_to_generate, prompt_texts, prompt_full_paths):
            # Re-seed per sample so generation is reproducible regardless of batch layout.
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Voice-cloning mode passes the per-sample reference audio/text; auto-voice mode
            # (prompt_wav_path/prompt_text=None) lets VoxCPM2 pick its own speaker.
            wav = model.generate(
                text=text,
                prompt_wav_path=prompt_path,
                prompt_text=prompt_text,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                normalize=args.normalize,
                denoise=False,  # denoiser not loaded (enable_denoiser=False)
                retry_badcase=not args.no_retry_badcase,
            )
            wavs.append(np.asarray(wav).reshape(-1))

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(wavs, batch["id"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = os.path.join(dataset_dir_name, f"output_{sample_id}.wav")
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        batch["gen_audio_filepath"] = gen_paths
        batch["prompt_audio_filepath"] = prompt_rel_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    # For voice cloning also keep the reference `prompt_audio`/`prompt_text`.
    dataset = load_tts_dataset(args, extra_columns=("prompt_text", "prompt_audio") if args.voice_clone else ())

    # ── TTFA probe: one request at a time via generate_streaming(); writes only a JSON sidecar ──
    # Prompt clips go to a temp dir so the results tree is untouched.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported lazily: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        output_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        chunk_counts = []  # chunks per utterance

        def _gen(job, early_stop):
            """One generation via generate_streaming() (same arguments as the eval), timestamping the first chunk."""
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            prompt_path = prompt_text = None
            if args.voice_clone:
                prompt_path = os.path.join(output_dir, "prompts", f"prompt_{job['id'][0]}.wav")
                pa = job["prompt_audio"][0]
                sf.write(prompt_path, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
                prompt_text = job["prompt_text"][0]
            first, chunks = None, []
            for chunk in model.generate_streaming(
                text=job[args.text_column][0],
                prompt_wav_path=prompt_path,
                prompt_text=prompt_text,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                normalize=args.normalize,
                denoise=False,
                retry_badcase=not args.no_retry_badcase,
            ):
                if first is None:
                    first = time.perf_counter()
                chunks.append(np.asarray(chunk).reshape(-1))
                if early_stop:
                    break
            chunk_counts.append(len(chunks))
            if not chunks:
                return np.zeros(1, dtype=np.float32), sampling_rate, None
            return np.concatenate(chunks), sampling_rate, first

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            # chunks_per_utterance distinguishes "no incremental output" (1 chunk) from late delivery.
            extra={"device": args.device, "note": "driven at batch size 1, generate_streaming()",
                   "chunks_per_utterance": chunk_counts},
        )
        return

    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(generate_tts, dataset, starts, 1, args.warmup_steps)  # serial model: warm up on single samples

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    for batch_start in tqdm(starts, desc="Generating"):
        batch = generate_tts(dataset[batch_start : batch_start + args.batch_size])

        # Append each sample to the JSONL immediately. `pred_text` is filled by stage 2 (ASR);
        # `sim` by stage 3 (speaker similarity, voice-clone only).
        for afp, ppath, alen, gtime, ref in zip(
            batch["gen_audio_filepath"], batch["prompt_audio_filepath"], batch["audio_length_s"],
            batch["generation_time_s"], batch["references"],
        ):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref, ppath if args.voice_clone else None))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser, batch_size=32, voice_clone=False)

    parser.add_argument(
        "--model_id",
        type=str,
        default="openbmb/VoxCPM2",
        help="VoxCPM model id, loadable with the `voxcpm` package.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/opt/models/VoxCPM2",
        help="Local dir with the VoxCPM2 weights (baked at image build); falls back to the Hub if missing.",
    )
    parser.add_argument("--cfg_value", type=float, default=2.0, help="LM guidance scale (model-card default 2.0).")
    parser.add_argument(
        "--inference_timesteps", type=int, default=10, help="Diffusion steps (model-card default 10; higher = quality)."
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Apply voxcpm's external text normalization (wetext) before synthesis."
    )
    parser.add_argument(
        "--no_retry_badcase",
        action="store_true",
        help="Disable voxcpm's retry-on-badcase (default: retry up to 3x when output is degenerately long).",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
