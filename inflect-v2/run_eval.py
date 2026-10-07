"""
TTS synthesis for the Open TTS Leaderboard (Inflect v2 backend, stage 1).

Covers the Inflect **v2** family — `owensong/Inflect-Micro-v2` (9.4M parameters) and
`owensong/Inflect-Nano-v2` (4.0M) — which share one codebase and one public API. Both are
fixed-voice English VITS models at 24 kHz: no reference conditioning, so there is no voice-clone
mode and no SIM stage.

`owensong/Inflect-Nano-v1` is a different architecture with its own backend (`inflect-nano/`).

The inference code lives *inside each model repo* (`inference.py` + `runtime/`), so this script
snapshot_downloads the repo and loads `InflectTTS` from it by path. Generation is one sample at a
time — `synthesize()` takes a single string, and the model has no batched API.

Stage 2 (`transformers/transcribe.py`, run in the shared `bezzam/evals` image) transcribes the
wavs with Qwen3-ASR to fill `pred_text`, and stage 4 (`normalizer.eval_utils.score_results`, run
locally) computes WER + RTFx from the completed manifest.
"""

import argparse
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)


def _load_engine(ckpt_dir: Path, device: str):
    """Import `InflectTTS` from the model repo's own inference.py and instantiate it.

    Loaded by file path: inference.py puts its `runtime/` dir on sys.path and imports generic
    module names (`commons`, `utils`, ...), so only one Inflect v2 model can be loaded per process.
    """
    entry = ckpt_dir / "inference.py"
    if not entry.is_file():
        raise FileNotFoundError(f"{entry} not found — is this an Inflect v2 model repo?")
    spec = importlib.util.spec_from_file_location("inflect_v2_inference", entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    # runtime/utils.py sets the ROOT logger to DEBUG at import (flooding output with httpx logs),
    # so restore the previous level.
    root_level = logging.getLogger().level
    try:
        spec.loader.exec_module(module)
        engine = module.InflectTTS(ckpt_dir, device=device)
    finally:
        logging.getLogger().setLevel(root_level)
    return engine


def main(args):
    from huggingface_hub import snapshot_download

    # The repo carries the inference code as well as the weights. local_dir= is required:
    # inference.py finds `runtime/` via `Path(__file__).resolve().parent`, which would follow the
    # HF cache's symlinks into blobs/.
    if args.checkpoint_path:
        ckpt_dir = Path(args.checkpoint_path)
    else:
        ckpt_dir = Path(snapshot_download(args.model_id, local_dir=Path(args.local_dir) / args.model_id.replace("/", "-")))
    print(f"Checkpoint: {ckpt_dir}")
    if not (ckpt_dir / "runtime").is_dir():
        raise RuntimeError(
            f"{ckpt_dir}/runtime is missing — inference.py resolves its imports relative to its own "
            "file, so the checkpoint must be a real directory tree, not a symlinked HF cache snapshot."
        )

    engine = _load_engine(ckpt_dir, args.device)
    sampling_rate = engine.sample_rate  # 24000
    print(f"Loaded Inflect v2 ({args.model_id}, sr={sampling_rate}, device={args.device})")
    print(
        f"TTS model size: {engine.deployed_parameters / 1e9:.4f}B parameters deployed "
        f"({engine.deployed_parameters:,}; {engine.checkpoint_parameters:,} in the checkpoint "
        "before weight-norm collapsing)"
    )

    is_cuda = str(args.device).startswith("cuda")

    def synth_one(text):
        """Synthesize one text; return (audio_1d_numpy, elapsed_s). Deterministic given `seed`."""
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
        start = time.perf_counter()
        _, audio = engine.synthesize(
            text, speed=args.speed, variation=args.variation, seed=args.seed
        )
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
        return np.asarray(audio, dtype=np.float32).reshape(-1), time.perf_counter() - start

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # No streaming API, so `first` is None and the probe records the whole-utterance time.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        run_probe(
            lambda job: (synth_one(job["text"])[0], sampling_rate, None),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            extra={"device": args.device, "seed": args.seed},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(path, audio, sampling_rate)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="owensong/Inflect-Micro-v2",
                        help="Inflect v2 model repo (Micro-v2 or Nano-v2); also names the manifest.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local Inflect v2 repo dir. Default: download --model_id under --local_dir.")
    parser.add_argument("--local_dir", type=str, default="/tmp/inflect_v2_models",
                        help="Where to materialise the model repo as real files (not cache symlinks).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking rate (0.5-2.0).")
    parser.add_argument("--variation", type=float, default=0.667,
                        help="Sampling noise scale (0.0-1.0); upstream's default is 0.667.")
    add_common_args(parser)
    # Seed: upstream default 0. Generation is deterministic given it.
    parser.set_defaults(seed=0)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
