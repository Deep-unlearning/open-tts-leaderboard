"""
TTS synthesis for the Open TTS Leaderboard (Inflect-Nano-v1 backend, stage 1).

Inflect-Nano-v1 (owensong/Inflect-Nano-v1) is a tiny (~4.6M param) FastSpeech-style
acoustic model + Snake-HiFiGAN vocoder that runs in-process from its own repo. This
script imports the repo's `inference` module (which self-registers its vendored paths on
import), loads the acoustic + vocoder checkpoints, synthesizes each target text with the
single built-in voice, times generation for RTFx, saves wavs, and writes a manifest with
`text`, `duration`, and generation `time` (with an empty `pred_text` placeholder).

The model has no batched inference API, so texts are synthesized one at a time.

Stage 2 (`transformers/transcribe.py`, run in the shared `bezzam/evals` image) transcribes the
wavs with Qwen3-ASR to fill `pred_text`, and stage 4 (`normalizer.eval_utils.score_results`, run
locally) computes WER + RTFx from the completed manifest. Fixed voice, so no stage 3 (SIM).
"""

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# Inflect-Nano outputs 24 kHz waveforms (fixed).
SAMPLING_RATE = 24_000


def main(args):
    set_seed(args.seed)

    # Make the model repo importable, then import its inference module. inference.py inserts
    # its own repo root + vendored frontend onto sys.path at import time, so this is enough.
    sys.path.insert(0, args.model_repo)
    from inference import load_acoustic, load_vocoder, synthesize, DEFAULT_ACOUSTIC, DEFAULT_VOCODER

    device = torch.device(args.device)
    acoustic_path = args.acoustic_weights or DEFAULT_ACOUSTIC
    vocoder_path = args.vocoder_weights or DEFAULT_VOCODER
    acoustic, speakers, ac_params = load_acoustic(acoustic_path, device)
    vocoder, vo_params = load_vocoder(vocoder_path, device)
    print(f"Loaded Inflect-Nano ({(ac_params + vo_params) / 1e6:.2f}M params, voices={list(speakers)})")

    is_cuda = device.type == "cuda"

    def synth_one(text):
        """Synthesize one text; return (audio, elapsed_seconds)."""
        if is_cuda:
            torch.cuda.synchronize(device=device)
        start = time.perf_counter()
        audio = synthesize(
            text, acoustic, vocoder, speakers, device,
            length_scale=args.length_scale, pitch_scale=args.pitch_scale, energy_scale=args.energy_scale,
        )
        if is_cuda:
            torch.cuda.synchronize(device=device)
        return np.asarray(audio).reshape(-1), time.perf_counter() - start

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
            lambda job: (synth_one(job["text"])[0], SAMPLING_RATE, None),
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe, extra={"device": args.device},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)  # so warm-up does not shift the RNG stream of the timed run

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(path, audio, SAMPLING_RATE, subtype="PCM_16")

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="owensong/Inflect-Nano-v1",
        help="Model id (used only for output-path / manifest naming).",
    )
    parser.add_argument(
        "--model_repo",
        type=str,
        default="/opt/Inflect-Nano-v1",
        help="Path to the cloned Inflect-Nano-v1 repo (provides the `inference` module + weights).",
    )
    parser.add_argument("--acoustic_weights", type=str, default=None, help="Override acoustic checkpoint path.")
    parser.add_argument("--vocoder_weights", type=str, default=None, help="Override vocoder checkpoint path.")
    # Prosody controls (defaults = neutral, matching the model's inference.py defaults).
    parser.add_argument("--length_scale", type=float, default=1.0, help="Duration scale (>1 = slower).")
    parser.add_argument("--pitch_scale", type=float, default=1.0, help="Pitch scale.")
    parser.add_argument("--energy_scale", type=float, default=1.0, help="Energy scale.")
    add_common_args(parser)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
