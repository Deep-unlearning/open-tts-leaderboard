"""
TTS synthesis for the Open TTS Leaderboard (Kokoro-82M backend, stage 1).

Kokoro-82M (hexgrad/Kokoro-82M) is an 82M StyleTTS2 + ISTFTNet model loaded in-process via
the `kokoro` package (`KPipeline`). This script loads the pipeline once, synthesizes each
target text with a fixed built-in voice, times generation for RTFx, saves wavs, and writes a
manifest with `text`, `duration`, and generation `time` (empty `pred_text`).

The pipeline chunks long inputs and yields one audio segment per chunk, so per row we
concatenate all yielded segments. There is no GPU-batched API, so rows are processed serially.

Stage 2 (`transformers/transcribe.py`, run in the shared `bezzam/evals` image) transcribes the
wavs with Qwen3-ASR to fill `pred_text`, and stage 4 (`normalizer.eval_utils.score_results`, run
locally) computes WER + RTFx from the completed manifest. Fixed voice, so no stage 3 (SIM).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from kokoro import KPipeline

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# Kokoro outputs 24 kHz waveforms (fixed constant, not returned by the API).
SAMPLING_RATE = 24_000


def main(args):
    set_seed(args.seed)

    # Pass repo_id explicitly to silence the default-repo warning.
    pipeline = KPipeline(lang_code=args.lang_code, repo_id=args.model_id, device=args.device)
    print(f"Loaded Kokoro ({args.model_id}, voice={args.voice}, lang_code={args.lang_code})")
    print_model_size(getattr(pipeline, "model", None))

    is_cuda = (args.device or ("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda")

    def synth_one(text):
        """Synthesize one text; concatenate all chunk segments. Return (audio_1d_numpy, elapsed_s)."""
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        segments = [audio for _gs, _ps, audio in pipeline(text, voice=args.voice, speed=args.speed)]
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if not segments:
            audio = np.zeros(1, dtype=np.float32)
        else:
            audio = torch.cat([s.reshape(-1) for s in segments]).detach().to(torch.float32).cpu().numpy()
        return audio, elapsed

    def synth_one_streaming(text, early_stop=False):
        """TTFA probe only: return (audio_1d_numpy, first_segment_ts, n_segments).

        KPipeline yields one segment per sentence, so a one-sentence prompt has nothing to stream
        (n_segments == 1): first audio is last audio.
        """
        first, segments = None, []
        n_chunks = 0
        for _gs, _ps, audio in pipeline(text, voice=args.voice, speed=args.speed):
            n_chunks += 1
            if first is None:
                # .cpu() forces the segment to be materialised, so the timestamp reflects audio
                # actually in hand rather than a queued device op.
                audio = audio.reshape(-1).detach().to(torch.float32).cpu()
                first = time.perf_counter()
                segments.append(audio)
                if early_stop:
                    break
                continue
            segments.append(audio.reshape(-1).detach().to(torch.float32).cpu())
        if not segments:
            return np.zeros(1, dtype=np.float32), None, 0
        return torch.cat(segments).numpy(), first, n_chunks

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Timestamps the first KPipeline segment (see synth_one_streaming).
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        segment_counts = []

        def _ttfa_gen(job, early_stop):
            audio, first, n_chunks = synth_one_streaming(job["text"], early_stop)
            segment_counts.append(n_chunks)
            return audio, SAMPLING_RATE, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            # segments_per_utterance == 1 means one sentence in, nothing to stream.
            extra={"device": args.device, "segments_per_utterance": segment_counts},
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
        sf.write(path, audio, SAMPLING_RATE)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="hexgrad/Kokoro-82M", help="Kokoro repo id.")
    parser.add_argument(
        "--voice",
        type=str,
        default="af_heart",
        help="Kokoro voice pack (e.g. af_heart, af_bella, am_michael). Prefix must match --lang_code.",
    )
    parser.add_argument("--lang_code", type=str, default="a", help="Kokoro G2P language code; must match the voice prefix (e.g. 'a' = American English, 'z' = Mandarin).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    add_common_args(parser)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
