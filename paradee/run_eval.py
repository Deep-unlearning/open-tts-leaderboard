"""
TTS synthesis for the Open TTS Leaderboard (Paradee backend, stage 1).

Paradee-8M (sahilmahendrakar/Paradee-8M-v1.0, https://github.com/sahilmahendrakar/paradee) is an
8M-parameter distillation of Kokoro-82M that speaks one voice (Kokoro's af_heart), English only.
The `paradee` package runs misaki G2P (Kokoro's own) and then one ONNX graph (phoneme ids -> 24 kHz
audio) in ONNX Runtime. This script loads it once, synthesizes each target text, times generation
for RTFx, saves wavs, and writes a manifest with `text`, `duration`, and generation `time` (empty
`pred_text`).

`Paradee.__call__` splits the text into sentences and runs the graph once per sentence, then
concatenates. The graph takes a single `[1, T]` sequence, so there is no batched API: rows are
processed serially, as in the Kokoro backend.
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm

from paradee import SAMPLE_RATE, Paradee
from paradee.tts import REPO, REVISION

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

ONNX_FILES = {True: "onnx/paradee_int8.onnx", False: "onnx/paradee.onnx"}


def _onnx_path(quantized):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO, ONNX_FILES[quantized], revision=REVISION)


def _use_cuda_session(tts, onnx_path, device, threads):
    """Replace the package's CPU session with a CUDA EP one on the same graph (same SessionOptions)."""
    device_id = int(device.split(":")[1]) if ":" in device else 0
    # Load pip-installed CUDA/cuDNN (nvidia-* wheels) when they are not on LD_LIBRARY_PATH.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    tts.session = ort.InferenceSession(
        onnx_path, so, providers=[("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    )
    # The CUDA EP silently falls back to CPU (e.g. missing cuDNN); refuse rather than mislabel.
    if "CUDAExecutionProvider" not in tts.session.get_providers():
        raise RuntimeError(f"--device={device} but the CUDA EP did not load: {tts.session.get_providers()}")


def _print_model_size(onnx_path):
    """Sum the ONNX initializer element counts (int8 weights count once; scales/zero points are tiny)."""
    try:
        import onnx

        model = onnx.load(onnx_path, load_external_data=False)
        n_params = sum(int(np.prod(init.dims)) for init in model.graph.initializer)
        print(f"TTS model size: {n_params / 1e9:.3f}B parameters (sum of ONNX initializers)")
    except Exception as e:
        print(f"Could not compute ONNX parameter count ({e}); model card reports 8.07M parameters.")


def main(args):
    set_seed(args.seed)

    onnx_path = _onnx_path(args.quantized)
    tts = Paradee(quantized=args.quantized, model_path=onnx_path, threads=args.threads)
    if args.device.startswith("cuda"):
        _use_cuda_session(tts, onnx_path, args.device, args.threads)
    print(
        f"Loaded Paradee ({args.model_id}@{REVISION}, {ONNX_FILES[args.quantized]}, "
        f"providers={tts.session.get_providers()}, threads={args.threads})"
    )
    _print_model_size(onnx_path)

    def synth_one(text):
        """Synthesize one text with the package's own sentence loop. Return (audio_1d_numpy, elapsed_s)."""
        start = time.perf_counter()
        audio = tts(text, speed=args.speed)
        elapsed = time.perf_counter() - start
        if audio.size == 0:
            audio = np.zeros(1, dtype=np.float32)
        return audio, elapsed

    def synth_one_streaming(text, early_stop=False):
        """TTFA probe only: the same sentence loop as Paradee.__call__, timestamping the first chunk.

        Returns (audio_1d_numpy, first_chunk_ts, n_chunks). A one-sentence prompt has nothing to
        stream (n_chunks == 1): first audio is last audio.
        """
        first, parts = None, []
        for ps in tts._chunks(text):
            parts.append(tts.generate_from_phonemes(ps, args.speed))
            if first is None:
                first = time.perf_counter()
                if early_stop:
                    break
        if not parts:
            return np.zeros(1, dtype=np.float32), None, 0
        return np.concatenate(parts), first, len(parts)

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Timestamps the first sentence's audio (see synth_one_streaming).
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        chunk_counts = []

        def _ttfa_gen(job, early_stop):
            audio, first, n_chunks = synth_one_streaming(job["text"], early_stop)
            chunk_counts.append(n_chunks)
            return audio, SAMPLE_RATE, first

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            # segments_per_utterance == 1 means one sentence in, nothing to stream.
            extra={"device": args.device, "providers": tts.session.get_providers(), "threads": args.threads,
                   "quantized": args.quantized, "segments_per_utterance": chunk_counts},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(path, audio, SAMPLE_RATE)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLE_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default=REPO,
                        help="HF model id (manifest/output naming; the package pins the repo + revision).")
    parser.add_argument("--quantized", action=argparse.BooleanOptionalAction, default=True,
                        help="int8 graph (the package default, 'use this one' per the model card); "
                             "--no-quantized for the fp32 graph.")
    parser.add_argument("--threads", type=int, default=1,
                        help="ONNX Runtime intra-op threads. Default 1 = the package default, so the "
                             "model is benchmarked as shipped.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    add_common_args(parser)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
