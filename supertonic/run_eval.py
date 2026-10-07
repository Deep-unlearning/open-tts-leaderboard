"""
TTS synthesis for the Open TTS Leaderboard (Supertonic 3 backend, stage 1).

Supertonic 3 (Supertone/supertonic-3) is a ~0.1B on-device TTS shipped as four ONNX graphs run
with ONNX Runtime. Batching is native (`TextToSpeech.batch` from supertone-inc/supertonic's
`helper.py`; text chunking is disabled in batch mode, fine for short eval texts). The batched
output is zero-padded, so each sample is trimmed to its predicted duration.

Fixed preset voices only (M1-M5 / F1-F5 style embeddings): the local pipeline has no zero-shot
cloning, so the dataset's prompt columns are dropped. Output is 44.1 kHz mono.

Timing: `ort.InferenceSession.run` is synchronous (outputs are on host when it returns, also
under the CUDA EP), so `time.perf_counter()` around the batch call is the full latency.

Stage 2 (`transformers/transcribe.py`) fills `pred_text` with Qwen3-ASR, and stage 4
(`normalizer.eval_utils.score_results`) computes WER + RTFx from the manifest (no stage 3 SIM:
fixed voice).
"""

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm


# The reference implementation lives in the cloned supertone-inc/supertonic repo
# (py/helper.py), baked into the image at /opt/supertonic by the Dockerfile.
sys.path.insert(0, os.environ.get("SUPERTONIC_HELPER_DIR", "/opt/supertonic/py"))
from helper import (  # noqa: E402
    TextToSpeech,
    load_cfgs,
    load_onnx_all,
    load_text_processor,
    load_voice_style,
)

from run_eval_utils import (  # noqa: E402
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

ONNX_FILES = (
    "duration_predictor.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
)


def _build_tts(onnx_dir, device):
    """helper.load_text_to_speech, but with the CUDA EP (upstream raises for use_gpu=True)."""
    opts = ort.SessionOptions()
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1]) if ":" in device else 0
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    cfgs = load_cfgs(onnx_dir)
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = load_onnx_all(onnx_dir, opts, providers)
    text_processor = load_text_processor(onnx_dir)
    tts = TextToSpeech(cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort)
    # Report the registered providers: the CUDA EP silently falls back to CPU (e.g. no cuDNN).
    for name, sess in [
        ("duration_predictor", dp_ort),
        ("text_encoder", text_enc_ort),
        ("vector_estimator", vector_est_ort),
        ("vocoder", vocoder_ort),
    ]:
        print(f"  {name}: providers={sess.get_providers()}")
    return tts


def _print_model_size(onnx_dir):
    """Sum ONNX initializer element counts across the four graphs (model size in B params)."""
    try:
        import onnx

        n_params = 0
        for fname in ONNX_FILES:
            model = onnx.load(os.path.join(onnx_dir, fname), load_external_data=False)
            for init in model.graph.initializer:
                n = 1
                for d in init.dims:
                    n *= d
                n_params += n
        print(f"TTS model size: {n_params / 1e9:.2f}B parameters (sum of ONNX initializers over {len(ONNX_FILES)} graphs)")
    except Exception as e:
        # Model card reports ~99M parameters across the public ONNX assets.
        print(f"Could not compute ONNX parameter count ({e}); model card reports ~0.10B parameters.")


def main(args):
    # Set seed for reproducibility (the vector estimator samples latent noise via numpy).
    set_seed(args.seed)

    onnx_dir = os.path.join(args.model_dir, "onnx")
    print(f"Loading Supertonic 3 from {onnx_dir} (device={args.device})...")
    tts = _build_tts(onnx_dir, args.device)
    sampling_rate = int(tts.sample_rate)  # 44100
    print(f"Loaded Supertonic 3 ({args.model_id}, sr={sampling_rate}, voice={args.voice})")
    _print_model_size(onnx_dir)

    # Fixed shipped voice style (JSON style embedding). `load_voice_style` builds a Style
    # whose leading dim must match the text batch size, so cache one per batch size
    # (only two sizes ever occur: batch_size and the final remainder).
    style_path = os.path.join(args.model_dir, "voice_styles", f"{args.voice}.json")
    if not os.path.exists(style_path):
        raise FileNotFoundError(f"Voice style not found: {style_path} (expected M1-M5/F1-F5)")
    style_cache = {}

    def get_style(bsz):
        if bsz not in style_cache:
            style_cache[bsz] = load_voice_style([style_path] * bsz)
        return style_cache[bsz]

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    def generate_tts(batch):
        """Synthesize speech for a minibatch of target texts; time the batch generation for RTFx."""
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)
        langs = [args.lang] * minibatch_size
        style = get_style(minibatch_size)  # cached; loaded outside the timed region on reuse

        # START TIMING (TTS batch generation; session.run is synchronous, see module docstring)
        start = time.perf_counter()

        # Returns zero-padded wavs (bsz, T_max) + per-sample durations (s).
        wavs, durations = tts.batch(texts_to_generate, langs, style, args.total_steps, args.speed)

        # END TIMING
        runtime = time.perf_counter() - start
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        wavs = np.asarray(wavs)
        durations = np.asarray(durations).reshape(-1)
        gen_paths, audio_length_s = [], []
        for b, sample_id in enumerate(batch["id"]):
            # Trim the zero-padding to the predicted duration (as in the upstream example).
            n_samples = int(sampling_rate * float(durations[b]))
            audio = np.asarray(wavs[b], dtype=np.float32).reshape(-1)[:n_samples]
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            path = os.path.join(model_dir, rel_path)
            sf.write(path, audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        batch["gen_audio_filepath"] = gen_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    dataset = load_tts_dataset(args)

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # TTFA is per-request, so generate_tts() is driven with one-row batches. model_dir and
    # dataset_dir_name are rebound to a temp dir (generate_tts reads them late-bound), so no
    # results are touched. No streaming API: `first` is None -> whole-utterance fallback.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        def _gen(job):
            return float(generate_tts(job)["audio_length_s"][0]), None, None

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix="", n=args.ttfa_probe,  # fixed voice: no clone mode
            extra={"device": args.device, "note": "driven at batch size 1"},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(generate_tts, dataset, starts, args.batch_size, args.warmup_steps)
    set_seed(args.seed)  # so warm-up does not shift the latent-noise stream of the timed run

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    for batch_start in tqdm(starts, desc="Generating"):
        batch = generate_tts(dataset[batch_start : batch_start + args.batch_size])

        # Append each sample to the JSONL immediately. `pred_text` is filled by stage 2 (ASR).
        for afp, alen, gtime, ref in zip(
            batch["gen_audio_filepath"], batch["audio_length_s"],
            batch["generation_time_s"], batch["references"],
        ):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="Supertone/supertonic-3",
        help="HF model id (used for manifest/output naming).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/opt/supertonic3",
        help="Local snapshot of Supertone/supertonic-3 with onnx/ + voice_styles/ (baked at image build).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="M1",
        help="Built-in voice style: one of M1-M5 / F1-F5 (JSON style embeddings shipped with the model).",
    )
    parser.add_argument("--lang", type=str, default="en", help="Language code (31 supported; 'na' = language-agnostic).")
    parser.add_argument(
        "--total_steps",
        type=int,
        default=8,
        help="Denoising steps for the vector estimator (5=fast/low to 12=high quality; model default 8).",
    )
    parser.add_argument("--speed", type=float, default=1.05, help="Speech rate (0.7 slow to 2.0 fast; model default 1.05).")
    add_common_args(parser, batch_size=32)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
