"""
ASR transcription for the Open TTS Leaderboard (stage 2).

Fills `pred_text` in a stage-1 manifest with Qwen3-ASR, in place and resumably. The ASR call
mirrors `transcribe()` in `transformers/run_eval.py`.
"""

import argparse
import json
import os

import torch
from transformers.audio_utils import load_audio

from transformers import AutoModelForMultimodalLM, AutoProcessor

DEFAULT_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"


def main(args):
    torch_dtype = getattr(torch, args.dtype)

    # ── Read manifest ────────────────────────────────────────────────────────
    with open(args.manifest_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not entries:
        raise ValueError(f"No entries found in manifest {args.manifest_path}")
    print(f"Transcribing {len(entries)} samples from {args.manifest_path}")

    # Wav paths are relative to the manifest's directory; absolute or unresolvable paths are used as-is.
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest_path))

    def _resolve(p):
        if not p or os.path.isabs(p):
            return p
        cand = os.path.join(manifest_dir, p)
        return cand if os.path.exists(cand) else p

    # ── Resume: decide what's left BEFORE paying for the ASR model ───────────
    # Only transcribe rows without a prediction (unless --overwrite). Checked before the model load
    # so re-running a finished split is ~free.
    todo = list(range(len(entries)))
    if not args.overwrite:
        todo = [i for i in todo if not entries[i].get("pred_text")]
    if len(todo) < len(entries):
        print(f"Resuming: {len(entries) - len(todo)} already transcribed, {len(todo)} remaining.")
    if not todo:
        print("Nothing to transcribe; manifest already complete:", os.path.abspath(args.manifest_path))
        return

    # ── ASR model ─────────────────────────────────────────────────────────────
    asr_processor = AutoProcessor.from_pretrained(args.asr_model_id)
    asr_model = AutoModelForMultimodalLM.from_pretrained(
        args.asr_model_id, dtype=torch_dtype, device_map=args.device
    )
    asr_model.eval()
    asr_sr = 16_000
    if getattr(asr_processor, "feature_extractor", None) is not None and hasattr(
        asr_processor.feature_extractor, "sampling_rate"
    ):
        asr_sr = asr_processor.feature_extractor.sampling_rate

    # Cap clip length: a runaway generation padded into a batch can OOM the ASR encoder, and
    # anything that long is degenerate and scores as an error anyway.
    max_samples = int(args.max_audio_seconds * asr_sr) if args.max_audio_seconds > 0 else None

    def transcribe(paths):
        speech = [load_audio(path, sampling_rate=asr_sr) for path in paths]
        if max_samples is not None:
            speech = [s[:max_samples] for s in speech]
        lang = args.asr_language or None  # '' => None => Qwen3-ASR auto-detects
        language = [lang] * len(speech)
        inputs = asr_processor.apply_transcription_request(speech, language=language).to(
            asr_model.device, asr_model.dtype
        )
        with torch.no_grad():
            output_ids = asr_model.generate(**inputs, max_new_tokens=args.asr_max_new_tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return asr_processor.decode(generated_ids, return_format="transcription_only")

    def rewrite_manifest():
        with open(args.manifest_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Batch clips of similar length together so padding=longest wastes less memory/compute, and
    # any capped long clips cluster into a few batches rather than inflating every batch.
    todo.sort(key=lambda i: entries[i].get("duration") or 0.0)

    # ── Transcribe in ASR minibatches; fill pred_text and persist after each chunk ─
    for start in range(0, len(todo), args.asr_batch_size):
        idxs = todo[start : start + args.asr_batch_size]
        preds = transcribe([_resolve(entries[i]["audio_filepath"]) for i in idxs])
        for i, pred in zip(idxs, preds):
            entries[i]["pred_text"] = pred
        rewrite_manifest()  # persist progress so a crash doesn't lose completed chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  transcribed {min(start + args.asr_batch_size, len(todo))}/{len(todo)}")

    print("Manifest updated with transcriptions:", os.path.abspath(args.manifest_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, required=True, help="JSONL manifest to transcribe (pred_text filled in place).")
    parser.add_argument("--asr_model_id", type=str, default=DEFAULT_ASR_MODEL_ID, help="ASR model used to transcribe for WER.")
    parser.add_argument(
        "--asr_language",
        type=str,
        default="en",
        help="Language hint for the ASR model. Pass '' (empty) to let Qwen3-ASR auto-detect.",
    )
    parser.add_argument("--asr_max_new_tokens", type=int, default=256, help="Max tokens for ASR transcription.")
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=30.0,
        help="Truncate each clip to at most this many seconds before transcribing (0 = no cap). "
             "Bounds ASR-encoder memory: a runaway TTS generation can be minutes long and OOM the "
             "conv frontend (a batch is padded to its longest clip). Mirrors score_similarity.py.",
    )
    parser.add_argument(
        "--asr_batch_size",
        type=int,
        default=32,
        help="Minibatch size for ASR transcription (clips are padded to the longest in the batch).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe every row even if it already has a pred_text (default: resume, skip filled rows).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="'cuda', 'cpu', or 'cuda:0'.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="ASR model dtype, e.g. 'bfloat16'.")
    args = parser.parse_args()

    main(args)
