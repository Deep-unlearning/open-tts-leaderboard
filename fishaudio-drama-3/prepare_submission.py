"""Validate a complete local Drama 3 evaluation and prepare an unexecuted dry run.

--results_dir is the model folder named fishaudio-drama-3, containing manifests
and .run.json receipts. It is not the parent results directory. This script only
reads local artifacts and writes submission_plan.json beneath --output_dir.
"""

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

from adapter import CODEC_ID, CODEC_REVISION, MODEL_ID, MODEL_REVISION, SERVER_REVISION

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SAFE = MODEL_ID.replace("/", "-")
MODEL_LICENSE = f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/LICENSE"
SEED_LANGUAGES = ("en", "zh")
CV3_LANGUAGES = ("en", "zh", "fr", "ja", "ko", "de", "es", "it", "ru")
EXPECTED_SPLITS = {
    **{("seed_tts_eval", "tts", language): "seed_tts" for language in SEED_LANGUAGES},
    **{("cv3_eval", "zero_shot", language): "cv3" for language in CV3_LANGUAGES},
}
EXPECTED_ROW_COUNTS = {
    ("seed_tts_eval", "tts", "en"): 1088,
    ("seed_tts_eval", "tts", "zh"): 2020,
    **{("cv3_eval", "zero_shot", language): 500 for language in CV3_LANGUAGES},
}
PINS = {
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "server_revision": SERVER_REVISION, "codec_id": CODEC_ID,
    "codec_revision": CODEC_REVISION,
}
SIM_SUFFIX = "_wavlm_seed_tts"
SIM_MODEL = "wavlm_large_finetune+ecapa_tdnn (wavlm_large_finetune.pth)"
XVECTOR_MODEL = "microsoft/wavlm-base-plus-sv"
AUDITED_NORMALIZER_AGGREGATE = "e13acd2491569e5e2d6a8f3b2530f5945073e56695263f4ec712ce496562c4b1"
ROW_FIELDS = ("audio_filepath", "duration", "time", "text", "pred_text", "prompt_audio_filepath")


def normalizer_aggregate_sha256(root):
    files = sorted((root / "normalizer").glob("*.py"))
    if not files:
        raise ValueError("Current repository has no normalizer/*.py files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def git_identity(root, revision):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", revision],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def finite_number(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def verify_completed_asr(path, rows):
    """A completed ASR deletion must not be confused with unscored generation."""
    stem = path.stem.removesuffix(SIM_SUFFIX)
    proof_path = path.parent / "scoring_receipts" / (stem + ".asr.json")
    if not proof_path.is_file():
        raise ValueError(f"{path.name}: blank ASR predictions require a successful ASR receipt")
    proof = json.loads(proof_path.read_text())
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row["audio_filepath"], row["pred_text"]], ensure_ascii=False).encode() + b"\n")
    source = REPO_ROOT / "transformers" / "transcribe.py"
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    if (not isinstance(proof, dict) or proof.get("status") != "complete" or proof.get("stage") != "asr"
            or proof.get("manifest") != stem + ".jsonl" or proof.get("rows") != len(rows)
            or proof.get("asr_model_id") != "Qwen/Qwen3-ASR-1.7B-hf"
            or not re.fullmatch(r"[0-9a-f]{40}", proof.get("asr_model_resolved_sha", ""))
            or not re.fullmatch(r"[0-9a-f]{40}", proof.get("source_revision", ""))
            or source_sha256 is None or proof.get("asr_source_sha256") != source_sha256
            or proof.get("source_sha256") != source_sha256
            or proof.get("ordered_prediction_sha256") != digest.hexdigest()):
        raise ValueError(f"{path.name}: successful ASR receipt does not match source, rows or predictions")


def read_manifest(path, expected_count, voice_clone, sim_model=None):
    rows = {}
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError(f"{path.name}:{number}: invalid or blank JSON row") from None
            if not isinstance(row, dict):
                raise TypeError(f"{path.name}:{number}: manifest row must be an object")
            audio = row.get("audio_filepath")
            if not isinstance(audio, str) or not audio.strip() or audio in rows:
                raise ValueError(f"{path.name}:{number}: missing or duplicate audio_filepath")
            if not isinstance(row.get("text"), str) or not row["text"].strip() or not isinstance(row.get("pred_text"), str):
                raise ValueError(f"{path.name}:{number}: reference text and string ASR pred_text are required")
            if any(not finite_number(row.get(field), positive=True) for field in ("duration", "time")):
                raise ValueError(f"{path.name}:{number}: duration/time must be positive finite numbers")
            if voice_clone and (not isinstance(row.get("prompt_audio_filepath"), str)
                                or not row["prompt_audio_filepath"].strip()):
                raise ValueError(f"{path.name}:{number}: missing cloning reference path")
            if sim_model and (not finite_number(row.get("sim")) or row.get("sim_model") != sim_model):
                raise ValueError(f"{path.name}:{number}: numeric SIM from {sim_model} is required")
            rows[audio] = row
    if len(rows) != expected_count:
        raise ValueError(f"{path.name}: row count differs from the completed run receipt")
    if any(not row["pred_text"].strip() for row in rows.values()):
        verify_completed_asr(path, list(rows.values()))
    return rows


def publisher_candidates(model_dir, voice_clone):
    """Mirror the publisher's selected mode so an extra manifest cannot slip through."""
    candidates = set()
    for path in model_dir.rglob(f"MODEL_{MODEL_SAFE}_DATASET_*.jsonl"):
        stem = path.stem.removesuffix(SIM_SUFFIX)
        if (any(marker in stem for marker in ("seed_tts_eval_tts_", "cv3_eval_zero_shot_"))
                and stem.endswith("_voice_clone") == voice_clone):
            if path.parent != model_dir:
                raise ValueError("Nested benchmark manifests could alter the publisher's recursive scoring; isolate results")
            candidates.add(stem + ".jsonl")
    return candidates


def prepare_plan(results_dir, output_dir, audited_sha256, voice_clone=False):
    if not isinstance(audited_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", audited_sha256):
        raise ValueError("--normalizer_sha256 must be the audited 64-character SHA-256")
    audited_sha256 = audited_sha256.lower()
    actual_sha256 = hashlib.sha256((REPO_ROOT / "normalizer" / "normalizer.py").read_bytes()).hexdigest()
    if audited_sha256 != actual_sha256:
        raise ValueError("Audited normalizer SHA-256 differs from the current repository; submission is blocked")
    aggregate_sha256 = normalizer_aggregate_sha256(REPO_ROOT)
    if aggregate_sha256 != AUDITED_NORMALIZER_AGGREGATE:
        raise ValueError("Normalizer aggregate differs from the adopted audit; revalidate changed normalizer helpers before submission")
    model_dir = Path(results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not model_dir.is_dir() or model_dir.name != MODEL_SAFE:
        raise ValueError(f"--results_dir must be the existing model folder named {MODEL_SAFE}")
    suffix = "_voice_clone" if voice_clone else ""
    selected = {}
    for receipt_path in sorted(model_dir.glob("*.run.json")):
        with receipt_path.open(encoding="utf-8") as source:
            receipt = json.load(source)
        if not isinstance(receipt, dict) or not isinstance(receipt.get("settings"), dict):
            raise TypeError(f"{receipt_path.name}: invalid run receipt")
        settings = receipt["settings"]
        if not isinstance(settings.get("voice_clone"), bool):
            raise TypeError(f"{receipt_path.name}: receipt must declare its voice_clone mode")
        if settings["voice_clone"] != voice_clone:
            continue
        if any(settings.get(name) != expected for name, expected in PINS.items()):
            raise ValueError(f"{receipt_path.name}: model, server, or codec identity differs from the pinned release")
        limit = settings.get("max_eval_samples", "missing")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit > 0):
            raise ValueError(f"{receipt_path.name}: smoke/subsampled runs cannot be submitted")
        count = settings.get("expected_samples")
        if (not isinstance(count, int) or isinstance(count, bool) or count <= 0
                or not isinstance(receipt.get("completed_samples"), int)
                or isinstance(receipt.get("completed_samples"), bool)
                or receipt["completed_samples"] != count or receipt.get("status") != "complete"):
            raise ValueError(f"{receipt_path.name}: receipt does not certify a completed full split")
        dataset_path = settings.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError(f"{receipt_path.name}: missing dataset_path")
        identity = (dataset_path.rsplit("/", 1)[-1], settings.get("dataset"), settings.get("split"))
        if identity not in EXPECTED_SPLITS or identity in selected:
            raise ValueError(f"{receipt_path.name}: unexpected or duplicate benchmark/config/split")
        expected_count = EXPECTED_ROW_COUNTS[identity]
        if count != expected_count:
            raise ValueError(
                f"{receipt_path.name}: canonical split {'/'.join(identity)} requires "
                f"{expected_count} rows; receipt certifies {count}"
            )
        stem = (f"MODEL_{MODEL_SAFE}_DATASET_{dataset_path.replace('/', '-')}_"
                f"{settings['dataset']}_{settings['split']}{suffix}")
        if receipt_path.name != stem + ".run.json":
            raise ValueError(f"{receipt_path.name}: filename does not match its run settings")
        manifest_path = model_dir / (stem + ".jsonl")
        legacy_model = XVECTOR_MODEL if voice_clone and identity[0] == "seed_tts_eval" else None
        rows = read_manifest(manifest_path, count, voice_clone, legacy_model)
        record = {
            "target": EXPECTED_SPLITS[identity] + suffix,
            "dataset_path": dataset_path, "dataset": settings["dataset"], "split": settings["split"],
            "receipt": receipt_path.name, "manifest": manifest_path.name,
            "expected_samples": count, "completed_samples": receipt["completed_samples"],
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        if not voice_clone and (model_dir / (stem + SIM_SUFFIX + ".jsonl")).exists():
            raise ValueError("Unexpected fixed-voice SIM fork would override a validated source manifest")
        if voice_clone:
            sim_path = model_dir / (stem + SIM_SUFFIX + ".jsonl")
            sim_rows = read_manifest(sim_path, count, True, SIM_MODEL)
            if rows.keys() != sim_rows.keys() or any(
                any(row.get(field) != sim_rows[audio].get(field) for field in ROW_FIELDS)
                for audio, row in rows.items()
            ):
                raise ValueError(f"{sim_path.name}: SIM fork differs from its source manifest")
            record.update(sim_manifest=sim_path.name,
                          sim_manifest_sha256=hashlib.sha256(sim_path.read_bytes()).hexdigest())
        selected[identity] = record
    missing = set(EXPECTED_SPLITS) - selected.keys()
    if missing:
        missing_names = ", ".join("/".join(identity) for identity in sorted(missing))
        raise ValueError(f"Missing completed full-split receipts: {missing_names}")
    validated_names = {record["manifest"] for record in selected.values()}
    if publisher_candidates(model_dir, voice_clone) != validated_names:
        raise ValueError("Publisher would select unvalidated/orphan manifests; isolate complete full-split results")
    targets = ["seed_tts" + suffix, "cv3" + suffix]
    metadata = {"license": MODEL_LICENSE, "voice_cloning": "yes", "batch_inference": "yes",
                "num_languages": "83", "model_size_b": "35.899643008", "transformers": "no"}
    command = [sys.executable, str(REPO_ROOT / "scripts" / "open_results_pr.py"),
               "--model_id", MODEL_ID, "--skip_sync", "--local_dir", str(model_dir.parent),
               "--out_dir", str(output_dir / "csv"), "--targets", *targets,
               "--sim_backend", "wavlm_seed_tts"]
    for name, value in metadata.items():
        command.extend(["--" + name, value])
    payload = {
        "status": "local_preflight_passed", "publisher_executed": False,
        "model": {**PINS, **metadata}, "voice_clone": voice_clone, "targets": targets,
        "results_dir": str(model_dir), "normalizer_sha256": actual_sha256,
        "normalizer_hash_file": "normalizer/normalizer.py",
        "normalizer_aggregate_sha256": aggregate_sha256,
        "normalizer_aggregate_algorithm": "SHA-256 of sorted normalizer/*.py filename + NUL + file bytes",
        "normalizer_committed_tree": git_identity(REPO_ROOT, "HEAD:normalizer"),
        "checkout_head": git_identity(REPO_ROOT, "HEAD"),
        "validated_runs": [selected[identity] for identity in EXPECTED_SPLITS],
        "publisher_argv": command, "publisher_shell_command": shlex.join(command),
        "notes": [
            "This plan contains no scores and does not execute scoring, network calls, or publication.",
            "The prepared command is a dry run without --open_pr; do not add publication flags without approval.",
            "--skip_sync only skips result synchronization: the generic publisher still scores locally and fetches current private leaderboard CSVs, requiring network and authorized access even in dry-run mode.",
            "Smoke/subsampled runs cannot be submitted; all 11 full splits in the selected mode were checked.",
            "Re-run this preflight immediately before the future dry run so the audited normalizer and input artifacts are checked again.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "submission_plan.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True,
                        help="Model folder named fishaudio-drama-3 containing local manifests and .run.json receipts.")
    parser.add_argument("--output_dir", required=True, help="Local destination for submission_plan.json.")
    parser.add_argument("--normalizer_sha256", required=True,
                        help="Audited SHA-256 of normalizer/normalizer.py (the aggregate audit is also enforced).")
    parser.add_argument("--voice_clone", action="store_true", help="Validate and select the voice-clone targets.")
    args = parser.parse_args()
    try:
        destination = prepare_plan(args.results_dir, args.output_dir, args.normalizer_sha256, args.voice_clone)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Local plan prepared: {destination}")
    print("Publisher command was not executed; its future dry run needs private CSV access.")


if __name__ == "__main__":
    main()
