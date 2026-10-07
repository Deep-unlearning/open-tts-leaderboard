"""
Score an already-generated TTS eval manifest for a given model.

Runs directly on the host, given the `normalizer/` deps from requirements.txt.

Usage:
```
python scripts/score_model.py --model_id microsoft/speecht5_tts
python scripts/score_model.py --model_id microsoft/speecht5_tts --results_dir results --language en
```

If no manifest for the model is found under --results_dir, the model's folder is first synced
(manifests only, no wavs) from the results bucket (default: hf-audio/tts_leaderboard_h200) — the
same sync `scripts/open_results_pr.py` and `submit_jobs.sh` do. Pass --sync to re-sync even when
local results exist, or --no_sync to never touch the bucket.
"""

import argparse
import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # so `from normalizer import eval_utils` resolves regardless of cwd

from normalizer import eval_utils  # noqa: E402

RESULTS_BUCKET = "hf-audio/tts_leaderboard_h200"


def has_local_results(results_dir, model_id, manifests=None):
    """Whether `results_dir` already holds manifests for `model_id` (same match as score_results)."""
    model_safe = model_id.replace("/", "-")
    found = [
        fp for fp in glob.glob(os.path.join(results_dir, "**", "*.jsonl"), recursive=True)
        if f"/{model_safe}/" in fp or f"MODEL_{model_safe}_DATASET_" in fp
    ]
    if manifests:
        wanted = {os.path.basename(m).removesuffix(".jsonl") for m in manifests}
        found = [fp for fp in found if any(os.path.basename(fp).startswith(w) for w in wanted)]
    return bool(found)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, required=True, help="Model id to score, e.g. 'microsoft/speecht5_tts'.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "results"),
        help="Directory containing manifest jsonl files (default: results at repo root).",
    )
    parser.add_argument("--language", type=str, default="en", help="Language code for normalization (e.g. 'en', 'de').")
    parser.add_argument(
        "--multilingual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply compound-word boundary normalization before scoring. Default: on for "
        "non-English alphabetic languages, off for English and CJK (scored per character).",
    )
    parser.add_argument("--csv_only", action="store_true", help="Suppress the per-dataset and composite printouts.")
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=None,
        help="Only score these manifest files (basenames or paths). --language applies to every "
        "manifest scored in one call and only CJK is auto-detected, so a folder holding several "
        "languages must be scored one language at a time, naming just that language's manifests.",
    )
    parser.add_argument(
        "--sim_backend",
        type=str,
        default="wavlm_seed_tts",
        choices=["xvector", "wavlm_seed_tts"],
        help="Which SIM manifest family to score. 'wavlm_seed_tts' (default) reads the "
        "_wavlm_seed_tts.jsonl forks the seed-tts-eval SIM model writes, falling back to the plain "
        "manifest (WER/RTFx only) where none exists; 'xvector' reads only the plain manifests. Only "
        "one family per call — the two SIM scales must not be averaged together.",
    )
    parser.add_argument("--bucket", type=str, default=RESULTS_BUCKET, help="Bucket to sync missing results from.")
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument("--sync", action="store_true", help="Sync from the bucket even if local results exist.")
    sync_group.add_argument("--no_sync", action="store_true", help="Never sync from the bucket.")
    args = parser.parse_args()

    if args.sync or (not args.no_sync and not has_local_results(args.results_dir, args.model_id, args.manifests)):
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        from open_results_pr import sync_bucket  # noqa: E402

        model_folder = args.model_id.replace("/", "-")
        if not args.sync:
            print(f"No local results for {args.model_id} in {args.results_dir}; syncing from {args.bucket}.")
        try:
            sync_bucket(args.bucket, model_folder, args.results_dir, os.environ.get("HF_TOKEN"))
        except subprocess.CalledProcessError as exc:
            print(
                f"ERROR: could not sync {args.bucket}/{model_folder} (exit {exc.returncode}). "
                f"Check the model folder exists in the bucket and that $HF_TOKEN can read it.",
                file=sys.stderr,
            )
            sys.exit(1)

    eval_utils.score_results(
        args.results_dir,
        model_id=args.model_id,
        multilingual=args.multilingual,
        csv_only=args.csv_only,
        language=args.language,
        manifests=args.manifests,
        sim_backend=args.sim_backend,
    )
