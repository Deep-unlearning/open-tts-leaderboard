#!/bin/bash
# HF Jobs Phonon evaluation: generate → transcribe → score (English, fixed voice).
# Usage: bash phonon/submit_jobs.sh [gradium/phonon-31m-en|gradium/phonon-40m-en]
# Local inference uses run_eval.py directly; TTFA uses submit_ttfa_jobs.sh.
set -euo pipefail
umask 077
RESULTS_BUCKET="${PHONON_RESULTS_BUCKET:?Set a private results bucket}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ──────────────────────────────────────────────────────────
# Dockerfile: https://huggingface.co/spaces/Steveeeeeeen/evals-phonon-040/blob/main/Dockerfile
TTS_SPACE="${PHONON_SPACE:-Steveeeeeeen/evals-phonon-040}"
RUN_EVAL_INJECT="$(inject_run_eval)"
DEFAULT_STAGES="generate transcribe"
SUPPORTS_VOICE_CLONE=false
STAGE_RETRIES=0
PHONON_THREADS="${PHONON_THREADS:-1}"
PHONON_VOICE="${PHONON_VOICE:-default}"
supports_language() { [[ "$1" == en ]]; }

# A retained local directory admits each stage once, including uncertain failures.
run_stage() {
    mkdir "${PHONON_ATTEMPT_DIR}/${1}_${MODEL_SAFE}_${DATASET_SAFE}_${DATASET}_${SPLIT}" || {
        echo "Inspect the existing private attempt before retrying." >&2; return 1;
    }
    "$1"
}

# ── Stage 1: the supplied Q8 runtime is CPU-only ────────────────────────────
generate_stage() {
    local revision="$PHONON_SEED_DATASET_REVISION" runner command
    [[ "$DATASET" != zero_shot ]] || revision="$PHONON_CV3_DATASET_REVISION"
    printf -v runner '%q ' python3 /app/phonon/run_eval.py \
        --model_id "$MODEL_ID" --checkpoint_path /checkpoint --voice "$PHONON_VOICE" \
        --dataset_path "$DATASET_PATH" --dataset_revision "$revision" \
        --dataset "$DATASET" --split "$SPLIT" --device cpu --threads "$PHONON_THREADS" \
        --warmup_steps 2 --max_eval_samples "$MAX_EVAL_SAMPLES"
    command="set -euo pipefail
        ${RUN_EVAL_INJECT} true || exit 1
        umask 077
        for existing in '${BUCKET_MANIFEST}' '${BUCKET_MANIFEST%.jsonl}.run.json' \
            '/results/${MODEL_FOLDER}/${DATASET_SAFE}_${DATASET}_${SPLIT}'; do
            test ! -e \"\$existing\" || { echo 'Existing quality artifacts require inspection.' >&2; exit 1; }
        done
        work_dir=\$(mktemp -d /tmp/phonon-quality.XXXXXXXX)
        ${runner} --output_root \"\$work_dir\"
        mkdir -p '/results/${MODEL_FOLDER}'
        source_model=\"\$work_dir/results/${MODEL_SAFE}\"
        cp -r \"\$source_model/${DATASET_SAFE}_${DATASET}_${SPLIT}\" '/results/${MODEL_FOLDER}/'
        cp \"\$source_model/$(basename "${BUCKET_MANIFEST%.jsonl}.run.json")\" '${BUCKET_MANIFEST%.jsonl}.run.json'
        cp \"\$source_model/$(basename "$BUCKET_MANIFEST")\" '${BUCKET_MANIFEST}'
    "
    local -a job=(hf jobs run --flavor "$FLAVOR" --timeout 2h --secrets HF_TOKEN)
    [[ -z "$ORG_NAME" ]] || job+=(--namespace "$ORG_NAME")
    "${job[@]}" --volume "$PHONON_CHECKPOINT_VOLUME" \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "$TTS_IMAGE" bash -c "$command"
}

# ── Datasets: "config split asr_language dataset_path" ─────────────────────
DATASET_CONFIGS=(
    "tts en en ${PHONON_SEED_DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
    "zero_shot en en ${PHONON_CV3_DATASET_PATH:-${CV3_EVAL_PATH}}"
)

# ── Model: output label only; weights come from the read-only checkpoint ────
MODEL_CONFIGS=("${PHONON_MODEL_ID:-gradium/phonon-31m-en}")
[[ $# -eq 0 ]] || MODEL_CONFIGS=("$@")

fail() { echo "Phonon: $*" >&2; exit 1; }
[[ ${#MODEL_CONFIGS[@]} -eq 1 ]] || fail "Select one checkpoint label per invocation."
for name in PHONON_CHECKPOINT_VOLUME PHONON_VOICE PHONON_SEED_DATASET_REVISION \
    PHONON_CV3_DATASET_REVISION PHONON_ATTEMPT_DIR; do
    [[ -n "${!name:-}" ]] || fail "Set ${name}."
done
[[ "$PHONON_SEED_DATASET_REVISION" =~ ^[0-9a-f]{40}$ && "$PHONON_CV3_DATASET_REVISION" =~ ^[0-9a-f]{40}$ ]] || fail "Pin both dataset revisions."
[[ "$PHONON_THREADS" =~ ^[1-8]$ && "$PHONON_VOICE" =~ ^[A-Za-z0-9_-]+$ ]] || fail "Select a fixed preset and 1–8 CPU threads."
[[ "$MAX_EVAL_SAMPLES" == -1 || "$MAX_EVAL_SAMPLES" =~ ^[1-9][0-9]*$ ]] || fail "Use a positive sample count or -1 for the full split."
[[ "$VOICE_CLONE" == false && ( -z "$ONLY_LANGS" || "$ONLY_LANGS" == en ) ]] || fail "English fixed-voice evaluation only."
case "${STAGES:-$DEFAULT_STAGES}" in
    "generate transcribe")
        [[ "${PHONON_DIAGNOSTIC_CPU:-false}" == true && "$FLAVOR" == cpu-upgrade ]] || fail "Q8 generation requires PHONON_DIAGNOSTIC_CPU=true FLAVOR=cpu-upgrade; A100/H200 are unsupported." ;;
    transcribe) ;;
    *) fail "Supported stages: 'generate transcribe' or 'transcribe'." ;;
esac

# Metadata only: require a Docker Space and keep both data buckets private.
"${PHONON_PYTHON:-python3}" - "$TTS_SPACE" "$RESULTS_BUCKET" "$PHONON_CHECKPOINT_VOLUME" \
    "$PHONON_ATTEMPT_DIR" "${MODEL_CONFIGS[0]}" "${DATASET_CONFIGS[@]}" <<'PY'
import os
from pathlib import Path
import re
import stat
import sys

try:
    from huggingface_hub import HfApi
    space, results, volume, attempts, model, *datasets = sys.argv[1:]
    repo = r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*"
    labels = (space, results, model, *(row.split()[3] for row in datasets))
    if not all(len(row.split()) == 4 for row in datasets) or not all(re.fullmatch(repo, value) and all(p not in (".", "..") for p in value.split("/")) for value in labels):
        raise ValueError("Invalid model or dataset selection")
    match = re.fullmatch(r"hf://buckets/(" + repo + r")(/[^:\r\n]*)?:/checkpoint:ro", volume)
    if not match or results.lower() == match[1].lower() or results == "hf-audio/tts_leaderboard_h200":
        raise ValueError("Separate private input and output buckets required")
    if match[2] and any(p in ("", ".", "..") for p in match[2][1:].split("/")):
        raise ValueError("Invalid checkpoint prefix")
    root = Path(attempts).expanduser().absolute()
    if root.resolve() != root or not root.is_dir() or root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError("Private attempt directory required")
    if any((p / ".git").exists() for p in (root, *root.parents)) or any(root.glob("*.json")):
        raise ValueError("Inspect earlier attempts outside Git before submitting")
    api = HfApi()
    image = api.space_info(space)
    if image.sdk != "docker" or not all(api.bucket_info(bucket).private is True for bucket in (results, match[1])):
        raise ValueError("Docker Space and private buckets required")
except Exception:
    sys.exit("Phonon private resource preflight failed.")
PY

run_pipeline
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
exit "$status"
