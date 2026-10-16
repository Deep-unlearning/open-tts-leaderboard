#!/usr/bin/env bash
# Preview bounded generation jobs; --execute submits them serially without scoring.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash fishaudio-drama-3/submit_jobs.sh [--execute]

Default: preview one Seed-TTS en generation job, 8 rows, H200, 30-minute timeout.
Execution requires TTS_SPACE=owner/public-space or an immutable TTS_IMAGE, plus RESULTS_BUCKET.
The image must retain the pinned server checkout, or use the explicit runtime bootstrap.

Environment:
  DATASET_SCOPE=seed_en|cv3_en|full    default seed_en; full means 11 serial jobs
  MAX_EVAL_SAMPLES=8                 -1 explicitly opts into full splits
  JOB_TIMEOUT=30m                   HF Job wall-clock limit per job
  FIRST_JOB_TIMEOUT=30m             optional tighter limit for the first selected job
  VOICE_CLONE=false                  true saves references for later SIM scoring
  TTFA_PROBE=0                      positive N writes only a TTFA sidecar
  BATCH_SIZE=1                      concurrent requests during offline generation
  WARMUP_STEPS=2                    untimed warm-up batches
  RESULTS_PREFIX=fish-drama-3/full   optional subfolder within RESULTS_BUCKET
  INPUTS_BUCKET=owner/private-bucket optional local dataset input mount, read-only
  INPUTS_PREFIX=fish-drama-3/inputs  optional subfolder within INPUTS_BUCKET
  TTS_IMAGE=repository@sha256:...    direct immutable Docker image; excludes TTS_SPACE
  BOOTSTRAP_RUNTIME=false           true installs the pinned runtime in the approved base
  DATASET_NAMESPACE=bezzam           default private eval-dataset namespace
  DATASET_PATH=owner/seed_tts_eval    override Seed-TTS repository
  CV3_EVAL_PATH=owner/cv3_eval        override CV3 repository
  PYTHON_BIN=python                 Python in the image's SGLang environment
  ORG_NAME=owner                    optional HF Jobs billing namespace

This launcher does not start ASR, SIM, scoring, result publication, or automatic retries.
EOF
}

EXECUTE=false
for option in "$@"; do
    case "$option" in
        --execute) EXECUTE=true ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$option" >&2; usage >&2; exit 2 ;;
    esac
done

MODEL_ID="${MODEL_ID:-fishaudio/drama-3}"
MODEL_REVISION="${MODEL_REVISION:-e0a9218aa48888270f06ef1ed9ea67622c0e17d2}"
SERVER_REVISION="${SERVER_REVISION:-12f65fa60fbb5217ad47f58b2e58b18092c818a1}"
[[ "$MODEL_ID" == fishaudio/drama-3 ]] || { echo 'Only fishaudio/drama-3 is supported.' >&2; exit 2; }
[[ "$MODEL_REVISION" == e0a9218aa48888270f06ef1ed9ea67622c0e17d2 ]] || { echo 'Unexpected Drama 3 model revision.' >&2; exit 2; }
[[ "$SERVER_REVISION" == 12f65fa60fbb5217ad47f58b2e58b18092c818a1 ]] || { echo 'Unexpected SGLang Omni revision.' >&2; exit 2; }

DATASET_SCOPE="${DATASET_SCOPE:-seed_en}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-8}"
JOB_TIMEOUT="${JOB_TIMEOUT:-30m}"
FIRST_JOB_TIMEOUT="${FIRST_JOB_TIMEOUT:-$JOB_TIMEOUT}"
VOICE_CLONE="${VOICE_CLONE:-false}"
TTFA_PROBE="${TTFA_PROBE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
FLAVOR="${FLAVOR:-h200}"
DATASET_NAMESPACE="${DATASET_NAMESPACE:-bezzam}"
DATASET_PATH="${DATASET_PATH:-${DATASET_NAMESPACE}/seed_tts_eval}"
CV3_EVAL_PATH="${CV3_EVAL_PATH:-${DATASET_NAMESPACE}/cv3_eval}"
ORG_NAME="${ORG_NAME:-}"
TTS_IMAGE="${TTS_IMAGE:-}"
BOOTSTRAP_RUNTIME="${BOOTSTRAP_RUNTIME:-false}"
INPUTS_BUCKET="${INPUTS_BUCKET:-}"
INPUTS_PREFIX="${INPUTS_PREFIX:-}"
[[ "$BOOTSTRAP_RUNTIME" == false || "$BOOTSTRAP_RUNTIME" == true ]] || { echo 'BOOTSTRAP_RUNTIME must be true or false.' >&2; exit 2; }
[[ -z "$TTS_IMAGE" || -z "${TTS_SPACE:-}" ]] || { echo 'TTS_IMAGE and TTS_SPACE are mutually exclusive.' >&2; exit 2; }
if [[ -n "$TTS_IMAGE" && ! "$TTS_IMAGE" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)+@sha256:[0-9a-f]{64}$ ]]; then
    echo 'TTS_IMAGE must be a repository pinned by a lowercase sha256 digest; moving tags are rejected.' >&2
    exit 2
fi
if [[ "$BOOTSTRAP_RUNTIME" == true ]]; then
    [[ "$TTS_IMAGE" == lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75 ]] || { echo 'Runtime bootstrap requires the approved immutable SGLang linux/amd64 base.' >&2; exit 2; }
    PYTHON_BIN="${PYTHON_BIN:-/opt/sglang/bin/python3}"
    [[ "$PYTHON_BIN" == /opt/sglang/bin/python3 ]] || { echo 'Runtime bootstrap requires PYTHON_BIN=/opt/sglang/bin/python3.' >&2; exit 2; }
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
RESULTS_PREFIX="${RESULTS_PREFIX:-}"
if [[ -n "$RESULTS_PREFIX" && ! "$RESULTS_PREFIX" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$ ]]; then
    echo 'RESULTS_PREFIX must contain safe relative path components; no traversal, whitespace, or shell metacharacters.' >&2
    exit 2
fi
RESULTS_SUBPATH="${RESULTS_PREFIX:+/${RESULTS_PREFIX}}"
if [[ -n "$INPUTS_PREFIX" && ! "$INPUTS_PREFIX" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$ ]]; then
    echo 'INPUTS_PREFIX must contain safe relative path components.' >&2
    exit 2
fi
[[ -z "$INPUTS_PREFIX" || -n "$INPUTS_BUCKET" ]] || { echo 'INPUTS_PREFIX requires INPUTS_BUCKET.' >&2; exit 2; }
[[ -z "$INPUTS_BUCKET" || "$INPUTS_BUCKET" =~ ^[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*$ ]] || { echo 'INPUTS_BUCKET must be owner/name.' >&2; exit 2; }
INPUTS_SUBPATH="${INPUTS_PREFIX:+/${INPUTS_PREFIX}}"
[[ "$MAX_EVAL_SAMPLES" =~ ^(-1|[1-9][0-9]*)$ ]] || { echo 'MAX_EVAL_SAMPLES must be -1 or positive.' >&2; exit 2; }
[[ "$TTFA_PROBE" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'TTFA_PROBE must be zero or positive.' >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'BATCH_SIZE must be positive.' >&2; exit 2; }
[[ "$WARMUP_STEPS" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'WARMUP_STEPS must be nonnegative.' >&2; exit 2; }
[[ "$JOB_TIMEOUT" =~ ^[1-9][0-9]*[smhd]$ ]] || { echo 'JOB_TIMEOUT must be a positive duration, e.g. 30m or 8h.' >&2; exit 2; }
[[ "$FIRST_JOB_TIMEOUT" =~ ^[1-9][0-9]*[smhd]$ ]] || { echo 'FIRST_JOB_TIMEOUT must be a positive duration.' >&2; exit 2; }
[[ "$VOICE_CLONE" == false || "$VOICE_CLONE" == true ]] || { echo 'VOICE_CLONE must be true or false.' >&2; exit 2; }
if [[ "$TTFA_PROBE" == 0 ]]; then
    [[ "$FLAVOR" == h200 ]] || { echo 'Offline RTFx generation requires FLAVOR=h200.' >&2; exit 2; }
else
    [[ "$FLAVOR" == h200 || "$FLAVOR" == a100-large ]] || { echo 'TTFA supports h200 or a100-large GPU jobs.' >&2; exit 2; }
    [[ "$BATCH_SIZE" == 1 ]] || { echo 'TTFA requires BATCH_SIZE=1.' >&2; exit 2; }
fi

JOB_CONFIGS=()
case "$DATASET_SCOPE" in
    seed_en) JOB_CONFIGS=("$DATASET_PATH tts en") ;;
    cv3_en) JOB_CONFIGS=("$CV3_EVAL_PATH zero_shot en") ;;
    full)
        JOB_CONFIGS=("$DATASET_PATH tts en" "$DATASET_PATH tts zh")
        for language in en zh fr ja ko de es it ru; do
            JOB_CONFIGS+=("$CV3_EVAL_PATH zero_shot $language")
        done
        ;;
    *) echo 'DATASET_SCOPE must be seed_en, cv3_en, or full.' >&2; exit 2 ;;
esac
for dataset_repo in "$DATASET_PATH" "$CV3_EVAL_PATH"; do
    [[ "$dataset_repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'Dataset IDs must be owner/repository.' >&2; exit 2; }
done
[[ -z "$ORG_NAME" || "$ORG_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid ORG_NAME.' >&2; exit 2; }

printf 'Model: %s @ %s\nSGLang Omni: %s\n' "$MODEL_ID" "$MODEL_REVISION" "$SERVER_REVISION"
printf 'Plan: %s serial job(s), scope=%s, rows per split=%s, clone=%s, TTFA=%s\n' "${#JOB_CONFIGS[@]}" "$DATASET_SCOPE" "$MAX_EVAL_SAMPLES" "$VOICE_CLONE" "$TTFA_PROBE"
printf 'Per job: flavor=%s timeout=%s concurrency=%s warm-up=%s\n' "$FLAVOR" "$JOB_TIMEOUT" "$BATCH_SIZE" "$WARMUP_STEPS"
[[ "$FIRST_JOB_TIMEOUT" == "$JOB_TIMEOUT" ]] || printf 'First job timeout: %s\n' "$FIRST_JOB_TIMEOUT"
SELECTED_IMAGE="${TTS_IMAGE:-${TTS_SPACE:+hf.co/spaces/${TTS_SPACE}}}"
printf 'Image: %s\nBucket: %s\n' "${SELECTED_IMAGE:-<set TTS_SPACE or TTS_IMAGE>}" "${RESULTS_BUCKET:-<set RESULTS_BUCKET>}"
printf 'Prefix: %s\n' "${RESULTS_PREFIX:-<bucket root>}"
printf 'Runtime bootstrap: %s\nInputs: %s\n' "$BOOTSTRAP_RUNTIME" "${INPUTS_BUCKET:-<dataset Hub access>}"
[[ -z "$INPUTS_BUCKET" ]] || printf 'Inputs prefix: %s (read-only at /inputs)\n' "${INPUTS_PREFIX:-<bucket root>}"
printf '  %s\n' "${JOB_CONFIGS[@]}"
if [[ "$EXECUTE" == false ]]; then
    echo 'Preview only. Pass --execute after approval of this bounded run.'
    exit 0
fi

[[ -n "$SELECTED_IMAGE" && -n "${RESULTS_BUCKET:-}" ]] || { echo '--execute requires explicit TTS_SPACE or TTS_IMAGE, and RESULTS_BUCKET.' >&2; exit 2; }
[[ -z "${TTS_SPACE:-}" || "$TTS_SPACE" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'TTS_SPACE must be owner/name.' >&2; exit 2; }
[[ "$RESULTS_BUCKET" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'RESULTS_BUCKET must be owner/name.' >&2; exit 2; }
command -v hf >/dev/null || { echo 'The hf CLI is required for --execute.' >&2; exit 2; }
command -v python3 >/dev/null || { echo 'Local python3 is required to verify remote job completion.' >&2; exit 2; }

# Shared injectors use GNU base64 -w0; normalize that call on macOS too.
base64() {
    if [[ "${1:-}" == -w0 ]]; then
        shift
        if [[ $# -gt 0 ]]; then command base64 < "$1"; else command base64; fi | tr -d '\r\n'
    else
        command base64 "$@"
    fi
}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

INJECTION=""
for backend_file in run_eval.py adapter.py input_dataset.py run_job.sh; do
    [[ -s "${BACKEND_DIR}/${backend_file}" ]] || { printf 'Missing backend file: %s\n' "$backend_file" >&2; exit 2; }
    INJECTION+="$(inject_cmd_gz "${BACKEND_DIR}/${backend_file}" "/app/fishaudio-drama-3/${backend_file}") "
done
BOOTSTRAP_COMMAND=""
if [[ "$BOOTSTRAP_RUNTIME" == true ]]; then
    INJECTION+="$(inject_cmd_gz "${BACKEND_DIR}/bootstrap_runtime.sh" /app/fishaudio-drama-3/bootstrap_runtime.sh) "
    BOOTSTRAP_COMMAND='bash /app/fishaudio-drama-3/bootstrap_runtime.sh &&'
fi
for shared_file in run_eval_utils.py ttfa_probe.py; do
    INJECTION+="$(inject_cmd_gz "${REPO_ROOT}/scripts/${shared_file}" "/app/scripts/${shared_file}") "
done
SOURCE_IDENTITY="$(python3 - "$REPO_ROOT" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted((root / "normalizer").glob("*.py")):
    digest.update(path.name.encode() + b"\0" + path.read_bytes())
def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
print(json.dumps({
    "checkout_head": git("rev-parse", "HEAD"),
    "committed_tree": git("rev-parse", "HEAD:normalizer"),
    "normalizer_file_sha256": hashlib.sha256((root / "normalizer/normalizer.py").read_bytes()).hexdigest(),
    "python_files_sha256": digest.hexdigest(),
}))
PY
)"

NEXT_JOB_TIMEOUT="$FIRST_JOB_TIMEOUT"
for job_config in "${JOB_CONFIGS[@]}"; do
    ACTIVE_JOB_TIMEOUT="$NEXT_JOB_TIMEOUT"
    NEXT_JOB_TIMEOUT="$JOB_TIMEOUT"
    read -r job_dataset_path job_dataset job_split <<< "$job_config"
    CLIENT_ARGS=(--model_id "$MODEL_ID" --model_revision "$MODEL_REVISION" --server_revision "$SERVER_REVISION"
        --dataset_path "$job_dataset_path" --dataset "$job_dataset" --split "$job_split"
        --max_eval_samples "$MAX_EVAL_SAMPLES" --batch_size "$BATCH_SIZE" --warmup_steps "$WARMUP_STEPS"
        --ttfa_probe "$TTFA_PROBE")
    if [[ "$VOICE_CLONE" == true ]]; then CLIENT_ARGS+=(--voice_clone); else CLIENT_ARGS+=(--no-voice_clone); fi
    [[ -z "$INPUTS_BUCKET" ]] || CLIENT_ARGS+=(--dataset_local_root /inputs)
    printf -v QUOTED_CLIENT_ARGS '%q ' "${CLIENT_ARGS[@]}"
    JOB_ARGS=(jobs run --flavor "$FLAVOR" --timeout "$ACTIVE_JOB_TIMEOUT" --format agent --secrets HF_TOKEN
        --env "JOB_FLAVOR=$FLAVOR" --env "PYTHON_BIN=$PYTHON_BIN"
        --env "OPEN_TTS_SOURCE_IDENTITY=$SOURCE_IDENTITY"
        --volume "hf://buckets/${RESULTS_BUCKET}${RESULTS_SUBPATH}:/results")
    [[ -z "$INPUTS_BUCKET" ]] || JOB_ARGS+=(--volume "hf://buckets/${INPUTS_BUCKET}${INPUTS_SUBPATH}:/inputs:ro")
    if [[ "$BOOTSTRAP_RUNTIME" == true ]]; then
        JOB_ARGS+=(--env SGLANG_OMNI_AUTO_CLONE=0 --env "DRAMA_RUNTIME_IMAGE=$TTS_IMAGE"
            --env LD_LIBRARY_PATH=/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64)
    fi
    [[ -z "$ORG_NAME" ]] || JOB_ARGS+=(--namespace "$ORG_NAME")
    printf 'Submitting generation job: %s / %s / %s\n' "$job_dataset_path" "$job_dataset" "$job_split"
    JOB_LOG="$(mktemp "${TMPDIR:-/tmp}/drama_job_XXXXXX")"
    trap 'rm -f "$JOB_LOG"' EXIT
    hf "${JOB_ARGS[@]}" "$SELECTED_IMAGE" bash -c \
        "set -e; ${INJECTION} ${SCRIPTS_PYTHONPATH} ${BOOTSTRAP_COMMAND} bash /app/fishaudio-drama-3/run_job.sh ${QUOTED_CLIENT_ARGS}" | tee "$JOB_LOG"
    IFS= read -r JOB_START_LINE < "$JOB_LOG"
    START_PATTERN='^id=([A-Za-z0-9_-]+)( name=[^[:space:]]+)? url=https://huggingface.co/jobs/([A-Za-z0-9_.-]+)/([A-Za-z0-9_-]+)$'
    [[ "$JOB_START_LINE" =~ $START_PATTERN ]] || { echo 'Cannot verify the submitted job identity; stopping without retry.' >&2; exit 1; }
    [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[4]}" ]] || { echo 'Job ID and URL disagree; stopping.' >&2; exit 1; }
    JOB_REFERENCE="${BASH_REMATCH[3]}/${BASH_REMATCH[1]}"
    # Older hf versions stream logs but return success even when the remote job failed.
    hf jobs inspect "$JOB_REFERENCE" --format json | python3 -c '
import json
import sys

expected_owner, expected_id = sys.argv[1].split("/", 1)
jobs = json.load(sys.stdin)
if not isinstance(jobs, list) or len(jobs) != 1:
    raise SystemExit("Expected exactly one remote job status; stopping without retry.")
job = jobs[0]
if job.get("id") != expected_id or job.get("owner", {}).get("name") != expected_owner:
    raise SystemExit("Inspected job identity does not match the submitted job.")
stage = job.get("status", {}).get("stage")
if stage != "COMPLETED":
    raise SystemExit(f"Remote job is {stage!r}, not COMPLETED; stopping without retry.")
print(f"Verified remote job {expected_owner}/{expected_id} completed.")
' "$JOB_REFERENCE"
    rm -f "$JOB_LOG"
    trap - EXIT
done
