#!/bin/bash
# HF Jobs TTS eval — Fish Audio Drama 3 backend (generate → transcribe → SIM* → score; *voice-clone only, VOICE_CLONE=true).
# Runs locally; submits HF Jobs. Usage: HF_TOKEN=hf_... bash submit_jobs.sh ["fishaudio/drama-3 8" ...]
# Shared orchestration (stages 2/3, injection, logging, scoring) lives in scripts/tts_jobs_common.sh.
# TTFA: bash submit_ttfa_jobs.sh fishaudio-drama-3 (sources this file in library mode for the payloads below).

# In-job runner: verifies the image's pinned SGLang Omni checkout, serves the pinned model snapshot on
# port 8000, runs the client, and copies results to the bucket on exit (even after a partial failure).
drama_job_script() {
    cat <<'DRAMA_JOB_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID=fishaudio/drama-3
MODEL_REVISION=e0a9218aa48888270f06ef1ed9ea67622c0e17d2
SERVER_REVISION=12f65fa60fbb5217ad47f58b2e58b18092c818a1
JOB_FLAVOR="${JOB_FLAVOR:-h200}"
DRAMA_BUCKET_MOUNT="${DRAMA_BUCKET_MOUNT:-/results}"
[[ "$JOB_FLAVOR" == h200 || "$JOB_FLAVOR" == a100-large ]] || { echo 'Unsupported GPU job flavor.' >&2; exit 2; }
[[ -d "$DRAMA_BUCKET_MOUNT" ]] || { echo 'Results bucket mount is unavailable; refusing generation.' >&2; exit 2; }

"$PYTHON_BIN" - "$SERVER_REVISION" <<'PY'
import pathlib
import subprocess
import sys

import sglang_omni

module_root = pathlib.Path(sglang_omni.__file__).resolve().parent
source_root = pathlib.Path(subprocess.check_output(
    ["git", "-C", str(module_root), "rev-parse", "--show-toplevel"], text=True
).strip())
if not (source_root / "sglang_omni").samefile(module_root):
    raise SystemExit("SGLang Omni must be installed from its retained Git checkout.")
revision = subprocess.check_output(
    ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
).strip()
if revision != sys.argv[1]:
    raise SystemExit(f"SGLang Omni revision mismatch: expected {sys.argv[1]}, got {revision}")
modified = subprocess.check_output(
    ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"], text=True
).strip()
if modified:
    raise SystemExit("SGLang Omni checkout has modified tracked files; refusing unpinned code.")
print(f"Verified SGLang Omni revision {revision}")
PY

SNAPSHOT_PATH="$("$PYTHON_BIN" - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
)"
[[ -d "$SNAPSHOT_PATH" ]] || { echo 'Pinned model snapshot directory was not returned.' >&2; exit 1; }
printf 'Verified model snapshot: %s @ %s\n' "$MODEL_ID" "$MODEL_REVISION"

cd "$BACKEND_DIR"
mkdir -p results
SERVER_PID=""
cleanup() {
    local job_status=$?
    trap - EXIT
    # Keep a partial attempt: mark its receipt failed, then copy whatever was generated.
    if [[ -d results/fishaudio-drama-3 ]]; then
        if [[ "$job_status" -ne 0 ]]; then
            "$PYTHON_BIN" - "$job_status" <<'PY' || echo 'Could not mark incomplete receipt; preserving its original bytes for inspection.' >&2
import json
from pathlib import Path
import sys

for path in Path("results/fishaudio-drama-3").glob("MODEL_*.run.json"):
    receipt = json.loads(path.read_text())
    receipt.update(status="failed", execution_exit_status=int(sys.argv[1]))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)
PY
        fi
        if cp -R results/fishaudio-drama-3 "$DRAMA_BUCKET_MOUNT/"; then
            printf 'Preserved generation outputs in the results mount (client exit status %s).\n' "$job_status"
        else
            echo 'Failed to preserve generation outputs; manual artifact inspection is required.' >&2
            [[ "$job_status" -ne 0 ]] || job_status=1
        fi
    fi
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$job_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$PYTHON_BIN" -m sglang_omni.cli serve --model-path "$SNAPSHOT_PATH" --model-name "$MODEL_ID" \
    --host 127.0.0.1 --port 8000 --allowed-local-media-path "$BACKEND_DIR/results" &
SERVER_PID=$!

# run_eval.py's argparse rejects any other model id / revision.
"$PYTHON_BIN" run_eval.py "$@" --model_id "$MODEL_ID" \
    --model_revision "$MODEL_REVISION" --server_revision "$SERVER_REVISION" --port 8000

MODEL_OUTPUT=results/fishaudio-drama-3
[[ -d "$MODEL_OUTPUT" ]] || { echo 'Client produced no model results directory.' >&2; exit 1; }
for sidecar in "$MODEL_OUTPUT"/TTFA_MODEL_*.json; do
    [[ -e "$sidecar" ]] || continue
    case "$sidecar" in
        *__sglang__"$JOB_FLAVOR".json) ;;
        *) mv "$sidecar" "${sidecar%.json}__sglang__${JOB_FLAVOR}.json" ;;
    esac
done
DRAMA_JOB_SCRIPT
}

inject_script() {
    local target="$1" payload
    payload="$(gzip -9c | base64 -w0)"
    printf "mkdir -p /app/fishaudio-drama-3 && echo '%s' | base64 -d | gzip -dc > %s &&" "$payload" "$target"
}

# The shared TTFA launcher reuses these payloads without running this launcher or
# changing its caller's shell options. Runtime scripts retain their own exit traps.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/tts_jobs_common.sh"

# ── Backend config ───────────────────────────────────────────────────────────
# Stage-1 image: pinned SGLang base + SGLang Omni checkout (Dockerfile in the Space: https://huggingface.co/spaces/bezzam/evals-drama3/blob/main/Dockerfile)
TTS_SPACE="${TTS_SPACE:-bezzam/evals-drama3}"
# gen FLAVOR (h200) is set centrally in tts_jobs_common.sh for RTFx comparability — don't override.
WARMUP_STEPS="${WARMUP_STEPS:-2}"   # untimed warm-up batches
SUPPORTS_VOICE_CLONE=true           # VOICE_CLONE=true clones each sample's reference speaker + scores SIM

# run_eval.py is large: gzip it, since a job body is one `bash -c` argument (128 KiB cap).
DRAMA_INJECT="$(inject_cmd_gz "${BACKEND_DIR}/run_eval.py" /app/fishaudio-drama-3/run_eval.py) "
DRAMA_INJECT+="$(inject_cmd_gz "${REPO_ROOT}/scripts/run_eval_utils.py" /app/scripts/run_eval_utils.py) "
DRAMA_INJECT+="$(drama_job_script | inject_script /app/fishaudio-drama-3/run_job.sh) "

# ── Stage 1: launch the pinned SGLang Omni server, then generate via the HTTP client. ──
# MODEL_CFG = "model_id concurrency". The server batches concurrent requests (up to 16 by default);
# 8 matches the upstream cookbook's non-streaming H200 benchmark and stays clear of vocoder OOMs.
generate_stage() {
    local _ BATCH_SIZE client_args
    read -r _ BATCH_SIZE <<< "${MODEL_CFG}"
    printf -v client_args '%q ' --dataset_path "${DATASET_PATH}" --dataset "${DATASET}" --split "${SPLIT}" \
        --max_eval_samples "${MAX_EVAL_SAMPLES}" --batch_size "${BATCH_SIZE:-8}" \
        --warmup_steps "${WARMUP_STEPS}" ${VOICE_CLONE_FLAG}
    hf jobs run \
        --flavor "${FLAVOR}" --timeout 8h --secrets HF_TOKEN ${NAMESPACE_ARG} \
        --env "JOB_FLAVOR=${FLAVOR}" \
        --volume "hf://buckets/${RESULTS_BUCKET}:/results" "${TTS_IMAGE}" \
        bash -c "
            set -e
            ${DRAMA_INJECT}
            ${SCRIPTS_PYTHONPATH}
            bash /app/fishaudio-drama-3/run_job.sh ${client_args}
        "
}

# ── Datasets: "config split asr_language [dataset_path]" (comment/uncomment to pick) ────────
DATASET_CONFIGS=(
    "tts en en"
    "tts zh zh"
    "zero_shot en en ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot zh zh ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot fr fr ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ja ja ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ko ko ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot de de ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot es es ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot it it ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
    "zero_shot ru ru ${CV3_EVAL_PATH:-hf-audio/cv3_eval}"
)

# ── Models: "model_id concurrency" (CLI args override; only fishaudio/drama-3 is supported) ──
MODEL_CONFIGS=(
    "fishaudio/drama-3 8"
)
[[ $# -gt 0 ]] && MODEL_CONFIGS=("$@")

run_pipeline
