#!/usr/bin/env bash
# Runs inside the selected HF Job image after the local launcher injects the backend.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID=fishaudio/drama-3
MODEL_REVISION=e0a9218aa48888270f06ef1ed9ea67622c0e17d2
SERVER_REVISION=12f65fa60fbb5217ad47f58b2e58b18092c818a1
JOB_FLAVOR="${JOB_FLAVOR:-h200}"
DRAMA_BUCKET_MOUNT="${DRAMA_BUCKET_MOUNT:-/results}"
[[ "$JOB_FLAVOR" == h200 || "$JOB_FLAVOR" == a100-large ]] || { echo 'Unsupported GPU job flavor.' >&2; exit 2; }

# Reject mismatched identity before imports, downloads, or server startup.
CLIENT_ARGS=("$@")
for (( argument_index=0; argument_index < ${#CLIENT_ARGS[@]}; argument_index++ )); do
    argument="${CLIENT_ARGS[$argument_index]}"
    case "$argument" in
        --model_id|--model_revision|--server_revision)
            (( argument_index + 1 < ${#CLIENT_ARGS[@]} )) || { echo 'Missing identity argument value.' >&2; exit 2; }
            identity_value="${CLIENT_ARGS[$((argument_index + 1))]}"
            case "$argument" in
                --model_id) expected_value="$MODEL_ID" ;;
                --model_revision) expected_value="$MODEL_REVISION" ;;
                --server_revision) expected_value="$SERVER_REVISION" ;;
            esac
            [[ "$identity_value" == "$expected_value" ]] || { printf 'Rejected mismatched %s.\n' "$argument" >&2; exit 2; }
            ;;
        --model_id=*|--model_revision=*|--server_revision=*)
            echo 'Use separate identity arguments and values.' >&2; exit 2 ;;
        --host*|--port*|--base_url*)
            echo 'The job uses its verified local server on port 8000.' >&2; exit 2 ;;
    esac
done

PYTHONPATH="$BACKEND_DIR${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" - "$DRAMA_BUCKET_MOUNT" "$JOB_FLAVOR" "$@" <<'PY'
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument("--dataset_path", default="bezzam/seed_tts_eval")
parser.add_argument("--dataset", default="tts")
parser.add_argument("--dataset_local_root")
parser.add_argument("--split", default="en")
parser.add_argument("--voice_clone", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--ttfa_probe", type=int, default=0)
parser.add_argument("--resume", action="store_true")
args, _ = parser.parse_known_args(sys.argv[3:])
if args.resume:
    raise SystemExit("Bucket resume is not supported; review and resume partial results separately.")
if any("/" in value or "\\" in value or value in ("", ".", "..") for value in (args.split,)):
    raise SystemExit("Split must be a filename component.")
bucket = Path(sys.argv[1])
if not bucket.is_dir():
    raise SystemExit("Results bucket mount is unavailable; refusing generation.")
suffix = "_voice_clone" if args.voice_clone else ""
dataset = args.dataset_path.replace("/", "-")
config = args.dataset.replace("/", "-")
stem = f"MODEL_fishaudio-drama-3_DATASET_{dataset}_{config}_{args.split}{suffix}"
if args.ttfa_probe:
    stem = f"TTFA_{stem}"
    candidates = [f"{stem}.json", f"{stem}.run.json",
                  f"{stem}__sglang__{sys.argv[2]}.json", f"{stem}__sglang__{sys.argv[2]}.run.json"]
else:
    candidates = [f"{stem}.jsonl", f"{stem}.run.json"]
for name in candidates:
    if (bucket / "fishaudio-drama-3" / name).exists():
        raise SystemExit("Targeted results already exist in the bucket; refusing a duplicate run or overwrite.")
if args.dataset_local_root:
    from input_dataset import resolve_dataset_source
    resolve_dataset_source(args)
    print("Verified selected private input Parquet checksum and canonical split provenance.")
print("Verified that this dataset, clone mode, and probe target have no existing bucket results.")
PY

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
    # Retain the single attempt even if inference fails partway through a split.
    # The existing-target guard still prevents an automatic restart/overwrite.
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
            printf 'Preserved generation outputs in the private results mount (client exit status %s).\n' "$job_status"
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
echo 'ASR, SIM, and scoring are separate steps; the exit handler preserves this attempt before shutdown.'
