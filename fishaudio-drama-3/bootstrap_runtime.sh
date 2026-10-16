#!/usr/bin/env bash
# Installs the pinned runtime inside an ephemeral job; never builds or publishes an image.
set -euo pipefail

EXPECTED_IMAGE=lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75
SERVER_REVISION=12f65fa60fbb5217ad47f58b2e58b18092c818a1
[[ "${DRAMA_RUNTIME_IMAGE:-}" == "$EXPECTED_IMAGE" ]] || { echo 'Runtime bootstrap requires the approved immutable SGLang base.' >&2; exit 2; }
[[ "${SGLANG_OMNI_AUTO_CLONE:-}" == 0 ]] || { echo 'Moving-main initialization must be disabled before bootstrap.' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-/opt/sglang/bin/python3}"
BOOTSTRAP_ROOT="${DRAMA_BOOTSTRAP_ROOT:-/opt/fish-drama-3-runtime}"
OMNI_ROOT="$BOOTSTRAP_ROOT/sglang-omni"
DRAMA_BUCKET_MOUNT="${DRAMA_BUCKET_MOUNT:-/results}"
[[ -x "$PYTHON_BIN" && -d "$DRAMA_BUCKET_MOUNT" ]] || { echo 'Expected Python interpreter or results mount is unavailable.' >&2; exit 2; }
BOOTSTRAP_PHASE=base-check
BOOTSTRAP_RECEIPT="$DRAMA_BUCKET_MOUNT/fishaudio-drama-3/runtime_bootstrap/$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
write_receipt() {
    "$PYTHON_BIN" - "$BOOTSTRAP_RECEIPT" "$1" "$BOOTSTRAP_PHASE" "$EXPECTED_IMAGE" "$OMNI_ROOT" "$SERVER_REVISION" "${2:-}" <<'PY' || true
import datetime
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

destination, exit_code, phase, image, source_root, server_revision, status = sys.argv[1:]
packages = {}
for name in ("torch", "torchvision", "torchaudio", "torchcodec", "sglang", "flashinfer-python",
             "flash-attn-4", "transformers", "descript-audiotools", "descript-audio-codec", "protobuf", "datasets"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
try:
    installed_source = subprocess.check_output(
        ["git", "-C", source_root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    installed_source = None
try:
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                            check=False, capture_output=True, text=True, timeout=60)
    # Public image packages may include direct URLs; strip URL credentials and query strings.
    dependency_freeze = re.sub(r"(https?://)[^/\s@]+@", r"\1<redacted>@", freeze.stdout)
    dependency_freeze = re.sub(r"(https?://[^\s?#]+)[?#][^\s]*", r"\1", dependency_freeze)
    freeze_status = freeze.returncode
except (OSError, subprocess.TimeoutExpired):
    dependency_freeze, freeze_status = "", None
def tool_version(command):
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()[:2000] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None
payload = {
    "status": status or ("complete" if exit_code == "0" else "failed"), "exit_code": int(exit_code),
    "phase": phase, "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "image": image, "python": sys.version.split()[0], "source_root": source_root,
    "expected_server_revision": server_revision, "installed_server_revision": installed_source,
    "transport_scope": "single node, all Drama 3 stages on GPU 0, CUDA IPC / SHM",
    "ucx_build_performed": False, "ucx_version": tool_version(["ucx_info", "-v"]),
    "sox_version": tool_version(["sox", "--version"]), "packages": packages,
    "dependency_freeze": dependency_freeze.splitlines(), "freeze_exit_code": freeze_status,
    "note": "Single-node bootstrap only; does not establish cross-node RDMA/UCX compatibility or certify synthesis.",
}
target = Path(destination)
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(target)
print(f"Runtime bootstrap receipt saved ({payload['status']}, phase={phase}).")
PY
}
finish_bootstrap() {
    bootstrap_exit=$?
    trap - EXIT
    write_receipt "$bootstrap_exit"
    exit "$bootstrap_exit"
}
trap finish_bootstrap EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
write_receipt 0 running

"$PYTHON_BIN" - <<'PY'
import os
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("Bootstrap requires Python 3.12 from the approved base.")
if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("Bootstrap requires the approved Linux amd64 image.")
if os.environ.get("CUDA_VERSION") != "13.0.3":
    raise SystemExit("Bootstrap requires the approved CUDA 13.0.3 base.")
import importlib.metadata
import torch

for name, expected in (("torch", "2.13.0"), ("sglang", "0.5.21"), ("flashinfer-python", "0.6.18")):
    installed = importlib.metadata.version(name).partition("+")[0]
    if installed != expected:
        raise SystemExit(f"Base dependency mismatch: {name} expected {expected}, got {installed}")
if torch.version.cuda is None or not torch.version.cuda.startswith("13."):
    raise SystemExit("The installed Torch build must target CUDA 13.")
if not torch.cuda.is_available():
    raise SystemExit("The approved GPU runtime is unavailable.")
if torch.cuda.device_count() != 1:
    raise SystemExit("This bootstrap supports one GPU with all model stages on GPU 0.")
print("Verified Python 3.12, CUDA 13.0.3, Torch 2.13.0, SGLang 0.5.21 and FlashInfer 0.6.18.")
PY

for bootstrap_tool in git uv ffmpeg; do
    command -v "$bootstrap_tool" >/dev/null || { printf 'Missing bootstrap tool: %s\n' "$bootstrap_tool" >&2; exit 1; }
done
[[ ! -e "$BOOTSTRAP_ROOT" ]] || { echo 'Bootstrap source directory already exists; refusing to replace it.' >&2; exit 1; }
mkdir -p "$BOOTSTRAP_ROOT"

# The pinned upstream Dockerfile adds the sox executable for audio dependencies.
BOOTSTRAP_PHASE=audio-system-dependencies
write_receipt 0 running
if ! command -v sox >/dev/null; then
    apt-get update
    apt-get install -y --no-install-recommends sox
    rm -rf /var/lib/apt/lists/*
fi

BOOTSTRAP_PHASE=omni-install
write_receipt 0 running
git init -q "$OMNI_ROOT"
git -C "$OMNI_ROOT" fetch -q --depth 1 https://github.com/sgl-project/sglang-omni.git "$SERVER_REVISION"
git -C "$OMNI_ROOT" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$OMNI_ROOT" rev-parse HEAD)" == "$SERVER_REVISION" ]] || { echo 'Omni source revision mismatch.' >&2; exit 1; }
"$PYTHON_BIN" - "$OMNI_ROOT/pyproject.toml" "$BOOTSTRAP_ROOT/omni-overrides.txt" <<'PY'
from pathlib import Path
import sys
import tomllib

with open(sys.argv[1], "rb") as source:
    requirements = tomllib.load(source)["tool"]["uv"]["override-dependencies"]
Path(sys.argv[2]).write_text("\n".join(requirements) + "\n", encoding="utf-8")
PY
(
    cd "$OMNI_ROOT"
    uv pip install --python "$PYTHON_BIN" --prerelease=allow --no-build-isolation \
        --overrides "$BOOTSTRAP_ROOT/omni-overrides.txt" -e '.[fishaudio]' datasets==3.6.0
    "$PYTHON_BIN" -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache
)

BOOTSTRAP_PHASE=verify-installed
write_receipt 0 running
"$PYTHON_BIN" "$OMNI_ROOT/.github/scripts/verify_omni_installed_pins.py" "$PYTHON_BIN" "$OMNI_ROOT"
"$PYTHON_BIN" - "$OMNI_ROOT" <<'PY'
import importlib.metadata
from pathlib import Path
import subprocess
import sys

import audiotools
import dac
import sglang_omni
import torch
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.router import CommRouter
from sglang_omni.models.fishaudio_drama3.config import Drama3PipelineConfig
from sglang_omni.platforms import current_platform
import sglang_omni.cli
import sglang_omni.relay

root = Path(sys.argv[1]).resolve()
if not (root / "sglang_omni").samefile(Path(sglang_omni.__file__).resolve().parent):
    raise SystemExit("Omni import does not resolve to the retained pinned source checkout.")
if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
    raise SystemExit("Installed Omni checkout has modified tracked files.")
for name, expected in (("descript-audiotools", "0.7.2"), ("descript-audio-codec", "1.0.0")):
    if importlib.metadata.version(name) != expected:
        raise SystemExit(f"Fish codec dependency mismatch: {name}")
if importlib.metadata.version("datasets") != "3.6.0":
    raise SystemExit("The evaluation runner requires datasets==3.6.0 for decoded prompt audio.")
if Drama3PipelineConfig.architecture != "FishOmniModel":
    raise SystemExit("Drama 3 pipeline registration mismatch.")
if torch.version.cuda is None or not torch.version.cuda.startswith("13.") or not torch.cuda.is_available():
    raise SystemExit("Installed runtime no longer has the required CUDA 13 GPU backend.")
if current_platform.get_intra_node_transport() != TransportKind.CUDA_IPC:
    raise SystemExit("This single-node bootstrap requires the pinned CUDA IPC transport.")
if any(stage.gpu != 0 for stage in Drama3PipelineConfig.model_fields["stages"].default):
    raise SystemExit("The pinned Drama 3 placement no longer uses only GPU 0.")
print("Verified pinned clean Omni source and Drama 3 codec imports.")
PY
BOOTSTRAP_PHASE=complete
