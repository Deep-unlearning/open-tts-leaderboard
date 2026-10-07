"""Mirror the seed-tts-eval SIM checkpoint (`wavlm_large_finetune.pth`) into an HF repo we own.

`transformers/score_similarity.py --sim_backend wavlm_seed_tts` needs WavLM-Large + ECAPA-TDNN
weights that upstream (microsoft/UniSpeech) only ships via Google Drive, which a job container
cannot reliably reach; this script re-hosts the same bytes on the Hub.

The file is sha256-verified against the digest pinned in score_similarity.py BEFORE upload, and the
remote LFS digest is re-checked after.

  python scripts/upload_sim_checkpoint.py                      # resolve source, verify, upload
  python scripts/upload_sim_checkpoint.py --src /path/to.pth   # use a local copy
  python scripts/upload_sim_checkpoint.py --dry_run            # verify + render the card, no upload

Needs `hf auth login` (or HF_TOKEN) with write scope. Runs on the host — no torch required.
"""

import argparse
import ast
import hashlib
import os
import sys

DEFAULT_REPO_ID = "bezzam/wavlm_large_finetune_seed_tts_eval"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_SIMILARITY = os.path.join(REPO_ROOT, "transformers", "score_similarity.py")

# Upstream source of truth: microsoft/UniSpeech, downstreams/speaker_verification, the
# "WavLM large / Fix pre-train: No" row (Vox1-O EER 0.431 — their best SV model).
UNISPEECH_README = "https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification"
GDRIVE_FILE_ID = "1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
GDRIVE_URL = f"https://drive.usercontent.google.com/download?id={GDRIVE_FILE_ID}&export=download&confirm=t"
EXPECTED_SIZE = 1301926579


def read_pinned_constants():
    """Pull the pinned checkpoint constants out of score_similarity.py without importing it.

    score_similarity.py imports torch at module scope and this script runs on the host, so it is
    parsed rather than executed; the digest here can never drift from the one the eval enforces.
    """
    tree = ast.parse(open(SCORE_SIMILARITY, encoding="utf-8").read())
    wanted = {"WAVLM_SEED_TTS_REPO_ID", "WAVLM_SEED_TTS_FILENAME", "WAVLM_SEED_TTS_SHA256"}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted and isinstance(node.value, ast.Constant):
                found[target.id] = node.value.value
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"Could not find {sorted(missing)} in {SCORE_SIMILARITY}")
    return found


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_from_gdrive(dest):
    """Stream the official Drive copy. `confirm=t` skips the interstitial for large files."""
    import requests

    print(f"Downloading the official checkpoint from Google Drive ({GDRIVE_FILE_ID})...")
    with requests.get(GDRIVE_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("text/html"):
            raise RuntimeError(
                "Drive returned an HTML interstitial instead of the file — the share link may have "
                f"changed. Download it manually from {UNISPEECH_README} and pass --src."
            )
        total, seen = int(r.headers.get("content-length", 0)), 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                seen += len(chunk)
                if total:
                    print(f"\r  {seen / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        print()
    return dest


def resolve_source(src, pinned, cache_dir):
    """Return a local path to the checkpoint, preferring copies we already have on disk."""
    if src:
        print(f"Source: {src} (--src)")
        return src

    # 1. Already in the HF cache from an eval run.
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=pinned["WAVLM_SEED_TTS_REPO_ID"],
            filename=pinned["WAVLM_SEED_TTS_FILENAME"],
            local_files_only=True,
        )
        print(f"Source: {path} (cached copy of {pinned['WAVLM_SEED_TTS_REPO_ID']})")
        return path
    except Exception:
        pass

    # 2. Otherwise fetch from the authoritative upstream.
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, pinned["WAVLM_SEED_TTS_FILENAME"])
    if os.path.exists(dest) and os.path.getsize(dest) == EXPECTED_SIZE:
        print(f"Source: {dest} (existing download)")
        return dest
    return download_from_gdrive(dest)


def model_card(repo_id, pinned, sha256):
    filename = pinned["WAVLM_SEED_TTS_FILENAME"]
    return f"""---
license: mit
tags:
- speaker-verification
- speaker-similarity
- wavlm
- ecapa-tdnn
- tts-evaluation
---

# WavLM-Large + ECAPA-TDNN speaker verification (seed-tts-eval SIM model)

A **verbatim mirror** of Microsoft's speaker-verification checkpoint used as the SIM (speaker
similarity) metric by seed-tts-eval, and therefore by the Seed-TTS / F5-TTS / CosyVoice line of
papers. Re-hosted so TTS evaluation jobs can fetch it from the Hub; the weights are unmodified.

## Provenance

| | |
|---|---|
| Origin | [microsoft/UniSpeech → downstreams/speaker_verification]({UNISPEECH_README}) |
| Row in upstream's table | **WavLM large**, *Fix pre-train:* **No** — their best SV model |
| Reported EER | Vox1-O **0.431**, Vox1-E 0.538, Vox1-H 1.154 |
| Upstream download | [Google Drive](https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view) |
| File | `{filename}` ({EXPECTED_SIZE:,} bytes) |
| sha256 | `{sha256}` |
| License | MIT, per the UniSpeech repository |

*Fix pre-train: No* means the WavLM-Large backbone was fine-tuned during speaker-verification
training, so these backbone weights do **not** equal `microsoft/wavlm-large`.

This mirror exists because neither upstream URL is reachable from a job container: the Google Drive
link requires a confirm-token dance, and the Azure blob URL in seed-tts-eval's vendored copy of the
README has an expired SAS token.

## Contents

`{filename}` is a `torch.save` dict with two top-level keys:

- `model` — 711 tensors: `feature_extract.model.*` (the WavLM-Large backbone, in original-WavLM
  naming), the ECAPA-TDNN head (`feature_weight`, `layer1`-`layer4`, `conv`, `pooling`, `bn`,
  `linear`), and `loss_calculator.projection.weight` (the training-time AM-softmax classifier,
  unused at inference).
- `best_valid_eer` — an unset sentinel (`100.0`); ignore it.

## Usage

Upstream runs this via `s3prl` + `torch.hub`. The open TTS leaderboard instead rebuilds the backbone
as an HF `WavLMModel` and remaps the checkpoint's keys onto it — verified **bit-exact** (0.0 max
difference across all 25 hidden states and the final 256-d embedding) against UniSpeech's original
`WavLM.py`. See `transformers/score_similarity.py`:

```bash
python transformers/score_similarity.py \\
    --manifest_path=<manifest>.jsonl \\
    --sim_backend=wavlm_seed_tts \\
    --sim_ckpt_repo={repo_id}
```

## Scale

Cosine similarities from this model are **not** interchangeable with those from
`microsoft/wavlm-base-plus-sv`. Measured on the VoxCeleb1 clips shipped with UniSpeech:

| | this model | `wavlm-base-plus-sv` |
|---|---|---|
| same speaker | 0.60 - 0.69 | 0.89 - 0.96 |
| different speaker | -0.17 - 0.18 | 0.60 - 0.84 |

Only the values from this model are comparable with published seed-tts-eval SIM numbers.

## Citation

```bibtex
@article{{chen2022wavlm,
  title={{WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing}},
  author={{Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and others}},
  journal={{IEEE Journal of Selected Topics in Signal Processing}},
  year={{2022}}
}}
```
"""


def main(args):
    pinned = read_pinned_constants()
    expected_sha = pinned["WAVLM_SEED_TTS_SHA256"]
    filename = pinned["WAVLM_SEED_TTS_FILENAME"]
    print(f"Pinned digest (from {os.path.relpath(SCORE_SIMILARITY, REPO_ROOT)}): {expected_sha}")

    path = resolve_source(args.src, pinned, args.download_dir)

    size = os.path.getsize(path)
    if size != EXPECTED_SIZE:
        raise SystemExit(f"REFUSING: {path} is {size:,} bytes, expected {EXPECTED_SIZE:,}.")
    print("Verifying sha256 (a few seconds for 1.3 GB)...")
    actual = sha256_of(path)
    if actual != expected_sha:
        raise SystemExit(
            f"REFUSING to upload: sha256 mismatch.\n  expected {expected_sha}\n  actual   {actual}\n"
            "This is not the checkpoint the eval pins. Do not publish it."
        )
    print(f"  sha256 OK: {actual}")

    card = model_card(args.repo_id, pinned, actual)
    if args.dry_run:
        print(f"\n--dry_run: verified, nothing uploaded. Card that would be written to {args.repo_id}:\n")
        print(card)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"Authenticated as {who.get('name')}")

    url = api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    print(f"Repo ready: {url} (private={args.private})")

    print(f"Uploading {filename} ({size / 1e9:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=filename,
        repo_id=args.repo_id,
        commit_message=f"Mirror {filename} from microsoft/UniSpeech (sha256 {actual[:12]})",
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        commit_message="Add model card documenting provenance, contents and SIM scale",
    )

    # Confirm the Hub stored the exact bytes: its LFS oid IS the file's sha256.
    info = api.model_info(args.repo_id, files_metadata=True)
    remote = next((s for s in info.siblings if s.rfilename == filename), None)
    lfs = getattr(remote, "lfs", None) if remote is not None else None
    remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if remote_sha != actual:
        raise SystemExit(f"Upload verification FAILED: remote sha256 {remote_sha} != {actual}")
    print(f"  remote sha256 OK: {remote_sha}")

    print(
        f"\nDone. https://huggingface.co/{args.repo_id}\n"
        f"Point the eval at it with:  --sim_ckpt_repo={args.repo_id}\n"
        + (
            "The repo is PRIVATE, so every job that scores SIM needs an HF_TOKEN with read access "
            "(the sim_stage in scripts/tts_jobs_common.sh already forwards one).\n"
            if args.private
            else ""
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help=f"Destination repo (default {DEFAULT_REPO_ID}).")
    parser.add_argument(
        "--src",
        default=None,
        help="Local path to wavlm_large_finetune.pth. Default: reuse a cached copy, else download "
             "from the official Google Drive link.",
    )
    parser.add_argument(
        "--download_dir",
        default=os.path.join(REPO_ROOT, ".cache"),
        help="Where to put the download when no local copy exists.",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo private (default: public).")
    parser.add_argument("--dry_run", action="store_true", help="Verify the checkpoint and render the card; upload nothing.")
    args = parser.parse_args()

    sys.exit(main(args))
