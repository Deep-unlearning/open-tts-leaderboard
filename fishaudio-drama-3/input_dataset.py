"""Verify a prepared private Parquet input mount without changing benchmark identity."""

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_dataset_source(args):
    """Return loader arguments and provenance, retaining the canonical output ID."""
    local_root = getattr(args, "dataset_local_root", None)
    if not local_root:
        return args, None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/(seed_tts_eval|cv3_eval)", args.dataset_path):
        raise ValueError("Private mounted inputs require an explicit canonical owner/dataset ID")
    key = args.dataset_path.split("/")[1]
    expected_config = "tts" if key == "seed_tts_eval" else "zero_shot"
    if args.dataset != expected_config:
        raise ValueError("Private input benchmark and config differ")
    root = Path(local_root).resolve(strict=True)
    provenance_path = root / "input_provenance.json"
    raw_provenance = provenance_path.read_bytes()
    provenance = json.loads(raw_provenance)
    if provenance.get("format_version") != 1:
        raise ValueError("Unsupported private input provenance version")
    dataset = provenance["datasets"][key]
    split_key = f"{args.dataset}/{args.split}"
    entry = dataset["splits"][split_key]
    expected_rows = ({"en": 1088, "zh": 2020}.get(args.split)
                     if key == "seed_tts_eval" else
                     (500 if args.split in ("en", "zh", "fr", "ja", "ko", "de", "es", "it", "ru") else None))
    if expected_rows is None or entry.get("rows") != expected_rows:
        raise ValueError("Private input row count differs from the full canonical split")
    relative = Path(entry["parquet_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Private input Parquet path must remain inside the mount")
    parquet = (root / relative).resolve(strict=True)
    dataset_root = (root / key).resolve(strict=True)
    if not dataset_root.is_relative_to(root) or not parquet.is_relative_to(dataset_root):
        raise ValueError("Private input Parquet path escapes its benchmark directory")
    if file_sha256(dataset_root / "README.md") != dataset.get("readme_sha256"):
        raise ValueError("Private input config mapping differs from preparation provenance")
    for field in ("parquet_sha256", "ordered_id_text_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get(field, "")):
            raise ValueError(f"Private input {field} must be an exact SHA256")
    if file_sha256(parquet) != entry["parquet_sha256"]:
        raise ValueError("Private input Parquet checksum differs from preparation provenance")
    loader_args = SimpleNamespace(**vars(args))
    loader_args.dataset_path = str(dataset_root)
    identity = {"type": "private_bucket_parquet", "dataset": key, "split": split_key,
                "provenance_sha256": hashlib.sha256(raw_provenance).hexdigest(),
                "prepared_split": entry, "source": dataset.get("source")}
    return loader_args, identity


def ordered_text_sha256(dataset, text_column):
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(json.dumps([str(row["id"]), row[text_column]], ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()
