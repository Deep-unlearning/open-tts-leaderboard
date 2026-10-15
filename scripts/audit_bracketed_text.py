"""Audit scored reference/ASR text in one model's local JSONL manifests.

Download manifests without audio first, for example:
    hf buckets sync hf://buckets/hf-audio/tts_leaderboard_h200/k2-fsa-OmniVoice \
        /tmp/omnivoice --include '*.jsonl'
    python scripts/audit_bracketed_text.py /tmp/omnivoice --output /tmp/audit.json

SIM forks are excluded. Reference counts deduplicate standard/voice-clone copies
by dataset, row position and text; predictions count each synthesis separately.
No normalization is applied, so counts describe the original text.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PAIRS = {"(": ")", "[": "]", "<": ">", "（": "）", "［": "］", "＜": "＞", "【": "】"}
BRACKETED_TEXT = re.compile("|".join(
    re.escape(left) + "[^" + re.escape(left + right) + "]*" + re.escape(right)
    for left, right in PAIRS.items()
))


def audit(directory, max_examples=3):
    manifests = []
    references = {}
    prediction_counts = Counter()
    for path in sorted(directory.glob("MODEL_*_DATASET_*.jsonl")):
        if path.stem.endswith("_wavlm_seed_tts"):
            continue
        raw = path.read_bytes()
        dataset = path.stem.split("_DATASET_", 1)[1].removesuffix("_voice_clone")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        fields = {}
        for field in ("text", "pred_text"):
            counts = Counter(rows=len(rows))
            examples = []
            for line, row in enumerate(rows, 1):
                text = row.get(field) or ""
                matches = BRACKETED_TEXT.findall(text)
                counts["nonempty"] += bool(text)
                counts["rows_with_brackets"] += bool(matches)
                counts["spans"] += len(matches)
                counts.update("pair_" + span[0] + PAIRS[span[0]] for span in matches)
                if matches and len(examples) < max_examples:
                    examples.append({"row": line, "text": text, "spans": matches})
                if field == "text":
                    references[dataset, line, text] = matches
            fields[field] = {**counts, "examples": examples}
            if field == "pred_text":
                prediction_counts.update(counts)
        manifests.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), **fields})
    if not manifests:
        raise ValueError(f"No MODEL_*_DATASET_*.jsonl manifests found in {directory}")
    reference_counts = Counter(rows=len(references))
    for matches in references.values():
        reference_counts["rows_with_brackets"] += bool(matches)
        reference_counts["spans"] += len(matches)
        reference_counts.update("pair_" + span[0] + PAIRS[span[0]] for span in matches)
    return {
        "directory": str(directory.resolve()),
        "manifest_count": len(manifests),
        "unique_references": dict(reference_counts),
        "predictions": dict(prediction_counts),
        "manifests": manifests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Local directory containing one model's manifests.")
    parser.add_argument("--output", type=Path, help="Save the full JSON report here; otherwise print it.")
    parser.add_argument("--max-examples", type=int, default=3, help="Examples per manifest/field (default: 3).")
    args = parser.parse_args()
    report = audit(args.directory, args.max_examples)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "manifests"}, ensure_ascii=False, indent=2))
        print(f"Full report: {args.output.resolve()}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
