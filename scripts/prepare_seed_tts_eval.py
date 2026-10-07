"""
Script to prepare the seed TTS evaluation data for the HF hub.

GitHub: https://github.com/BytedanceSpeech/seed-tts-eval
Download link: https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP/edit

Download via CLI:
```
pip install gdown --break-system-packages
gdown 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP -O seedtts_testset.tar
tar -xf seedtts_testset.tar
```

The test set is organized using meta files. Each line is pipe-separated:

    filename | prompt text | prompt audio | text to synthesize | ground truth audio (if exists)

Different tasks use different meta files. This script uploads the English and Chinese data:
  - Zero-shot TTS:              {en,zh}/meta.lst                     (4 fields, no ground truth)
  - Zero-shot TTS (hard case):  zh/hardcase.lst                      (4 fields, no ground truth, ZH only)
  - Zero-shot voice conversion: {en,zh}/non_para_reconstruct_meta.lst (5 fields, with ground truth)

Each task is pushed as a config of the dataset; each language is a split within it.
The ZH-only hard cases live under the `tts` config as a `zh_hard` split.

Usage:
```
python scripts/prepare_seed_tts_eval.py <hf_dataset_id> [--data-dir seedtts_testset] [--langs en zh]
```
"""

import argparse
from pathlib import Path

from datasets import Audio, Dataset, Features, Value


# One entry per meta file:
#   config: config name (Hub config)
#   meta: meta file relative to the language dir
#   has_ground_truth: whether the meta file has a 5th ground-truth-audio field
#   langs: languages for which this task exists
#   split: optional split-name override (defaults to the language), e.g. `zh_hard`
TASKS = [
    {"config": "tts", "meta": "meta.lst", "has_ground_truth": False, "langs": ["en", "zh"]},
    {"config": "tts", "meta": "hardcase.lst", "has_ground_truth": False, "langs": ["zh"], "split": "zh_hard"},
    {"config": "voice_conversion", "meta": "non_para_reconstruct_meta.lst", "has_ground_truth": True, "langs": ["en", "zh"]},
]


def parse_meta(meta_path: Path, lang_dir: Path, has_ground_truth: bool) -> list[dict]:
    """Parse a pipe-separated meta file into a list of records with absolute audio paths."""
    records = []
    with open(meta_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("|")
            expected = 5 if has_ground_truth else 4
            if len(fields) != expected:
                raise ValueError(
                    f"{meta_path}:{lineno} expected {expected} fields, got {len(fields)}: {line!r}"
                )

            record = {
                "id": fields[0],
                "prompt_text": fields[1],
                "prompt_audio": str(lang_dir / fields[2]),
                "text": fields[3],
            }
            if has_ground_truth:
                record["ground_truth_audio"] = str(lang_dir / fields[4])
            records.append(record)
    return records


def build_dataset(records: list[dict], has_ground_truth: bool) -> Dataset:
    feature_dict = {
        "id": Value("string"),
        "prompt_text": Value("string"),
        "prompt_audio": Audio(),
        "text": Value("string"),
    }
    if has_ground_truth:
        feature_dict["ground_truth_audio"] = Audio()

    columns = {key: [r[key] for r in records] for key in feature_dict}
    return Dataset.from_dict(columns, features=Features(feature_dict))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="HF dataset id to push to, e.g. 'username/seedtts-eval-en'")
    parser.add_argument(
        "--data-dir",
        default="seedtts_testset",
        help="Path to the extracted seedtts_testset directory (default: seedtts_testset).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en", "zh"],
        help="Language subdirectories to upload (default: en zh).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for task in TASKS:
        config_name = task["config"]
        has_ground_truth = task["has_ground_truth"]
        langs = [lang for lang in args.langs if lang in task["langs"]]
        for lang in langs:
            split = task.get("split", lang)
            lang_dir = data_dir / lang
            meta_path = lang_dir / task["meta"]
            if not meta_path.is_file():
                print(f"Skipping '{config_name}/{split}': {meta_path} not found.")
                continue

            print(f"Parsing '{config_name}/{split}' from {meta_path} ...")
            records = parse_meta(meta_path, lang_dir, has_ground_truth)
            dataset = build_dataset(records, has_ground_truth)
            print(f"  {len(dataset)} examples")

            print(f"Pushing config '{config_name}' split '{split}' to {args.repo_id} ...")
            dataset.push_to_hub(
                args.repo_id,
                config_name=config_name,
                # Always private — the license does not permit redistributing this test set.
                private=True,
                split=split,
            )
            print(f"  done ({config_name}/{split}).")

    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
