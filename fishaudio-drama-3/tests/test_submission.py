"""Local synthetic submission contracts; no CSVs, audio, providers, or model jobs."""

import copy
import hashlib
import io
import json
import math
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import prepare_submission as submission

CANONICAL_ROW_COUNTS = dict(submission.EXPECTED_ROW_COUNTS)


class CanonicalRowCountTests(unittest.TestCase):
    def test_row_counts_cover_every_published_split(self):
        self.assertEqual(submission.EXPECTED_ROW_COUNTS, {
            ("seed_tts_eval", "tts", "en"): 1088,
            ("seed_tts_eval", "tts", "zh"): 2020,
            **{("cv3_eval", "zero_shot", language): 500
               for language in ("en", "zh", "fr", "ja", "ko", "de", "es", "it", "ru")},
        })
        self.assertEqual(set(submission.EXPECTED_ROW_COUNTS), set(submission.EXPECTED_SPLITS))


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "repo"
        normalizer = self.repo / "normalizer"
        normalizer.mkdir(parents=True)
        (normalizer / "b.py").write_text("b = 2\n", encoding="utf-8")
        (normalizer / "a.py").write_text("a = 1\n", encoding="utf-8")
        (normalizer / "normalizer.py").write_text("normalize = True\n", encoding="utf-8")
        self.results = self.root / "results" / submission.MODEL_SAFE
        self.results.mkdir(parents=True)
        self.output = self.root / "submission"
        self.sha256 = hashlib.sha256((normalizer / "normalizer.py").read_bytes()).hexdigest()
        self.aggregate = submission.normalizer_aggregate_sha256(self.repo)
        self.addCleanup(patch.stopall)
        patch.object(submission, "REPO_ROOT", self.repo).start()
        patch.object(submission, "AUDITED_NORMALIZER_AGGREGATE", self.aggregate).start()
        patch.object(submission, "EXPECTED_ROW_COUNTS", {identity: 1 for identity in submission.EXPECTED_SPLITS}).start()

    def write_json(self, path, value):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def populate(self, voice_clone=False, limit=None):
        self.runs = []
        suffix = "_voice_clone" if voice_clone else ""
        for (benchmark, config, language), target in submission.EXPECTED_SPLITS.items():
            dataset_path = "local/" + benchmark
            stem = (f"MODEL_{submission.MODEL_SAFE}_DATASET_{dataset_path.replace('/', '-')}_"
                    f"{config}_{language}{suffix}")
            receipt_path = self.results / (stem + ".run.json")
            manifest = self.results / (stem + ".jsonl")
            rows = [{"audio_filepath": f"{benchmark}_{language}/output_0.wav", "text": "Fixture target.",
                     "pred_text": "Fixture transcript.", "duration": 1.0, "time": 0.5}]
            if voice_clone:
                rows[0].update(prompt_audio_filepath=f"{benchmark}_{language}/prompt_0.wav")
                if target == "seed_tts":
                    rows[0].update(sim=0.7, sim_model=submission.XVECTOR_MODEL)
            receipt = {"status": "complete", "completed_samples": 1,
                       "settings": {**submission.PINS, "dataset_path": dataset_path,
                                    "dataset": config, "split": language, "voice_clone": voice_clone,
                                    "expected_samples": 1, "max_eval_samples": limit}}
            self.write_json(receipt_path, receipt)
            self.write_rows(manifest, rows)
            fork = self.results / (stem + submission.SIM_SUFFIX + ".jsonl")
            sim_rows = copy.deepcopy(rows)
            if voice_clone:
                sim_rows[0].update(sim=0.8, sim_model=submission.SIM_MODEL)
                self.write_rows(fork, sim_rows)
            self.runs.append((receipt_path, receipt, manifest, rows, fork, sim_rows))

    def prepare(self, voice_clone=False):
        return submission.prepare_plan(self.results, self.output, self.sha256, voice_clone)

    def test_fixed_voice_plan_contains_only_an_explicit_unexecuted_dry_run(self):
        self.populate()
        path = self.prepare()
        payload = json.loads(path.read_text())
        command = payload["publisher_argv"]
        self.assertEqual(shlex.split(payload["publisher_shell_command"]), command)
        self.assertFalse(payload["publisher_executed"])
        self.assertNotIn("--open_pr", command)
        self.assertIn("--skip_sync", command)
        self.assertEqual(command[command.index("--local_dir") + 1], str(self.results.parent))
        self.assertEqual(command[command.index("--out_dir") + 1], str(self.output / "csv"))
        self.assertEqual(payload["targets"], ["seed_tts", "cv3"])
        self.assertEqual(len(payload["validated_runs"]), 11)
        self.assertEqual(payload["normalizer_sha256"], self.sha256)
        self.assertEqual(payload["normalizer_aggregate_sha256"], self.aggregate)
        self.assertEqual(payload["normalizer_hash_file"], "normalizer/normalizer.py")
        self.assertIn(submission.MODEL_REVISION, payload["model"]["license"])
        self.assertTrue(any("private leaderboard CSVs" in note for note in payload["notes"]))
        self.assertEqual(list(self.output.iterdir()), [path])
        self.assertFalse((self.output / "csv").exists())

    def test_normalizer_hash_has_the_generation_aggregate_format(self):
        expected = hashlib.sha256(b"a.py\0a = 1\nb.py\0b = 2\nnormalizer.py\0normalize = True\n").hexdigest()
        self.assertEqual(self.aggregate, expected)
        with self.assertRaisesRegex(ValueError, "differs"):
            submission.prepare_plan(self.results, self.output, "0" * 64)
        self.assertFalse(self.output.exists())
        (self.repo / "normalizer" / "a.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            self.prepare()

    def test_all_normalizer_hash_arguments_are_required_and_well_formed(self):
        for bad in (None, "", "unknown", "z" * 64):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "audited"):
                submission.prepare_plan(self.results, self.output, bad)
        with patch.object(sys, "argv", ["prepare_submission.py", "--results_dir", str(self.results),
                                       "--output_dir", str(self.output)]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            submission.main()
        self.assertEqual(error.exception.code, 2)

    def test_voice_cloning_requires_both_published_sim_scales_for_seed(self):
        self.populate(voice_clone=True)
        payload = json.loads(self.prepare(True).read_text())
        self.assertEqual(payload["targets"], ["seed_tts_voice_clone", "cv3_voice_clone"])
        self.assertTrue(all("sim_manifest" in run for run in payload["validated_runs"]))
        _, _, manifest, rows, _, _ = self.runs[0]
        rows[0]["sim_model"] = submission.SIM_MODEL
        self.write_rows(manifest, rows)
        with self.assertRaisesRegex(ValueError, "microsoft/wavlm-base-plus-sv"):
            self.prepare(True)

    def test_fork_requires_complete_valid_scores_and_matching_asr_rows(self):
        self.populate(voice_clone=True)
        _, _, _, _, fork, original = self.runs[-1]
        for field, bad in (("sim", ""), ("sim", None), ("sim", True), ("sim", math.nan),
                           ("sim_model", submission.XVECTOR_MODEL), ("pred_text", "Changed transcript.")):
            rows = copy.deepcopy(original)
            rows[0][field] = bad
            self.write_rows(fork, rows)
            with self.subTest(field=field, bad=bad), self.assertRaisesRegex(ValueError, "SIM"):
                self.prepare(True)
        fork.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare(True)

    def test_incomplete_smoke_and_wrong_pin_receipts_are_rejected(self):
        self.populate()
        path, original, *_ = self.runs[0]
        changes = [("max_eval_samples", 8), ("model_id", "fishaudio/s2-pro"),
                   ("model_revision", "old"), ("server_revision", "old"),
                   ("codec_revision", "old"), ("expected_samples", 2)]
        for field, value in changes:
            receipt = copy.deepcopy(original)
            receipt["settings"][field] = value
            self.write_json(path, receipt)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()
        for field, value in (("status", "running"), ("completed_samples", 0)):
            receipt = copy.deepcopy(original)
            receipt[field] = value
            self.write_json(path, receipt)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_truncated_full_receipts_are_rejected_before_manifests_for_every_transport(self):
        with patch.object(submission, "EXPECTED_ROW_COUNTS", CANONICAL_ROW_COUNTS), \
                patch.object(submission, "read_manifest") as read_manifest:
            for (benchmark, config, language), expected in CANONICAL_ROW_COUNTS.items():
                for dataset_source in (None, {"type": "private_bucket_parquet"}):
                    with self.subTest(benchmark=benchmark, language=language, source=dataset_source):
                        for path in self.results.glob("*.run.json"):
                            path.unlink()
                        dataset_path = "local/" + benchmark
                        stem = (f"MODEL_{submission.MODEL_SAFE}_DATASET_{dataset_path.replace('/', '-')}_"
                                f"{config}_{language}")
                        receipt = {"status": "complete", "completed_samples": 1, "settings": {
                            **submission.PINS, "dataset_path": dataset_path, "dataset": config,
                            "split": language, "voice_clone": False, "max_eval_samples": -1,
                            "expected_samples": 1, "dataset_source": dataset_source,
                        }}
                        self.write_json(self.results / (stem + ".run.json"), receipt)
                        with self.assertRaisesRegex(ValueError, f"requires {expected} rows"):
                            self.prepare()
                        read_manifest.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_nonpositive_full_split_limits_are_accepted(self):
        self.populate(limit=0)
        self.prepare()
        for path, receipt, *_ in self.runs:
            receipt["settings"]["max_eval_samples"] = -1
            self.write_json(path, receipt)
        self.prepare()

    def test_every_required_split_and_unambiguous_receipts_are_required(self):
        self.populate()
        path, receipt, *_ = self.runs[-1]
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing.*cv3_eval/zero_shot/ru"):
            self.prepare()
        self.write_json(path, receipt)
        duplicate = self.results / ("duplicate.run.json")
        self.write_json(duplicate, receipt)
        with self.assertRaisesRegex(ValueError, "filename|duplicate"):
            self.prepare()

    def test_bad_asr_timing_and_duplicate_manifest_rows_are_rejected(self):
        self.populate()
        _, _, path, original, *_ = self.runs[0]
        for field, bad in (("pred_text", "  "), ("duration", 0), ("duration", math.inf),
                           ("time", -1), ("time", True), ("time", "0.5")):
            rows = copy.deepcopy(original)
            rows[0][field] = bad
            self.write_rows(path, rows)
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                self.prepare()
        self.write_rows(path, original * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare()

    def test_completed_asr_deletion_requires_matching_receipt_and_keeps_the_row(self):
        self.populate()
        _, _, manifest, rows, *_ = self.runs[0]
        rows[0]["pred_text"] = ""
        self.write_rows(manifest, rows)
        with self.assertRaisesRegex(ValueError, "successful ASR receipt"):
            self.prepare()
        source = self.repo / "transformers" / "transcribe.py"
        source.parent.mkdir()
        source.write_text("asr_source_fixture = True\n")
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        prediction_hash = hashlib.sha256(json.dumps(
            [rows[0]["audio_filepath"], ""], ensure_ascii=False,
        ).encode() + b"\n").hexdigest()
        proof = {"status": "complete", "stage": "asr", "manifest": manifest.name, "rows": 1,
                 "asr_model_id": "Qwen/Qwen3-ASR-1.7B-hf", "asr_model_resolved_sha": "b" * 40,
                 "source_revision": "c" * 40, "asr_source_sha256": source_sha256,
                 "source_sha256": source_sha256, "ordered_prediction_sha256": prediction_hash}
        receipt_path = manifest.parent / "scoring_receipts" / (manifest.stem + ".asr.json")
        receipt_path.parent.mkdir()
        self.write_json(receipt_path, proof)
        self.prepare()
        self.assertEqual(submission.read_manifest(manifest, 1, False)[rows[0]["audio_filepath"]]["pred_text"], "")
        proof["ordered_prediction_sha256"] = "0" * 64
        self.write_json(receipt_path, proof)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.prepare()

    def test_orphan_publisher_manifest_cannot_bypass_receipt_checks(self):
        self.populate()
        _, _, manifest, rows, *_ = self.runs[0]
        orphan = self.results / manifest.name.replace("_en.jsonl", "_zh_hard.jsonl")
        self.write_rows(orphan, rows)
        with self.assertRaisesRegex(ValueError, "unvalidated/orphan"):
            self.prepare()

    def test_nested_manifest_and_fixed_voice_fork_cannot_override_checked_rows(self):
        self.populate()
        _, _, manifest, rows, fork, *_ = self.runs[0]
        self.write_rows(fork, rows)
        with self.assertRaisesRegex(ValueError, "fixed-voice SIM fork"):
            self.prepare()
        fork.unlink()
        backup = self.results / "backup"
        backup.mkdir()
        self.write_rows(backup / manifest.name, rows)
        with self.assertRaisesRegex(ValueError, "Nested benchmark"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
