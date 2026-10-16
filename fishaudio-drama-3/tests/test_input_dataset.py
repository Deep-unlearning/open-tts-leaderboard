"""Private input identity and integrity checks, without benchmark data or downloads."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from input_dataset import ordered_text_sha256, resolve_dataset_source


class InputDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset_root = self.root / "seed_tts_eval"
        self.dataset_root.mkdir()
        self.parquet = self.dataset_root / "en.parquet"
        self.parquet.write_bytes(b"fake parquet fixture; never decoded")
        self.readme = self.dataset_root / "README.md"
        self.readme.write_text("test config mapping")
        self.entry = {"rows": 1088, "parquet_file": "seed_tts_eval/en.parquet",
                      "parquet_sha256": hashlib.sha256(self.parquet.read_bytes()).hexdigest(),
                      "ordered_id_text_sha256": "a" * 64}
        self.provenance = {"format_version": 1, "datasets": {"seed_tts_eval": {
            "source": {"revision": "public-source-pin"}, "splits": {"tts/en": self.entry},
            "readme_sha256": hashlib.sha256(self.readme.read_bytes()).hexdigest()}}}
        self.args = SimpleNamespace(dataset_path="owner/seed_tts_eval", dataset="tts", split="en",
                                    dataset_local_root=str(self.root))
        self.write_provenance()

    def write_provenance(self):
        self.metadata = self.root / "input_provenance.json"
        self.metadata.write_text(json.dumps(self.provenance))

    def test_default_remote_loader_is_unchanged(self):
        self.args.dataset_local_root = None
        actual, source = resolve_dataset_source(self.args)
        self.assertIs(actual, self.args)
        self.assertIsNone(source)

    def test_mounted_inputs_keep_canonical_output_identity_and_record_hashes(self):
        actual, source = resolve_dataset_source(self.args)
        self.assertEqual(actual.dataset_path, str(self.dataset_root.resolve()))
        self.assertEqual(self.args.dataset_path, "owner/seed_tts_eval")
        self.assertEqual(source["prepared_split"], self.entry)
        self.assertEqual(source["source"], {"revision": "public-source-pin"})
        self.assertEqual(source["provenance_sha256"], hashlib.sha256(self.metadata.read_bytes()).hexdigest())

    def test_modified_parquet_is_rejected(self):
        self.parquet.write_bytes(b"changed reference audio/text")
        with self.assertRaisesRegex(ValueError, "checksum"):
            resolve_dataset_source(self.args)

    def test_modified_loader_mapping_is_rejected(self):
        self.readme.write_text("different config")
        with self.assertRaisesRegex(ValueError, "config mapping"):
            resolve_dataset_source(self.args)

    def test_noncanonical_alias_partial_split_and_bad_config_are_rejected(self):
        self.args.dataset_path = "owner/seed_tts_eval_api_texts"
        with self.assertRaisesRegex(ValueError, "canonical"):
            resolve_dataset_source(self.args)
        self.args.dataset_path = "owner/seed_tts_eval"
        self.args.dataset = "zero_shot"
        with self.assertRaisesRegex(ValueError, "config differ"):
            resolve_dataset_source(self.args)
        self.args.dataset = "tts"
        self.entry["rows"] = 8
        self.write_provenance()
        with self.assertRaisesRegex(ValueError, "full canonical split"):
            resolve_dataset_source(self.args)

    def test_parquet_cannot_escape_mount_or_benchmark_directory(self):
        for relative in ("../outside.parquet", "/tmp/outside.parquet", "elsewhere/en.parquet"):
            with self.subTest(relative=relative):
                self.entry["parquet_file"] = relative
                self.write_provenance()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    resolve_dataset_source(self.args)

    def test_id_text_digest_preserves_order_unicode_and_spoken_brackets(self):
        rows = [{"id": 1, "text": "你好 (spoken) [spoken]."}, {"id": 2, "text": "Next."}]
        expected = hashlib.sha256(b"".join(
            json.dumps([str(row["id"]), row["text"]], ensure_ascii=False).encode() + b"\n" for row in rows
        )).hexdigest()
        self.assertEqual(ordered_text_sha256(rows, "text"), expected)
        self.assertNotEqual(ordered_text_sha256(rows[::-1], "text"), expected)

    def test_missing_input_mount_stops_job_before_gpu_runtime_import(self):
        result = subprocess.run(
            ["bash", str(Path(__file__).resolve().parents[1] / "run_job.sh"),
             "--dataset_local_root", str(self.root / "missing-inputs")],
            env={**os.environ, "PYTHON_BIN": sys.executable, "DRAMA_BUCKET_MOUNT": str(self.root)},
            capture_output=True, text=True, check=False, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing-inputs", result.stderr)
        self.assertNotIn("No module named 'sglang_omni'", result.stderr)


if __name__ == "__main__":
    unittest.main()
