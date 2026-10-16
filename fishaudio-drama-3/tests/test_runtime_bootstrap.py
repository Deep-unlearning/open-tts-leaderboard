"""Fail-closed bootstrap checks that never reach installation or network commands."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
APPROVED_IMAGE = "lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75"


class BootstrapTests(unittest.TestCase):
    def run_bootstrap(self, **environment):
        return subprocess.run(
            ["bash", str(BACKEND / "bootstrap_runtime.sh")], check=False,
            env={**os.environ, **environment}, capture_output=True, text=True, timeout=15,
        )

    def test_moving_or_unapproved_image_is_rejected(self):
        result = self.run_bootstrap(DRAMA_RUNTIME_IMAGE="lmsysorg/sglang:latest")
        self.assertEqual(result.returncode, 2)
        self.assertIn("approved immutable", result.stderr)

    def test_moving_main_initialization_must_be_disabled(self):
        result = self.run_bootstrap(DRAMA_RUNTIME_IMAGE=APPROVED_IMAGE, SGLANG_OMNI_AUTO_CLONE="1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Moving-main", result.stderr)

    def test_failed_base_check_saves_a_private_receipt_before_any_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_bootstrap(
                DRAMA_RUNTIME_IMAGE=APPROVED_IMAGE, SGLANG_OMNI_AUTO_CLONE="0",
                PYTHON_BIN=sys.executable, DRAMA_BUCKET_MOUNT=directory,
                DRAMA_BOOTSTRAP_ROOT=str(root / "runtime"), CUDA_VERSION="unapproved",
            )
            self.assertNotEqual(result.returncode, 0)
            receipts = list((root / "fishaudio-drama-3/runtime_bootstrap").glob("*.json"))
            self.assertEqual(len(receipts), 1, result.stderr)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["phase"], "base-check")
            self.assertEqual(receipt["image"], APPROVED_IMAGE)
            self.assertEqual(receipt["expected_server_revision"], "12f65fa60fbb5217ad47f58b2e58b18092c818a1")
            self.assertIn("dependency_freeze", receipt)
            self.assertFalse((root / "runtime").exists())


if __name__ == "__main__":
    unittest.main()
