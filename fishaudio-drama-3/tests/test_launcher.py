"""Offline launcher contracts; the hf CLI, model hub, and server are replaced locally."""

import base64
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
MODEL_REVISION = "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
SERVER_REVISION = "12f65fa60fbb5217ad47f58b2e58b18092c818a1"
APPROVED_IMAGE = "lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.capture = self.root / "submitted.jsonl"
        self.env = os.environ.copy()
        for name in (
            "MODEL_ID", "MODEL_REVISION", "SERVER_REVISION", "DATASET_SCOPE",
            "MAX_EVAL_SAMPLES", "JOB_TIMEOUT", "FIRST_JOB_TIMEOUT", "VOICE_CLONE", "TTFA_PROBE",
            "BATCH_SIZE", "WARMUP_STEPS", "FLAVOR", "DATASET_NAMESPACE",
            "DATASET_PATH", "CV3_EVAL_PATH", "ORG_NAME", "PYTHON_BIN",
            "TTS_SPACE", "TTS_IMAGE", "BOOTSTRAP_RUNTIME", "RESULTS_BUCKET", "RESULTS_PREFIX",
            "INPUTS_BUCKET", "INPUTS_PREFIX",
        ):
            self.env.pop(name, None)
        self.env.update(PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                        HF_CAPTURE_FILE=str(self.capture))
        self.executable("hf", """
import json, os, sys
with open(os.environ['HF_CAPTURE_FILE'], 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ.get('HF_MOCK_FAIL'):
    raise SystemExit(7)
if sys.argv[1:3] == ['jobs', 'inspect']:
    print(json.dumps([{'id': 'abc123', 'owner': {'name': 'test'},
                       'status': {'stage': os.environ.get('HF_REMOTE_STAGE', 'COMPLETED')}}]))
else:
    print('id=abc123 url=https://huggingface.co/jobs/test/abc123')
""")

    def executable(self, name, source):
        script = self.bin / name
        script.write_text(f"#!{sys.executable}\n{source}")
        script.chmod(0o755)
        return script

    def launch(self, *arguments, **environment):
        return subprocess.run(
            ["bash", str(BACKEND / "submit_jobs.sh"), *arguments],
            check=False, env={**self.env, **environment}, capture_output=True, text=True, timeout=15,
        )

    def submissions(self):
        if not self.capture.exists():
            return []
        return [arguments for line in self.capture.read_text().splitlines()
                if (arguments := json.loads(line))[:2] == ["jobs", "run"]]

    def test_default_is_single_bounded_preview_without_submission(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 serial job(s), scope=seed_en, rows per split=8", result.stdout)
        self.assertIn("flavor=h200 timeout=30m concurrency=1", result.stdout)
        self.assertIn("Prefix: <bucket root>", result.stdout)
        self.assertNotIn("zero_shot", result.stdout)
        self.assertEqual(self.submissions(), [])

    def test_execute_requires_explicit_destinations(self):
        result = self.launch("--execute")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires explicit TTS_SPACE or TTS_IMAGE, and RESULTS_BUCKET", result.stderr)
        self.assertEqual(self.submissions(), [])

    def test_first_job_can_use_a_tighter_remaining_timeout(self):
        result = self.launch("--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
                             DATASET_SCOPE="full", JOB_TIMEOUT="2h", FIRST_JOB_TIMEOUT="114m")
        self.assertEqual(result.returncode, 0, result.stderr)
        timeouts = [call[call.index("--timeout") + 1] for call in self.submissions()]
        self.assertEqual(timeouts, ["114m"] + ["2h"] * 10)
        self.assertIn("First job timeout: 114m", result.stdout)

    def test_invalid_first_timeout_fails_before_submission(self):
        result = self.launch("--execute", FIRST_JOB_TIMEOUT="114m; false")
        self.assertEqual(result.returncode, 2)
        self.assertIn("FIRST_JOB_TIMEOUT", result.stderr)
        self.assertEqual(self.submissions(), [])

    def test_result_prefix_mounts_the_exact_bucket_subfolder(self):
        result = self.launch(
            "--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
            RESULTS_PREFIX="fish-drama-3/full-20261015",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.submissions()
        self.assertEqual(len(calls), 1)
        self.assertIn(
            "hf://buckets/test/private/fish-drama-3/full-20261015:/results",
            calls[0],
        )
        self.assertIn("Prefix: fish-drama-3/full-20261015", result.stdout)

    def test_invalid_result_prefixes_are_rejected_before_submission(self):
        for prefix in (
            "../outside", "fish/../outside", "fish/./run", "/absolute", "fish//run", "fish/",
            "fish run", "fish\trun", "fish\nrun", "fish;run", "fish$(run)", "fish`run`",
            "fish:run", "fish\\run", "fish%2F..", "fish/*", "fish/&run", "fish/?run",
        ):
            with self.subTest(prefix=prefix):
                result = self.launch("--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
                                     RESULTS_PREFIX=prefix)
                self.assertEqual(result.returncode, 2)
                self.assertIn("RESULTS_PREFIX", result.stderr)
        self.assertEqual(self.submissions(), [])

    def test_wrong_model_and_revision_fail_before_submission(self):
        for environment in (
            {"MODEL_ID": "fishaudio/s2-pro"},
            {"MODEL_REVISION": "main"},
            {"SERVER_REVISION": "main"},
        ):
            with self.subTest(environment=environment):
                result = self.launch("--execute", **environment)
                self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.submissions(), [])

    def test_bounded_submission_injects_every_runtime_dependency(self):
        result = self.launch("--execute", TTS_SPACE="test/drama-image", RESULTS_BUCKET="test/private")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.submissions()
        self.assertEqual(len(calls), 1)
        arguments = calls[0]
        self.assertEqual(arguments[:6], ["jobs", "run", "--flavor", "h200", "--timeout", "30m"])
        self.assertIn("hf://buckets/test/private:/results", arguments)
        self.assertIn("hf.co/spaces/test/drama-image", arguments)
        command = arguments[-1]
        self.assertIn("--dataset_path bezzam/seed_tts_eval --dataset tts --split en", command)
        self.assertIn("--max_eval_samples 8", command)
        self.assertIn("--no-voice_clone", command)
        self.assertNotIn("--dataset_local_root", command)
        self.assertNotIn("bootstrap_runtime.sh", command)
        self.assertNotIn("SGLANG_OMNI_AUTO_CLONE=0", arguments)
        syntax = subprocess.run(["bash", "-n", "-c", command], check=False, capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        injected = dict(re.findall(r"echo '([^']+)' \| base64 -d \| gzip -dc > ([^ ]+)", command))
        payloads = {destination: gzip.decompress(base64.b64decode(encoded))
                    for encoded, destination in injected.items()}
        expected = {
            "/app/fishaudio-drama-3/run_eval.py": BACKEND / "run_eval.py",
            "/app/fishaudio-drama-3/adapter.py": BACKEND / "adapter.py",
            "/app/fishaudio-drama-3/input_dataset.py": BACKEND / "input_dataset.py",
            "/app/fishaudio-drama-3/run_job.sh": BACKEND / "run_job.sh",
            "/app/scripts/run_eval_utils.py": BACKEND.parent / "scripts/run_eval_utils.py",
            "/app/scripts/ttfa_probe.py": BACKEND.parent / "scripts/ttfa_probe.py",
        }
        self.assertEqual(payloads.keys(), expected.keys())
        for destination, source in expected.items():
            self.assertEqual(payloads[destination], source.read_bytes())

    def test_direct_image_bootstrap_is_explicit_and_immutable(self):
        result = self.launch("--execute", TTS_IMAGE=APPROVED_IMAGE, BOOTSTRAP_RUNTIME="true",
                             RESULTS_BUCKET="test/private")
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.submissions()[0]
        self.assertIn(APPROVED_IMAGE, arguments)
        self.assertIn("SGLANG_OMNI_AUTO_CLONE=0", arguments)
        self.assertIn("PYTHON_BIN=/opt/sglang/bin/python3", arguments)
        self.assertIn(f"DRAMA_RUNTIME_IMAGE={APPROVED_IMAGE}", arguments)
        command = arguments[-1]
        self.assertIn("bash /app/fishaudio-drama-3/bootstrap_runtime.sh &&", command)
        self.assertLess(command.rfind("bash /app/fishaudio-drama-3/bootstrap_runtime.sh"),
                        command.rfind("bash /app/fishaudio-drama-3/run_job.sh"))
        self.assertIn("Runtime bootstrap: true", result.stdout)
        syntax = subprocess.run(["bash", "-n", "-c", command], check=False, capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_invalid_images_and_bootstrap_combinations_fail_before_submission(self):
        invalid = (
            {"TTS_IMAGE": "lmsysorg/sglang:latest"},
            {"TTS_IMAGE": "lmsysorg/sglang@sha256:abc"},
            {"TTS_IMAGE": "lmsysorg/../sglang@sha256:" + "a" * 64},
            {"TTS_IMAGE": APPROVED_IMAGE + ";run"},
            {"TTS_IMAGE": APPROVED_IMAGE, "TTS_SPACE": "test/image"},
            {"TTS_SPACE": "test/image", "BOOTSTRAP_RUNTIME": "true"},
            {"TTS_IMAGE": "lmsysorg/sglang@sha256:" + "a" * 64, "BOOTSTRAP_RUNTIME": "true"},
            {"TTS_IMAGE": APPROVED_IMAGE, "BOOTSTRAP_RUNTIME": "yes"},
            {"TTS_IMAGE": APPROVED_IMAGE, "BOOTSTRAP_RUNTIME": "true", "PYTHON_BIN": "other-python"},
        )
        for environment in invalid:
            with self.subTest(environment=environment):
                result = self.launch("--execute", RESULTS_BUCKET="test/private", **environment)
                self.assertEqual(result.returncode, 2)
        self.assertEqual(self.submissions(), [])

    def test_inputs_mount_is_read_only_and_preserves_canonical_dataset_identity(self):
        result = self.launch(
            "--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
            INPUTS_BUCKET="test/private-inputs", INPUTS_PREFIX="fish-drama-3/inputs",
            DATASET_PATH="test/seed_tts_eval",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.submissions()[0]
        self.assertIn("hf://buckets/test/private-inputs/fish-drama-3/inputs:/inputs:ro", arguments)
        self.assertIn("--dataset_local_root /inputs", arguments[-1])
        self.assertIn("--dataset_path test/seed_tts_eval", arguments[-1])
        self.assertIn("Inputs prefix: fish-drama-3/inputs (read-only at /inputs)", result.stdout)

    def test_inputs_bucket_root_can_be_mounted_without_a_prefix(self):
        result = self.launch("--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
                             INPUTS_BUCKET="test/inputs")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("hf://buckets/test/inputs:/inputs:ro", self.submissions()[0])

    def test_invalid_inputs_are_rejected_before_submission(self):
        for environment in (
            {"INPUTS_PREFIX": "fish/inputs"}, {"INPUTS_BUCKET": "../inputs"},
            {"INPUTS_BUCKET": "test/inputs", "INPUTS_PREFIX": "../outside"},
            {"INPUTS_BUCKET": "test/inputs", "INPUTS_PREFIX": "fish/inputs;run"},
            {"INPUTS_BUCKET": "test/inputs", "INPUTS_PREFIX": "fish inputs"},
        ):
            with self.subTest(environment=environment):
                result = self.launch("--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
                                     **environment)
                self.assertEqual(result.returncode, 2)
        self.assertEqual(self.submissions(), [])

    def test_full_scope_is_explicit_and_job_failures_are_not_retried(self):
        result = self.launch("--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
                             DATASET_SCOPE="full", MAX_EVAL_SAMPLES="-1", HF_MOCK_FAIL="1")
        self.assertEqual(result.returncode, 7)
        self.assertIn("11 serial job(s)", result.stdout)
        self.assertEqual(len(self.submissions()), 1)

    def test_remote_failure_stops_full_scope_even_when_run_cli_returns_zero(self):
        result = self.launch("--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
                             DATASET_SCOPE="full", HF_REMOTE_STAGE="ERROR")
        self.assertEqual(result.returncode, 1)
        self.assertIn("not COMPLETED", result.stderr)
        self.assertEqual(len(self.submissions()), 1)

    def test_cv3_ttfa_is_single_request_and_explicit_hardware(self):
        result = self.launch("--execute", TTS_SPACE="test/image", RESULTS_BUCKET="test/private",
                             DATASET_SCOPE="cv3_en", TTFA_PROBE="6", FLAVOR="a100-large")
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.submissions()[0]
        self.assertIn("JOB_FLAVOR=a100-large", arguments)
        self.assertIn("--ttfa_probe 6", arguments[-1])
        self.assertIn("--dataset zero_shot --split en", arguments[-1])
        result = self.launch(TTFA_PROBE="6", BATCH_SIZE="2")
        self.assertNotEqual(result.returncode, 0)

    def runtime(self, *, revision=SERVER_REVISION, client_failure=False, bucket_files=(),
                arguments=("--ttfa_probe", "6")):
        backend = self.root / "backend"
        backend.mkdir()
        shutil.copyfile(BACKEND / "run_job.sh", backend / "run_job.sh")
        source = self.root / "source"
        package = source / "sglang_omni"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "cli.py").write_text("""
import json, os, signal, sys
from pathlib import Path
Path(os.environ['SERVER_ARGS']).write_text(json.dumps(sys.argv[1:]))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.pause()
""")
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        bucket = self.root / "bucket"
        bucket_model = bucket / "fishaudio-drama-3"
        bucket_model.mkdir(parents=True)
        for filename in bucket_files:
            (bucket_model / filename).write_text("{}")
        (source / "huggingface_hub.py").write_text("""
import os
from pathlib import Path
def snapshot_download(repo_id, revision):
    assert repo_id == 'fishaudio/drama-3'
    assert revision == 'e0a9218aa48888270f06ef1ed9ea67622c0e17d2'
    Path(os.environ['SNAPSHOT_CALLED']).touch()
    return os.environ['MODEL_SNAPSHOT']
""")
        (backend / "run_eval.py").write_text("""
import json, os, sys, time
from pathlib import Path
for attempt in range(200):
    if Path(os.environ['SERVER_ARGS']).exists():
        break
    time.sleep(0.01)
else:
    raise SystemExit('Mock server did not start')
Path(os.environ['CLIENT_CAPTURE_FILE']).write_text(json.dumps(sys.argv[1:]))
if os.environ.get('CLIENT_FAILURE'):
    if os.environ['CLIENT_FAILURE'] == 'partial':
        output = Path('results/fishaudio-drama-3')
        output.mkdir()
        (output / 'MODEL_partial.run.json').write_text(json.dumps({'status': 'running', 'completed_samples': 1}))
        (output / 'output_retained.wav').write_bytes(b'retained audio')
    raise SystemExit(9)
output = Path('results/fishaudio-drama-3')
output.mkdir()
(output / 'TTFA_MODEL_test.json').write_text('{}')
""")
        self.executable("git", """
import os, sys
if sys.argv[-1] == '--show-toplevel':
    print(os.environ['SOURCE_ROOT'])
elif sys.argv[-1] == 'HEAD':
    print(os.environ['MOCK_REVISION'])
""")
        self.executable("cp", """
import os, shutil, sys
from pathlib import Path
source = Path(sys.argv[-2])
shutil.copytree(source, Path(sys.argv[-1]) / source.name, dirs_exist_ok=True)
Path(os.environ['OUTPUT_COPIED']).touch()
""")
        environment = {**self.env, "PYTHON_BIN": sys.executable, "PYTHONPATH": str(source),
                       "SOURCE_ROOT": str(source), "MOCK_REVISION": revision,
                       "MODEL_SNAPSHOT": str(snapshot), "SERVER_ARGS": str(self.root / "server.json"),
                       "DRAMA_BUCKET_MOUNT": str(bucket),
                       "CLIENT_CAPTURE_FILE": str(self.root / "client.json"),
                       "SNAPSHOT_CALLED": str(self.root / "snapshot-called"),
                       "OUTPUT_COPIED": str(self.root / "output-copied")}
        if client_failure:
            environment["CLIENT_FAILURE"] = "partial" if client_failure == "partial" else "1"
        result = subprocess.run(["bash", str(backend / "run_job.sh"), *arguments],
                                check=False, env=environment, capture_output=True, text=True, timeout=10)
        return result, backend

    def test_runtime_uses_verified_snapshot_and_tags_sidecar(self):
        result, backend = self.runtime()
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = json.loads((self.root / "server.json").read_text())
        self.assertEqual(arguments[0], "serve")
        self.assertIn(str(self.root / "snapshot"), arguments)
        self.assertIn("--model-name", arguments)
        self.assertIn("fishaudio/drama-3", arguments)
        self.assertIn(str(backend / "results"), arguments)
        self.assertTrue((backend / "results/fishaudio-drama-3/TTFA_MODEL_test__sglang__h200.json").exists())
        self.assertTrue((self.root / "output-copied").exists())

    def test_wrong_runtime_revision_fails_before_download(self):
        result, _ = self.runtime(revision="wrong")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("revision mismatch", result.stderr)
        self.assertFalse((self.root / "snapshot-called").exists())
        self.assertFalse((self.root / "server.json").exists())

    def test_client_failure_does_not_copy_results(self):
        result, _ = self.runtime(client_failure=True)
        self.assertEqual(result.returncode, 9)
        self.assertFalse((self.root / "output-copied").exists())

    def test_client_failure_preserves_partial_audio_and_failed_receipt(self):
        result, _ = self.runtime(client_failure="partial")
        self.assertEqual(result.returncode, 9, result.stderr)
        model = self.root / "bucket" / "fishaudio-drama-3"
        self.assertEqual((model / "output_retained.wav").read_bytes(), b"retained audio")
        receipt = json.loads((model / "MODEL_partial.run.json").read_text())
        self.assertEqual(receipt, {"status": "failed", "completed_samples": 1, "execution_exit_status": 9})

    def test_existing_manifest_blocks_before_download_or_server(self):
        result, _ = self.runtime(
            arguments=(), bucket_files=("MODEL_fishaudio-drama-3_DATASET_bezzam-seed_tts_eval_tts_en.jsonl",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing a duplicate run", result.stderr)
        self.assertFalse((self.root / "snapshot-called").exists())
        self.assertFalse((self.root / "server.json").exists())

    def test_existing_run_metadata_blocks_before_download(self):
        result, _ = self.runtime(
            arguments=(), bucket_files=("MODEL_fishaudio-drama-3_DATASET_bezzam-seed_tts_eval_tts_en.run.json",))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "snapshot-called").exists())

    def test_existing_tagged_ttfa_blocks_before_download(self):
        result, _ = self.runtime(bucket_files=(
            "TTFA_MODEL_fishaudio-drama-3_DATASET_bezzam-seed_tts_eval_tts_en__sglang__h200.json",))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "snapshot-called").exists())

    def test_other_clone_mode_remains_allowed(self):
        result, _ = self.runtime(arguments=(), bucket_files=(
            "MODEL_fishaudio-drama-3_DATASET_bezzam-seed_tts_eval_tts_en_voice_clone.run.json",))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_other_dataset_remains_allowed(self):
        result, _ = self.runtime(arguments=(), bucket_files=(
            "MODEL_fishaudio-drama-3_DATASET_bezzam-cv3_eval_zero_shot_en.run.json",))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_probe_remains_allowed_when_offline_results_exist(self):
        result, _ = self.runtime(bucket_files=(
            "MODEL_fishaudio-drama-3_DATASET_bezzam-seed_tts_eval_tts_en.run.json",))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_probe_on_other_hardware_remains_allowed(self):
        result, _ = self.runtime(bucket_files=(
            "TTFA_MODEL_fishaudio-drama-3_DATASET_bezzam-seed_tts_eval_tts_en__sglang__a100-large.json",))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bucket_resume_is_rejected_before_download(self):
        result, _ = self.runtime(arguments=("--resume",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Bucket resume is not supported", result.stderr)
        self.assertFalse((self.root / "snapshot-called").exists())


if __name__ == "__main__":
    unittest.main()
