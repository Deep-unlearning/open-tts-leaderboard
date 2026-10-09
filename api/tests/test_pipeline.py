"""Offline contract checks for API orchestration and independent result export."""

import base64
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))
import run_pipeline
import score_results


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.models = ModuleType("api.models")
        self.models.MODELS = {
            "fish/s2-pro": SimpleNamespace(languages=("en", "zh", "fr"), reference_mode="inline"),
            "example/voice": SimpleNamespace(languages=("en",), reference_mode=None),
        }
        self.models.get_model = self.models.MODELS.__getitem__
        self.modules_patch = patch.dict(sys.modules, {"api.models": self.models})
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.modules_patch.start()
        self.env_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.addCleanup(self.env_patch.stop)

    def args(self, *flags):
        return run_pipeline.make_parser().parse_args(["--models", "fish/s2-pro", *flags])

    def test_generation_only_needs_no_bucket(self):
        args = self.args("--stages", "generate", "--max_eval_samples", "8")
        run_pipeline.validate_args(args)
        config = run_pipeline.dataset_configs(args)[0]
        command = run_pipeline.generation_command(args, "fish/s2-pro", config)
        self.assertEqual(command[command.index("--max_eval_samples") + 1], "8")
        self.assertNotIn("HF_TOKEN", command)

    def test_remote_stages_require_separate_explicit_bucket(self):
        with self.assertRaisesRegex(ValueError, "dedicated API"):
            run_pipeline.validate_args(self.args())
        with self.assertRaisesRegex(ValueError, "separate bucket"):
            run_pipeline.validate_args(self.args("--results_bucket", run_pipeline.H200_BUCKET))

    def test_clone_capability_validated_before_generation(self):
        args = run_pipeline.make_parser().parse_args(["--models", "example/voice", "--stages", "generate", "--voice_clone"])
        with self.assertRaisesRegex(ValueError, "no inline"):
            run_pipeline.validate_args(args)

    def test_dataset_configs_match_shared_manifest_names(self):
        args = self.args("--only_langs", "en", "fr")
        configs = run_pipeline.dataset_configs(args)
        self.assertEqual([(c.dataset, c.split, c.language) for c in configs],
                         [("tts", "en", "en"), ("zero_shot", "en", "en"), ("zero_shot", "fr", "fr")])
        path = run_pipeline.manifest_path("fish/s2-pro", configs[0], True)
        self.assertEqual(path.name, "MODEL_fish-s2-pro_DATASET_bezzam-seed_tts_eval_tts_en_voice_clone.jsonl")

    def test_injection_is_portable_and_quotes_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.py"
            source.write_text("print('hello')\n", encoding="utf-8")
            snippet = run_pipeline.injected_script(source, "/app/path with spaces/script.py")
            tokens = shlex.split(snippet)
            self.assertEqual(tokens[:2], ["python", "-c"])
            self.assertIn(base64.b64encode(source.read_bytes()).decode(), tokens[2])
            self.assertNotIn("base64 -w0", snippet)

    def test_pipeline_orders_generation_upload_jobs_download_score(self):
        args = self.args("--datasets", "seed_tts", "--only_langs", "en", "--results_bucket", "test/api-results",
                         "--voice_clone", "--dry_run")
        with patch.object(run_pipeline, "run_command") as runner:
            run_pipeline.run_pipeline(args)
        commands = [call.args[0] for call in runner.call_args_list]
        self.assertEqual(len(commands), 6)
        self.assertTrue(commands[0][1].endswith("run_eval.py"))
        self.assertEqual(commands[1][:3], ["hf", "buckets", "sync"])
        self.assertEqual(commands[2][:3], ["hf", "jobs", "run"])
        self.assertIn("--asr_language en", commands[2][-1])
        self.assertIn("--overwrite", commands[2][-1])
        self.assertIn("--sim_backend wavlm_seed_tts", commands[3][-1])
        self.assertEqual(commands[4][-2:], ["--exclude", "*.wav"])
        self.assertNotIn("--delete", commands[4])
        self.assertTrue(commands[5][1].endswith("score_results.py"))

    def test_remote_only_downloads_bucket_manifests_without_uploading_stale_local_state(self):
        args = self.args("--datasets", "seed_tts", "--only_langs", "en", "--results_bucket", "test/api-results",
                         "--stages", "transcribe", "score", "--dry_run")
        with patch.object(run_pipeline, "run_command") as runner:
            run_pipeline.run_pipeline(args)
        commands = [call.args[0] for call in runner.call_args_list]
        syncs = [command for command in commands if command[:3] == ["hf", "buckets", "sync"]]
        self.assertEqual(len(syncs), 2)
        self.assertTrue(all(command[3].startswith("hf://buckets/") for command in syncs))

    def test_full_split_ttfa_works_only_without_scorer_stages(self):
        args = self.args("--stages", "generate", "--ttfa_probe=-1")
        run_pipeline.validate_args(args)
        with self.assertRaisesRegex(ValueError, "sidecar only"):
            run_pipeline.validate_args(self.args("--ttfa_probe=-1"))

    def test_overwrite_forwarded_to_generation(self):
        args = self.args("--stages", "generate", "--overwrite")
        config = run_pipeline.dataset_configs(args)[0]
        self.assertIn("--overwrite", run_pipeline.generation_command(args, "fish/s2-pro", config))

    def test_local_fixture_not_relabelled_across_splits(self):
        args = self.args("--stages", "generate", "--input_jsonl", "samples.jsonl")
        with self.assertRaisesRegex(ValueError, "avoid relabelling"):
            run_pipeline.validate_args(args)

    def test_zero_workers_and_unknown_language_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_workers"):
            run_pipeline.validate_args(self.args("--stages", "generate", "--max_workers", "0"))
        with self.assertRaisesRegex(ValueError, "language codes"):
            run_pipeline.validate_args(self.args("--stages", "generate", "--only_langs", "xx"))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.calls = []

    def manifest(self, language="en", clone=False):
        path = self.directory / f"MODEL_fish-s2-pro_DATASET_test-eval_tts_{language}{'_voice_clone' if clone else ''}.jsonl"
        rows = [
            {"audio_filepath": f"tts_{language}/output_{index}.wav", "text": "some text", "pred_text": "some text",
             "duration": 1.2, "time": None, "timing_backend": "api", "language": language,
             "model_id": "fish/s2-pro", "api_latency_s": latency, "api_attempts": 1,
             "api_ttfa_ms": 100 + index * 100}
            for index, latency in enumerate((1.0, 3.0))
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        metadata = {"model_id": "fish/s2-pro", "provider": "fish", "model": "s2-pro", "voice": None,
                    "language": language, "dataset_path": "test/eval", "dataset": "tts", "split": language,
                    "voice_clone": clone, "n_samples": 2, "n_completed": 2, "n_failed": 0, "complete": True,
                    "max_workers": 1, "wall_time_s": 4.0, "api_throughput_rtfx": .6}
        path.with_suffix(".run.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path, rows

    def evaluate(self, directory, **kwargs):
        self.calls.append(kwargs)
        return {}, {"fish/s2-pro | test-eval": {"wer": 0.0, "metric": "WER", "sim": 95.0, "rtfx": None}}

    def test_language_scoped_quality_and_separate_api_metrics(self):
        en, _ = self.manifest("en")
        fr, _ = self.manifest("fr")
        rows = score_results.collect_results([en, fr], evaluator=self.evaluate)
        self.assertEqual({call["language"] for call in self.calls}, {"en", "fr"})
        self.assertEqual(rows[0]["api_latency_p50_s"], 2.0)
        self.assertAlmostEqual(rows[0]["api_latency_p95_s"], 2.9)
        self.assertEqual(rows[0]["api_ttfa_p50_ms"], 150)
        self.assertEqual(rows[0]["api_throughput_rtfx"], .6)
        self.assertIsNone(rows[0]["sim"])
        self.assertNotIn("rtfx", rows[0])

    def test_refuses_gpu_time_and_mismatched_language(self):
        path, rows = self.manifest()
        rows[0]["time"] = .5
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "GPU/API"):
            score_results.collect_results([path], evaluator=self.evaluate)
        rows[0]["time"] = None
        rows[0]["language"] = "fr"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "language/model"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_missing_metadata_is_rejected(self):
        path, _ = self.manifest()
        path.with_suffix(".run.json").unlink()
        with self.assertRaisesRegex(ValueError, "metadata"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_stale_sim_fork_is_rejected(self):
        path, rows = self.manifest(clone=True)
        rows[0]["api_latency_s"] = 999
        fork = path.with_name(path.stem + "_wavlm_seed_tts.jsonl")
        fork.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "stale"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_incomplete_counts_and_missing_asr_are_rejected(self):
        path, rows = self.manifest()
        sidecar = path.with_suffix(".run.json")
        metadata = json.loads(sidecar.read_text())
        metadata["complete"] = False
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "incomplete API"):
            score_results.collect_results([path], evaluator=self.evaluate)
        metadata["complete"] = True
        metadata["n_samples"] = 3
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "sample counts"):
            score_results.collect_results([path], evaluator=self.evaluate)
        metadata["n_samples"] = 2
        sidecar.write_text(json.dumps(metadata))
        del rows[0]["pred_text"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "ASR predictions"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_clone_similarity_required_and_known_short_exclusions_allowed(self):
        path, rows = self.manifest(clone=True)
        with self.assertRaisesRegex(ValueError, "incomplete.*similarity"):
            score_results.collect_results([path], evaluator=self.evaluate)
        model = "wavlm_large_finetune+ecapa_tdnn (wavlm_large_finetune.pth)"
        for row in rows:
            row["prompt_audio_filepath"] = "reference.wav"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        rows[0].update(sim=.95, sim_model=model)
        rows[1].update(sim=None, sim_model=model, sim_note="clip shorter than the 320-sample minimum")
        fork = path.with_name(path.stem + "_wavlm_seed_tts.jsonl")
        fork.write_text("".join(json.dumps(row) + "\n" for row in rows))
        exported = score_results.collect_results([path], evaluator=self.evaluate)
        self.assertEqual(exported[0]["sim"], 95.0)
        rows[0]["pred_text"] = "stale transcription"
        fork.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "stale"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_export_merges_evaluated_splits_without_placeholder_models(self):
        en, _ = self.manifest("en")
        fr, _ = self.manifest("fr")
        score_results.write_results(score_results.collect_results([en], evaluator=self.evaluate), self.directory)
        csv_path, json_path = score_results.write_results(score_results.collect_results([fr], evaluator=self.evaluate), self.directory)
        document = json.loads(json_path.read_text())
        self.assertEqual(len(document["results"]), 2)
        self.assertEqual({row["model_id"] for row in document["results"]}, {"fish/s2-pro"})
        self.assertNotIn(",rtfx,", csv_path.read_text().splitlines()[0])
        self.assertEqual(document["timing_backend"], "api")


if __name__ == "__main__":
    unittest.main()
