"""Offline contracts: no model downloads, provider calls, credentials or GPU."""

import base64
import io
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import wave
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import adapter
import run_eval


def wav_bytes(frames=b"\0\0" * 80):
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(frames)
    return data.getvalue()


def event(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def delta(data):
    return event({"type": "speech.audio.delta", "audio": base64.b64encode(data).decode()})


class Response:
    status = 200

    def __init__(self, body=b"", blocks=None, headers=None):
        self.body = io.BytesIO(body)
        self.blocks = iter(blocks or [])
        self.headers = headers or {}
        self.closed = False

    def read(self, *args):
        return self.body.read(*args)

    def read1(self, size):
        return next(self.blocks, b"")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def stream_response(blocks, **headers):
    return Response(blocks=blocks, headers={
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Sample-Rate": "8000", "X-Channels": "1", "X-Bit-Depth": "16", **headers,
    })


class AdapterTests(unittest.TestCase):
    def test_released_identity_and_payload_preserve_verbatim_text(self):
        text = "Say (this) [and this], please."
        payload = adapter.speech_payload(text)
        self.assertEqual(payload, {
            "model": "fishaudio/drama-3", "input": text,
            "temperature": 1.0, "top_p": 0.9, "top_k": 30, "max_new_tokens": 2048,
            "stream": False, "response_format": "wav",
        })
        self.assertNotIn("seed", payload)
        self.assertNotIn("language", payload)

    def test_reference_and_supported_streaming_protocol(self):
        refs = [{"audio_path": "/safe/prompt.wav", "text": "Reference."}]
        payload = adapter.speech_payload("Target.", refs, streaming=True)
        self.assertEqual(payload["references"], refs)
        self.assertEqual(payload["response_format"], "pcm")
        self.assertEqual(payload["stream_format"], "sse")
        self.assertTrue(payload["stream"])

    def test_synthesis_retains_stop_and_length_audio_once_with_reason(self):
        client = adapter.DramaClient()
        for reason in ("stop", "length"):
            with self.subTest(reason=reason):
                client._open = Mock(return_value=Response(wav_bytes(), headers={"X-Finish-Reason": reason}))
                result = client.synthesize_result("Hello.")
                self.assertEqual(result.audio, wav_bytes())
                self.assertEqual(result.finish_reason, reason)
                self.assertIs(result.generation_token_limit_reached, reason == "length")
                self.assertEqual(client._open.call_count, 1)

    def test_bytes_api_remains_compatible_for_capped_audio(self):
        client = adapter.DramaClient()
        client._open = Mock(return_value=Response(wav_bytes(), headers={"X-Finish-Reason": "length"}))
        self.assertEqual(client.synthesize("Hello."), wav_bytes())
        client._open.assert_called_once()

    def test_buffered_missing_or_invalid_termination_and_invalid_wav_are_fatal_once(self):
        client = adapter.DramaClient()
        for reason in (None, "error", "cancelled", "unknown"):
            with self.subTest(reason=reason):
                client._open = Mock(return_value=Response(wav_bytes(), headers={"X-Finish-Reason": reason}))
                with self.assertRaisesRegex(RuntimeError, repr(reason)):
                    client.synthesize_result("Hello.")
                client._open.assert_called_once()
        for bad in (b"", b"RIFF", wav_bytes(b""), wav_bytes()[:-2]):
            for reason in ("stop", "length"):
                with self.subTest(bad=bad[:4], reason=reason):
                    client._open = Mock(return_value=Response(bad, headers={"X-Finish-Reason": reason}))
                    with self.assertRaises(ValueError):
                        client.synthesize_result("Hello.")
                    client._open.assert_called_once()

    def test_split_sse_and_pcm_frames_timestamp_first_playable_frame(self):
        client = adapter.DramaClient()
        blob = b":keepalive\r\n\r\n" + delta(b"") + delta(b"\x01") + delta(b"\x00")
        blob += event({"type": "speech.audio.done", "finish_reason": "stop"})
        response = stream_response([blob[i:i + 3] for i in range(0, len(blob), 3)])
        client._open = Mock(return_value=response)
        with patch.object(adapter.time, "perf_counter", return_value=123.0) as clock:
            duration, first, count = client.stream("Target.")
        self.assertEqual(duration, 1 / 8000)
        self.assertEqual(first, 123.0)
        self.assertEqual(count, 2)
        clock.assert_called_once()
        self.assertTrue(response.closed)

    def test_early_stop_closes_after_complete_audio_frame(self):
        client = adapter.DramaClient()
        response = stream_response([delta(b"\x01"), delta(b"\x00"), b"invalid later"])
        client._open = Mock(return_value=response)
        result = client.stream_result("Target.", early_stop=True)
        self.assertEqual(result.duration, 1 / 8000)
        self.assertIsNotNone(result.first_audio_timestamp)
        self.assertEqual(result.response_chunks, 2)
        self.assertIsNone(result.finish_reason)
        self.assertIsNone(result.generation_token_limit_reached)
        self.assertTrue(result.early_stopped)
        self.assertTrue(response.closed)
        client._open.assert_called_once()

    def test_complete_sse_retains_capped_audio_and_marks_terminal_reason(self):
        for reason in ("stop", "length"):
            with self.subTest(reason=reason):
                client = adapter.DramaClient()
                response = stream_response([
                    delta(b"\x01\x00"), delta(b"\x02\x00"),
                    event({"type": "speech.audio.done", "finish_reason": reason}),
                ])
                client._open = Mock(return_value=response)
                result = client.stream_result("Target.")
                self.assertEqual(result.duration, 2 / 8000)
                self.assertEqual(result.finish_reason, reason)
                self.assertIs(result.generation_token_limit_reached, reason == "length")
                self.assertFalse(result.early_stopped)
                self.assertEqual(result.response_chunks, 2)
                self.assertTrue(response.closed)
                client._open.assert_called_once()

    def test_stream_rejects_truncation_errors_and_invalid_audio(self):
        done = event({"type": "speech.audio.done", "finish_reason": "stop"})
        bad_streams = [
            [delta(b"\0\0")], [done], [delta(b"\0"), done],
            [delta(b"\0\0"), event({"type": "speech.audio.done"})],
            [delta(b"\0\0"), event({"type": "speech.audio.done", "finish_reason": "error"})],
            [delta(b"\0\0"), event({"type": "speech.audio.done", "finish_reason": "unknown"})],
            [event({"type": "speech.audio.done", "finish_reason": "length"})],
            [delta(b"\0"), event({"type": "speech.audio.done", "finish_reason": "length"})],
            [event({"type": "error", "error": {"message": "private text"}})],
            [event({"type": "speech.audio.delta", "audio": "!"}), done],
            [b"data: broken\n\n"], [delta(b"\0\0"), done[:-1]],
        ]
        client = adapter.DramaClient()
        for blocks in bad_streams:
            with self.subTest(blocks=blocks):
                response = stream_response(blocks)
                client._open = Mock(return_value=response)
                with self.assertRaises(RuntimeError) as raised:
                    client.stream("Target.")
                self.assertNotIn("private text", str(raised.exception))
                self.assertTrue(response.closed)
                self.assertEqual(client._open.call_count, 1)

    def test_no_buffered_or_wrong_format_fallback_requests(self):
        for headers in ({"Content-Type": "audio/wav"}, {"X-Bit-Depth": "32"}, {"X-Channels": "2"}):
            response = stream_response([], **headers)
            client = adapter.DramaClient()
            client._open = Mock(return_value=response)
            with self.assertRaises(RuntimeError):
                client.stream("Target.")
            self.assertEqual(client._open.call_count, 1)

    def test_wrong_server_model_fails_before_synthesis(self):
        for model in ("fishaudio/s2-pro", "fish/s2.1-pro", "fishaudio/drama-3-int4"):
            client = adapter.DramaClient()
            client._open = Mock(side_effect=[Response(), Response(json.dumps({"data": [{"id": model}]}).encode())])
            with self.assertRaisesRegex(RuntimeError, "identity"):
                client.wait_for_server()
            self.assertEqual([c.args[0] for c in client._open.call_args_list], ["/health", "/v1/models"])

    def test_http_error_is_redacted_without_retry(self):
        client = adapter.DramaClient()
        client.opener = Mock()
        client.opener.open.side_effect = urllib.error.HTTPError(
            "http://private", 500, "private text", {}, io.BytesIO(b"private body"),
        )
        with self.assertRaisesRegex(RuntimeError, "500") as raised:
            client.synthesize("private target")
        self.assertNotIn("private", str(raised.exception))
        client.opener.open.assert_called_once()


class Dataset(list):
    _fingerprint = "offline-corpus"


@contextmanager
def working_directory(path):
    before = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dataset = Dataset([
            {"id": i, "text": f"Target ({i}) [spoken].", "prompt_text": "Reference.",
             "prompt_audio": {"array": [0.0, 0.25, -0.25] * 20, "sampling_rate": 8000}}
            for i in range(3)
        ])
        self.args = run_eval.parser().parse_args(["--warmup_steps=0", "--batch_size=2", "--max_eval_samples=3"])
        self.client = Mock()
        self.client.synthesize_result.return_value = adapter.SynthesisResult(wav_bytes(), "stop")

    def evaluate(self, args=None):
        with working_directory(self.tmp.name), redirect_stdout(io.StringIO()), \
                patch.object(run_eval, "DramaClient", return_value=self.client), \
                patch.object(run_eval, "load_tts_dataset", return_value=self.dataset):
            run_eval.main(args or self.args)

    def files(self):
        return list(Path(self.tmp.name).rglob("MODEL_*.jsonl"))

    def test_standard_manifest_and_no_accidental_rerun(self):
        self.evaluate()
        manifest = self.files()[0]
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertIn("MODEL_fishaudio-drama-3_DATASET_", manifest.name)
        self.assertEqual([r["text"] for r in rows], [r["text"] for r in self.dataset])
        self.assertTrue(all(not Path(r["audio_filepath"]).is_absolute() for r in rows))
        self.assertTrue(all(r["time"] > 0 and r["duration"] == 0.01 for r in rows))
        self.assertTrue(all("prompt_audio_filepath" not in r for r in rows))
        self.assertTrue(all(r["finish_reason"] == "stop" and r["generation_token_limit_reached"] is False for r in rows))
        with self.assertRaisesRegex(ValueError, "Results already exist"):
            self.evaluate()
        self.assertEqual(self.client.synthesize_result.call_count, 3)
        self.client.reset_mock()
        self.args.resume = True
        self.evaluate()
        self.client.synthesize_result.assert_not_called()
        self.client.wait_for_server.assert_not_called()

    def test_capped_output_is_retained_without_cherry_picking_or_retry(self):
        # Exercise the real adapter-to-runner path, with one distinct response per row.
        self.args.batch_size = 1
        self.client = adapter.DramaClient()
        self.client.wait_for_server = Mock()
        audio = [wav_bytes(b"\x01\x00" * 80), wav_bytes(b"\x02\x00" * 80), wav_bytes(b"\x03\x00" * 80)]
        self.client._open = Mock(side_effect=[
            Response(data, headers={"X-Finish-Reason": reason})
            for data, reason in zip(audio, ("stop", "length", "stop"))
        ])
        self.evaluate()
        manifest = self.files()[0]
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual([row["text"] for row in rows], [row["text"] for row in self.dataset])
        self.assertEqual([row["finish_reason"] for row in rows], ["stop", "length", "stop"])
        self.assertEqual([row["generation_token_limit_reached"] for row in rows], [False, True, False])
        self.assertEqual([(manifest.parent / row["audio_filepath"]).read_bytes() for row in rows], audio)
        self.assertEqual(self.client._open.call_count, 3)
        self.assertEqual([call.args[1]["input"] for call in self.client._open.call_args_list],
                         [row["text"] for row in self.dataset])
        self.assertTrue(all(call.args[1]["max_new_tokens"] == 2048 for call in self.client._open.call_args_list))
        metadata = json.loads(manifest.with_suffix(".run.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["completed_samples"], 3)
        self.assertEqual(metadata["finish_reason_counts"], {"stop": 2, "length": 1})
        self.assertEqual(metadata["generation_token_limit_reached_samples"], 1)
        self.assertEqual(metadata["settings"]["output_handling_policy"], adapter.OUTPUT_HANDLING_POLICY)

    def test_resume_recounts_capped_rows_and_keeps_audio_and_order(self):
        self.args.batch_size = 1
        self.client.synthesize_result.side_effect = [
            adapter.SynthesisResult(wav_bytes(b"\x01\x00" * 80), "length"), RuntimeError("interrupted"),
        ]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.evaluate()
        manifest = self.files()[0]
        first = manifest.read_text()
        first_row = json.loads(first)
        first_audio = (manifest.parent / first_row["audio_filepath"]).read_bytes()
        metadata_path = manifest.with_suffix(".run.json")
        metadata = json.loads(metadata_path.read_text())
        # A crash can leave stale metadata after the completed manifest row was flushed.
        metadata["finish_reason_counts"] = {"stop": 999, "length": 0}
        metadata["generation_token_limit_reached_samples"] = 0
        metadata_path.write_text(json.dumps(metadata))
        self.args.resume = True
        self.client.reset_mock()
        self.client.synthesize_result.side_effect = [
            adapter.SynthesisResult(wav_bytes(), "stop"), adapter.SynthesisResult(wav_bytes(), "length"),
        ]
        self.evaluate()
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(manifest.read_text().splitlines()[0] + "\n", first)
        self.assertEqual((manifest.parent / first_row["audio_filepath"]).read_bytes(), first_audio)
        self.assertEqual([row["text"] for row in rows], [row["text"] for row in self.dataset])
        self.assertEqual(self.client.synthesize_result.call_count, 2)
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata["finish_reason_counts"], {"stop": 1, "length": 2})
        self.assertEqual(metadata["generation_token_limit_reached_samples"], 2)
        self.client.reset_mock()
        self.evaluate()
        self.client.synthesize_result.assert_not_called()
        self.client.wait_for_server.assert_not_called()
        self.assertEqual(json.loads(metadata_path.read_text())["generation_token_limit_reached_samples"], 2)

    def test_warmup_token_limits_are_not_counted_as_evaluation_rows(self):
        self.args.batch_size = 1
        self.args.warmup_steps = 1
        self.client.synthesize_result.side_effect = [adapter.SynthesisResult(wav_bytes(), reason)
                                                   for reason in ("length", "stop", "length", "stop")]
        self.evaluate()
        metadata = json.loads(self.files()[0].with_suffix(".run.json").read_text())
        self.assertEqual(self.client.synthesize_result.call_count, 4)
        self.assertEqual(metadata["completed_samples"], 3)
        self.assertEqual(metadata["finish_reason_counts"], {"stop": 2, "length": 1})
        self.assertEqual(metadata["generation_token_limit_reached_samples"], 1)

    def test_progress_reports_retained_row_and_cap_counts_without_sample_contents(self):
        self.args.batch_size = 1
        self.args.max_eval_samples = 51
        self.dataset = Dataset({**self.dataset[0], "id": index} for index in range(51))
        self.client.synthesize_result.side_effect = [adapter.SynthesisResult(wav_bytes(), reason)
                                                   for reason in ["length"] + ["stop"] * 50]
        with patch("builtins.print") as printed:
            self.evaluate()
        progress = [call for call in printed.call_args_list if str(call.args[0]).startswith("Generation progress:")]
        self.assertEqual([call.args[0] for call in progress], [
            "Generation progress: completed=50/51; token_limit_reached=1",
            "Generation progress: completed=51/51; token_limit_reached=1",
        ])
        self.assertTrue(all(call.kwargs == {"flush": True} for call in progress))

    def test_resume_rejects_old_output_policy_or_inconsistent_termination_fields(self):
        self.evaluate()
        manifest = self.files()[0]
        metadata_path = manifest.with_suffix(".run.json")
        metadata = json.loads(metadata_path.read_text())
        old = json.loads(metadata_path.read_text())
        del old["settings"]["output_handling_policy"]
        metadata_path.write_text(json.dumps(old))
        self.args.resume = True
        self.client.reset_mock()
        with self.assertRaisesRegex(ValueError, "settings"):
            self.evaluate()
        metadata_path.write_text(json.dumps(metadata))
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        rows[0]["generation_token_limit_reached"] = True
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with self.assertRaisesRegex(ValueError, "termination metadata"):
            self.evaluate()
        self.client.synthesize_result.assert_not_called()
        self.client.wait_for_server.assert_not_called()

    def test_partial_batch_resume_keeps_completed_rows_once(self):
        self.evaluate()
        manifest = self.files()[0]
        first = manifest.read_text().splitlines()[0]
        manifest.write_text(first + "\n")
        self.client.reset_mock()
        self.args.resume = True
        self.evaluate()
        rows = manifest.read_text().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], first)
        self.assertEqual(self.client.synthesize_result.call_count, 2)

    def test_resume_refuses_different_corpus_settings_or_missing_audio(self):
        self.evaluate()
        self.args.resume = True
        self.args.temperature = 0.8
        with self.assertRaisesRegex(ValueError, "settings"):
            self.evaluate()
        self.args.temperature = 1.0
        self.dataset[0]["text"] = "Changed."
        with self.assertRaisesRegex(ValueError, "settings"):
            self.evaluate()
        self.dataset[0]["text"] = "Target (0) [spoken]."
        manifest = self.files()[0]
        row = json.loads(manifest.read_text().splitlines()[0])
        (manifest.parent / row["audio_filepath"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.evaluate()

    def test_cloning_persists_reference_and_relative_sim_path(self):
        self.args.voice_clone = True
        self.evaluate()
        manifest = self.files()[0]
        self.assertTrue(manifest.name.endswith("_voice_clone.jsonl"))
        row = json.loads(manifest.read_text().splitlines()[0])
        self.assertEqual(row["sim"], "")
        self.assertFalse(Path(row["prompt_audio_filepath"]).is_absolute())
        self.assertTrue((manifest.parent / row["prompt_audio_filepath"]).is_file())
        reference = self.client.synthesize_result.call_args_list[0].args[1][0]
        self.assertTrue(Path(reference["audio_path"]).is_absolute())
        self.assertEqual(reference["text"], "Reference.")

    def test_failure_does_not_write_placeholder_or_completed_manifest_row(self):
        self.args.batch_size = 1
        self.client.synthesize_result.side_effect = [adapter.SynthesisResult(wav_bytes(), "stop"), RuntimeError("failed")]
        with self.assertRaises(RuntimeError):
            self.evaluate()
        rows = self.files()[0].read_text().splitlines()
        self.assertEqual(len(rows), 1)
        metadata = json.loads(next(Path(self.tmp.name).rglob("*.run.json")).read_text())
        self.assertEqual(metadata["completed_samples"], 1)
        self.assertEqual(metadata["status"], "running")
        self.assertFalse(metadata["metrics_ready"])

    def test_invalid_reference_fails_before_synthesis(self):
        self.args.voice_clone = True
        self.dataset[0]["prompt_audio"]["array"] = [float("nan")]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.evaluate()
        self.client.synthesize_result.assert_not_called()

    def test_wrong_older_model_is_rejected_by_cli(self):
        with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
            run_eval.parser().parse_args(["--model_id=fishaudio/s2-pro"])

    def test_cloning_probe_preserves_existing_references_and_cleans_temporary_wavs(self):
        self.args.voice_clone = True
        self.evaluate()
        manifest = self.files()[0]
        rows_before = manifest.read_bytes()
        row = json.loads(manifest.read_text().splitlines()[0])
        prompt = manifest.parent / row["prompt_audio_filepath"]
        prompt_before = prompt.read_bytes()
        self.dataset[0]["prompt_audio"]["array"] = [0.9] * 50
        self.args.ttfa_probe = 2
        self.client.stream_result.side_effect = lambda *args, **kwargs: adapter.StreamResult(
            0.01, time.perf_counter(), 2, None if kwargs["early_stop"] else "stop", kwargs["early_stop"],
        )
        self.evaluate()
        self.assertEqual(manifest.read_bytes(), rows_before)
        self.assertEqual(prompt.read_bytes(), prompt_before)
        self.assertEqual(list(manifest.parent.glob(".ttfa_refs_*")), [])
        self.assertEqual(len(list(manifest.parent.glob("TTFA_*.json"))), 1)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.evaluate()

    def test_probe_records_stop_length_and_unobserved_early_stop_per_sample(self):
        self.args.ttfa_probe = -1
        reasons = iter(("stop", "length", None))

        def stream(*args, early_stop):
            reason = next(reasons)
            self.assertEqual(early_stop, reason is None)
            return adapter.StreamResult(0.01, time.perf_counter(), 2, reason, early_stop)

        self.client.stream_result.side_effect = stream
        # The real shared probe controls the early-stop tail; make its boundary small here.
        import ttfa_probe
        with patch.object(ttfa_probe, "FULL_GEN_SAMPLES", 2):
            self.evaluate()
        sidecar = json.loads(next(Path(self.tmp.name).rglob("TTFA_*.json")).read_text())
        rows = sidecar["samples"]
        self.assertEqual([row["finish_reason"] for row in rows], ["stop", "length", None])
        self.assertEqual([row["generation_token_limit_reached"] for row in rows], [False, True, None])
        self.assertEqual([row["early_stopped"] for row in rows], [False, False, True])
        self.assertEqual([row["full_generation"] for row in rows], [True, True, False])
        self.assertIsNone(rows[-1]["ttfa_frac_of_gen"])
        self.assertEqual(sidecar["finish_reason_counts"], {"stop": 1, "length": 1})
        self.assertEqual(sidecar["generation_token_limit_reached_samples"], 1)
        self.assertEqual(sidecar["terminal_reason_unobserved_samples"], 1)
        self.assertEqual(sidecar["output_handling_policy"], adapter.OUTPUT_HANDLING_POLICY)
        self.assertEqual(self.client.stream_result.call_count, 3)
        self.assertEqual(self.files(), [])

    def test_completed_resume_preserves_original_generation_provenance(self):
        with patch.object(run_eval, "normalizer_identity", return_value={"original": "revision"}):
            self.evaluate()
        self.args.resume = True
        with patch.object(run_eval, "normalizer_identity", return_value={"later": "revision"}):
            self.evaluate()
        metadata = json.loads(next(Path(self.tmp.name).rglob("*.run.json")).read_text())
        self.assertEqual(metadata["normalizer_at_generation"], {"original": "revision"})

    def test_private_source_provenance_keeps_canonical_result_name(self):
        source = {"prepared_split": {"rows": 1088}, "provenance_sha256": "a" * 64}
        loader_args = run_eval.SimpleNamespace(**vars(self.args))
        loader_args.dataset_path = "/inputs/seed_tts_eval"
        with patch.object(run_eval, "resolve_dataset_source", return_value=(loader_args, source)):
            self.evaluate()
        manifest = self.files()[0]
        self.assertIn("bezzam-seed_tts_eval_tts_en", manifest.name)
        self.assertNotIn("inputs", manifest.name)
        metadata = json.loads(manifest.with_suffix(".run.json").read_text())
        self.assertEqual(metadata["settings"]["dataset_source"], source)

    def test_bad_private_source_row_count_stops_before_server_requests(self):
        source = {"prepared_split": {"rows": 2}, "provenance_sha256": "a" * 64}
        with patch.object(run_eval, "resolve_dataset_source", return_value=(self.args, source)), \
                self.assertRaisesRegex(ValueError, "row count"):
            self.evaluate()
        self.client.wait_for_server.assert_not_called()
        self.client.synthesize_result.assert_not_called()

    def test_injected_job_keeps_source_checkout_provenance(self):
        identity = {"checkout_head": "source-revision", "normalizer_file_sha256": "audited-hash"}
        path = str(Path(self.tmp.name, "backend", "run_eval.py"))
        with patch.object(run_eval, "__file__", path), \
                patch.dict(os.environ, {"OPEN_TTS_SOURCE_IDENTITY": json.dumps(identity)}):
            actual = run_eval.normalizer_identity()
        self.assertEqual(actual["checkout_head"], "source-revision")
        self.assertEqual(actual["normalizer_file_sha256"], "audited-hash")


if __name__ == "__main__":
    unittest.main()
