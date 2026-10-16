"""Open TTS stage 1 / TTFA for fishaudio/drama-3 through pinned SGLang Omni.

Run from this directory; see README.md for the bounded, approval-gated job plan.
"""

import argparse
import array
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from adapter import (
    CODEC_ID,
    CODEC_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_HANDLING_POLICY,
    SERVER_REVISION,
    DramaClient,
    Sampling,
    wav_duration,
)
from input_dataset import ordered_text_sha256, resolve_dataset_source
from run_eval_utils import (
    add_common_args,
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    open_manifest,
    output_paths,
    print_next_steps,
    wav_rel_path,
    write_entry,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model_id", default=MODEL_ID, choices=[MODEL_ID])
    result.add_argument("--model_revision", default=MODEL_REVISION, choices=[MODEL_REVISION])
    result.add_argument("--server_revision", default=SERVER_REVISION, choices=[SERVER_REVISION])
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8000)
    result.add_argument("--server_timeout", type=int, default=1200)
    result.add_argument("--request_timeout", type=int, default=300)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--top_p", type=float, default=0.9)
    result.add_argument("--top_k", type=int, default=30)
    result.add_argument("--max_new_tokens", type=int, default=2048)
    result.add_argument("--dataset_local_root", help="Verified private Parquet mount; keeps dataset_path as benchmark ID")
    add_common_args(result, batch_size=1, voice_clone=False)
    return result


def write_prompt(path, audio):
    """Persist a mono PCM16 reference readable by both the local server and SIM."""
    values = audio["array"]
    rate = int(audio["sampling_rate"])
    if rate <= 0 or not len(values):
        raise ValueError("Reference audio must be nonempty with a positive sample rate")
    if len(values) / rate > 60:
        raise ValueError("Drama 3 reference audio exceeds the server's 60 second limit")
    pcm = array.array("h")
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Reference audio contains non-finite samples")
        pcm.append(round(max(-1.0, min(1.0, value)) * 32767))
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())


def prepare_reference(row, paths, voice_clone):
    if not voice_clone:
        return None, None
    relative = os.path.join(paths.dataset_dir_name, "prompts", f"prompt_{row['id']}.wav")
    path = Path(paths.model_dir, relative).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_prompt(path, row["prompt_audio"])
    return relative, [{"audio_path": str(path), "text": row["prompt_text"] or ""}]


def normalizer_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "normalizer").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    # HF jobs inject the backend without the Git checkout or normalizer files.
    # Preserve the launcher's source identity without claiming to run normalization.
    source_identity = os.environ.get("OPEN_TTS_SOURCE_IDENTITY")
    if not files and source_identity:
        identity = json.loads(source_identity)
        identity["note"] = "Source checkout provenance only; generation does not run normalization."
        return identity

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"checkout_head": git("rev-parse", "HEAD"),
            "committed_tree": git("rev-parse", "HEAD:normalizer"),
            "normalizer_file_sha256": (hashlib.sha256((root / "normalizer/normalizer.py").read_bytes()).hexdigest()
                                       if (root / "normalizer/normalizer.py").is_file() else None),
            "python_files_sha256": digest.hexdigest() if files else None,
            "note": "Generation does not use the normalizer; scoring revision must be recorded separately."}


def corpus_hash(dataset, args):
    digest = hashlib.sha256()
    for row in dataset:
        record = [str(row["id"]), row[args.text_column]]
        if args.voice_clone:
            record.append(row["prompt_text"])
            audio = row["prompt_audio"]
            record.append(int(audio["sampling_rate"]))
            # Hash the decoded reference without retaining private audio/text in metadata.
            samples = array.array("d", (float(x) for x in audio["array"]))
            if sys.byteorder != "little":
                samples.byteswap()
            record.append(hashlib.sha256(samples.tobytes()).hexdigest())
        digest.update(json.dumps(record, ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()


def write_metadata(path, payload):
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temp, path)


def print_generation_progress(completed, expected, capped):
    print(f"Generation progress: completed={completed}/{expected}; token_limit_reached={capped}", flush=True)


def main(args):
    if args.batch_size <= 0 or args.warmup_steps < 0 or args.server_timeout <= 0:
        raise ValueError("batch_size/server_timeout must be positive; warmup_steps must be nonnegative")
    sampling = Sampling(args.temperature, args.top_p, args.top_k, args.max_new_tokens)
    paths = output_paths(args, "_voice_clone" if args.voice_clone else "")
    loader_args, input_source = resolve_dataset_source(args)
    dataset = load_tts_dataset(loader_args, ("prompt_text", "prompt_audio") if args.voice_clone else ())
    if not len(dataset):
        raise ValueError("No evaluation samples")
    if input_source:
        prepared = input_source["prepared_split"]
        expected = min(prepared["rows"], args.max_eval_samples) if args.max_eval_samples and args.max_eval_samples > 0 else prepared["rows"]
        if len(dataset) != expected:
            raise ValueError("Loaded private input row count differs from preparation provenance")
        if expected == prepared["rows"] and ordered_text_sha256(dataset, args.text_column) != prepared["ordered_id_text_sha256"]:
            raise ValueError("Loaded private input ID/text order differs from preparation provenance")
    ids = [str(row["id"]) for row in dataset]
    if len(set(ids)) != len(ids) or any("/" in sid or "\\" in sid or sid in (".", "..") for sid in ids):
        raise ValueError("Sample IDs must be unique filenames")
    client = DramaClient(args.host, args.port, args.request_timeout, sampling)

    if args.ttfa_probe != 0:
        from ttfa_probe import probe_filename, run_probe, sample_indices
        suffix = "_voice_clone" if args.voice_clone else ""
        name = probe_filename(paths.model_safe, args.dataset_path.replace("/", "-"), args.dataset, args.split, suffix)
        if Path(paths.model_dir, name).exists():
            raise ValueError("TTFA sidecar already exists; use a separate results directory for a new probe")
        client.wait_for_server(args.server_timeout)
        # Stay beneath the server's allowed results root, but never overwrite stage-1 references.
        with tempfile.TemporaryDirectory(dir=paths.model_dir, prefix=".ttfa_refs_") as probe_dir:
            probe_paths = SimpleNamespace(model_dir=paths.model_dir, dataset_dir_name=Path(probe_dir).name)
            jobs = []
            for index in sample_indices(args.ttfa_probe, len(dataset)):
                row = dataset[index]
                _, reference = prepare_reference(row, probe_paths, args.voice_clone)
                jobs.append((row[args.text_column], reference))
            chunks = []
            outcomes = []

            def generate(job, early_stop):
                result = client.stream_result(*job, early_stop=early_stop)
                chunks.append(result.response_chunks)
                outcomes.append({"finish_reason": result.finish_reason,
                                 "generation_token_limit_reached": result.generation_token_limit_reached,
                                 "early_stopped": result.early_stopped})
                return result.duration, None, result.first_audio_timestamp

            payload = run_probe(
                generate, jobs, model_id=args.model_id, dataset_path=args.dataset_path,
                dataset_config=args.dataset, split=args.split, out_dir=paths.model_dir,
                mode_suffix=suffix, n=args.ttfa_probe,
                extra={"engine": "sglang", "model_revision": args.model_revision,
                       "server_revision": args.server_revision, "codec_revision": CODEC_REVISION,
                       "sampling": asdict(sampling), "request_seed_supported": False,
                       "output_handling_policy": OUTPUT_HANDLING_POLICY,
                       "dataset_source": input_source,
                       "response_chunks": chunks, "batch_size": 1},
            )
            for row, outcome in zip(payload["samples"], outcomes):
                row.update(outcome)
            payload["finish_reason_counts"] = {
                reason: sum(outcome["finish_reason"] == reason for outcome in outcomes)
                for reason in ("stop", "length")
            }
            payload["generation_token_limit_reached_samples"] = payload["finish_reason_counts"]["length"]
            payload["terminal_reason_unobserved_samples"] = sum(outcome["early_stopped"] for outcome in outcomes)
            write_metadata(str(Path(paths.model_dir, name)), payload)
        return

    metadata_path = paths.manifest_path.removesuffix(".jsonl") + ".run.json"
    settings = {
        "model_id": args.model_id, "model_revision": args.model_revision,
        "server_revision": args.server_revision, "codec_id": CODEC_ID, "codec_revision": CODEC_REVISION,
        "dataset_path": args.dataset_path, "dataset": args.dataset, "split": args.split,
        "text_column": args.text_column, "voice_clone": args.voice_clone,
        "batch_size": args.batch_size, "sampling": asdict(sampling),
        "output_handling_policy": OUTPUT_HANDLING_POLICY,
        "request_seed_supported": False, "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "input_sha256": corpus_hash(dataset, args), "expected_samples": len(dataset),
        "max_eval_samples": args.max_eval_samples,
        "dataset_source": input_source,
    }
    previous = None
    if os.path.exists(paths.manifest_path) or os.path.exists(metadata_path):
        if not args.resume:
            raise ValueError("Results already exist; use --resume with identical settings or a separate results directory")
        if not os.path.exists(metadata_path):
            raise ValueError("Cannot resume without the run metadata")
        with open(metadata_path, encoding="utf-8") as source:
            previous = json.load(source)
        if previous["settings"] != settings:
            raise ValueError("Resume settings or input corpus differ from the recorded run")
        if previous.get("completed_samples", 0) and not os.path.exists(paths.manifest_path):
            raise ValueError("Recorded completed samples are missing their manifest")
    done = load_done_entries(paths.manifest_path, args.resume)
    if os.path.exists(paths.manifest_path):
        with open(paths.manifest_path, encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if len(rows) != len(done):
            raise ValueError("Manifest contains duplicate or invalid rows")
    expected = {wav_rel_path(paths.dataset_dir_name, sid) for sid in ids}
    if not set(done) <= expected:
        raise ValueError("Manifest contains samples outside this corpus")
    finish_reason_counts = {"stop": 0, "length": 0}
    for relative, entry in done.items():
        reason = entry.get("finish_reason")
        if reason not in finish_reason_counts or entry.get("generation_token_limit_reached") is not (reason == "length"):
            raise ValueError("Completed sample is missing consistent generation termination metadata")
        finish_reason_counts[reason] += 1
        duration = wav_duration(Path(paths.model_dir, relative).read_bytes())
        if not math.isclose(duration, entry["duration"], abs_tol=1e-6):
            raise ValueError("Recorded audio duration does not match completed sample")
        if args.voice_clone and not Path(paths.model_dir, entry["prompt_audio_filepath"]).is_file():
            raise ValueError("Completed sample is missing its reference audio")
    pending = [dataset[i] for i, sid in enumerate(ids) if wav_rel_path(paths.dataset_dir_name, sid) not in done]
    metadata = previous or {"settings": settings, "normalizer_at_generation": normalizer_identity(),
                            "metrics_ready": False}
    metadata.update(completed_samples=len(done), status="running", finish_reason_counts=finish_reason_counts,
                    generation_token_limit_reached_samples=finish_reason_counts["length"])
    if previous and pending:
        metadata.setdefault("resume_normalizer_snapshots", []).append(normalizer_identity())
    write_metadata(metadata_path, metadata)
    if not pending:
        metadata.update(status="complete", completed_samples=len(dataset))
        write_metadata(metadata_path, metadata)
        print_generation_progress(len(dataset), len(dataset), finish_reason_counts["length"])
        print("All samples already complete; no synthesis requests sent.")
        return
    client.wait_for_server(args.server_timeout)

    def generate(row):
        return client.synthesize_result(row[args.text_column], row["_reference"])

    def prepare(batch):
        prepared = []
        for row in batch:
            prompt_path, reference = prepare_reference(row, paths, args.voice_clone)
            prepared.append({**row, "_prompt_path": prompt_path, "_reference": reference})
        return prepared

    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        for start in range(0, min(len(pending), args.warmup_steps * args.batch_size), args.batch_size):
            list(executor.map(generate, prepare(pending[start:start + args.batch_size])))
        with open_manifest(paths.manifest_path, done) as manifest:
            for start in range(0, len(pending), args.batch_size):
                batch = prepare(pending[start:start + args.batch_size])
                begin = time.perf_counter()
                outputs = list(executor.map(generate, batch))
                elapsed = (time.perf_counter() - begin) / len(batch)
                for row, result in zip(batch, outputs):
                    relative = wav_rel_path(paths.dataset_dir_name, row["id"])
                    Path(paths.model_dir, relative).write_bytes(result.audio)
                    entry = manifest_entry(
                        relative, wav_duration(result.audio), elapsed, row[args.text_column], row["_prompt_path"],
                    )
                    entry.update(finish_reason=result.finish_reason,
                                 generation_token_limit_reached=result.generation_token_limit_reached)
                    write_entry(manifest, entry)
                    finish_reason_counts[result.finish_reason] += 1
                manifest.flush()
                previous_completed = metadata["completed_samples"]
                metadata["completed_samples"] += len(batch)
                metadata["generation_token_limit_reached_samples"] = finish_reason_counts["length"]
                write_metadata(metadata_path, metadata)
                if (metadata["completed_samples"] < len(dataset)
                        and metadata["completed_samples"] // 50 > previous_completed // 50):
                    print_generation_progress(metadata["completed_samples"], len(dataset), finish_reason_counts["length"])
    metadata["status"] = "complete"
    write_metadata(metadata_path, metadata)
    print_generation_progress(len(dataset), len(dataset), finish_reason_counts["length"])
    print("Results saved at path:", os.path.abspath(paths.manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    main(parser().parse_args())
