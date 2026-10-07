"""
TTS synthesis for the Open TTS Leaderboard (Voxtral TTS backend, stage 1).

Voxtral-4B-TTS-2603 (mistralai/Voxtral-4B-TTS-2603) is Mistral's ~4B open-weights TTS. It is
served as the model card prescribes, with `vllm serve <model> --omni` (started by
voxtral-tts/submit_jobs.sh in the same container), and this script is an HTTP client of the
OpenAI-compatible `/v1/audio/speech` endpoint. Each minibatch of requests is POSTed concurrently
(the server's continuous batching schedules them together) and the batch wall-clock time is used
for RTFx (per-sample time = batch time / batch size).

Fixed voice only (`--voice`, default `casual_male`): the open-source checkpoint lacks the audio
tokenizer's encoder weights, so cloning from `prompt_audio` is impossible and there is no SIM stage.
Sampling uses the server's stage defaults; the endpoint exposes no per-request sampling knobs.
"""

import argparse
import io
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)


def _report_param_count(model_id):
    """Print the model size in B parameters, summed from the `consolidated.safetensors` header
    (a cache hit: the server just loaded it); falls back to the model-card figure."""
    try:
        if Path(model_id).is_dir():
            path = str(Path(model_id) / "consolidated.safetensors")
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_id, "consolidated.safetensors")
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        n_params = 0
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            count = 1
            for dim in meta.get("shape", []):
                count *= dim
            n_params += count
        print(f"TTS model size: {n_params / 1e9:.2f}B parameters (from consolidated.safetensors header)")
    except Exception as e:
        print(
            "TTS model size: ~4B parameters per the model card "
            f"(Ministral-3-3B backbone + acoustic transformer + audio codec); header parse failed: {e}"
        )


def _server_alive(pid):
    """True if the `vllm serve` process is still running (pid=None → unknown, assume alive)."""
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_for_server(base_url, timeout_s, server_pid=None):
    """Block until the vLLM server's /health endpoint answers. If `server_pid` is given, a dead
    server process aborts the wait immediately instead of burning the whole timeout."""
    print(f"Waiting for vLLM server at {base_url} (timeout {timeout_s}s, pid={server_pid})...", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=5.0).status_code == 200:
                print("vLLM server is up.")
                return
        except httpx.HTTPError:
            pass
        if not _server_alive(server_pid):
            raise RuntimeError(
                f"`vllm serve` (pid {server_pid}) exited before the server became healthy — "
                "see the vllm serve log printed below."
            )
        time.sleep(5.0)
    raise RuntimeError(
        f"vLLM server at {base_url} did not become healthy within {timeout_s}s — "
        "see the vllm serve log printed below."
    )


def main(args):
    _wait_for_server(args.base_url, args.server_startup_timeout, args.server_pid)

    print(f"Serving Voxtral TTS model {args.model_id} via `vllm serve --omni` (voice={args.voice})")
    _report_param_count(args.model_id)

    # Layout: results/<model_safe>/  MODEL_<safe>_DATASET_<dsid>.jsonl  +  <dsid>/output_<id>.wav
    # where <dsid> = <dataset_safe>_<dataset>_<split>. Manifest paths are relative to model_dir so
    # later stages resolve them wherever the folder is copied (local or HF bucket).
    mode_suffix = ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name = paths.model_dir, paths.dataset_dir_name

    # One client with a connection pool sized for the concurrent minibatch.
    client = httpx.Client(
        base_url=args.base_url,
        timeout=httpx.Timeout(args.request_timeout, connect=30.0),
        limits=httpx.Limits(max_connections=args.batch_size, max_keepalive_connections=args.batch_size),
    )

    def build_payload(text):
        """One /v1/audio/speech request body (model-card format, preset voice)."""
        return {
            "input": text,
            "model": args.model_id,
            "response_format": "wav",
            "voice": args.voice,
        }

    def post_one(payload):
        r = client.post("/v1/audio/speech", json=payload)
        r.raise_for_status()
        return r.content  # encoded wav bytes

    # Canonical RIFF/WAVE header length, skipped when timestamping the first audio chunk so an
    # eagerly flushed header does not count as audio. Raw PCM has no header.
    WAV_HEADER_BYTES = 44
    VOXTRAL_SAMPLE_RATE = 24_000  # model card: 24 kHz mono

    # Streaming dialects to try on /v1/audio/speech, most capable first (as in higgs/run_eval.py).
    # vLLM-Omni streams only with stream=true AND response_format="pcm"; the last entry is the
    # buffered request, so the probe still yields a whole-utterance number otherwise.
    STREAM_MODES = ({"stream": True, "response_format": "pcm"}, {"stream": True}, {})
    stream_mode = [None]  # index into STREAM_MODES; pinned after the first request that works

    def post_one_streaming(payload, path, early_stop=False):
        """POST reading the body INCREMENTALLY: (bytes, first_audio_ts, n_chunks, content_type).

        Used only by the TTFA probe; the batched (RTFx) path keeps the buffered `post_one`. The
        dialect is negotiated once against STREAM_MODES and then pinned. (`/v1/audio/speech/stream`
        is a WebSocket endpoint, not a POST route.) `first_audio_ts` is None when the body arrived
        as a single chunk, so the probe falls back to the whole-utterance clock.
        """
        rejected = None
        for index in range(stream_mode[0] or 0, len(STREAM_MODES)):
            extra = STREAM_MODES[index]
            first, total, n_chunks, parts = None, 0, 0, []
            # Raw PCM carries no container header, so its very first byte is already audio.
            skip_bytes = 0 if extra.get("response_format") == "pcm" else WAV_HEADER_BYTES
            with client.stream("POST", path, json={**payload, **extra}) as r:
                # A 4xx while the dialect is still unknown means this server build does not
                # accept these fields: try the next one instead of failing the job.
                if r.status_code >= 400 and stream_mode[0] is None and index + 1 < len(STREAM_MODES):
                    rejected = r.read()[:300]
                    print(f"  [ttfa] server rejected {extra} ({r.status_code}): {rejected}\n"
                          f"        trying {STREAM_MODES[index + 1]}", flush=True)
                    continue
                r.raise_for_status()
                content_type = r.headers.get("content-type", "")
                if stream_mode[0] is None:
                    stream_mode[0] = index
                    print(f"  [ttfa] streaming route: {path} {extra or 'none (buffered)'} "
                          f"-> {content_type}", flush=True)
                for chunk in r.iter_bytes():
                    if not chunk:
                        continue
                    n_chunks += 1
                    total += len(chunk)
                    parts.append(chunk)
                    if first is None and total > skip_bytes:
                        first = time.perf_counter()
                        if early_stop:
                            # Leaving the `with` block closes the connection.
                            break
            if n_chunks <= 1 and not early_stop:
                first = None
            return b"".join(parts), first, n_chunks, content_type
        raise RuntimeError(f"No usable /v1/audio/speech streaming dialect (last rejection: {rejected})")

    def _audio_seconds(blob):
        """Duration of a response body; headerless PCM falls back to 24 kHz mono 16-bit."""
        try:
            info = sf.info(io.BytesIO(blob))
            return info.frames / info.samplerate
        except Exception:
            return len(blob) / float(2 * VOXTRAL_SAMPLE_RATE)

    def generate_tts(batch):
        """Synthesize speech for a minibatch of target texts; time the batch generation for RTFx."""
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)
        payloads = [build_payload(t) for t in texts_to_generate]

        # START TIMING (TTS batch generation). Wall clock: the GPU work happens in the server,
        # with the whole minibatch in flight concurrently.
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=minibatch_size) as pool:
            wav_blobs = list(pool.map(post_one, payloads))
        runtime = time.perf_counter() - start

        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for blob, sample_id in zip(wav_blobs, batch["id"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            path = os.path.join(model_dir, rel_path)
            with open(path, "wb") as f:
                f.write(blob)  # server returns a complete wav; store as-is
            info = sf.info(io.BytesIO(blob))
            gen_paths.append(rel_path)
            audio_length_s.append(info.frames / info.samplerate)

        batch["gen_audio_filepath"] = gen_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    # Keep only `id` and the target text, so nothing triggers audio decoding.
    dataset = load_tts_dataset(args)

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # TTFA is per-request, so the probe sends one streaming request at a time and returns before
    # the batched path; WER/RTFx runs never reach this branch.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        stream_chunks = []  # per-sample chunk counts, reported in the sidecar

        endpoints_used = []

        def _gen(job, early_stop):
            payload = build_payload(job[args.text_column][0])
            blob, first, n_chunks, ctype = post_one_streaming(
                payload, "/v1/audio/speech", early_stop)
            stream_chunks.append(n_chunks)
            endpoints_used.append(
                f"/v1/audio/speech {STREAM_MODES[stream_mode[0] or 0]} [{ctype}]")
            return _audio_seconds(blob), None, first

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            # No "device": this backend is an HTTP client; the GPU belongs to the server.
            extra={"note": "driven at batch size 1, one request in flight",
                   "voice": args.voice,
                   # Chunks per response; 1 everywhere means the server buffered.
                   "response_chunks": stream_chunks,
                   "endpoints": endpoints_used},
        )
        return

    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(generate_tts, dataset, starts, args.batch_size, args.warmup_steps)

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    for batch_start in tqdm(starts, desc="Generating"):
        batch = generate_tts(dataset[batch_start : batch_start + args.batch_size])

        # Append each sample to the JSONL immediately. `pred_text` is filled by stage 2 (ASR).
        for afp, alen, gtime, ref in zip(
            batch["gen_audio_filepath"], batch["audio_length_s"],
            batch["generation_time_s"], batch["references"],
        ):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref))
        manifest_file.flush()

    manifest_file.close()
    client.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Voxtral-4B-TTS-2603",
        help="Voxtral TTS model id, as served by `vllm serve <model_id> --omni`.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running `vllm serve --omni` server.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="casual_male",
        help="Preset voice (casual_male, neutral_female, ...; 20 available).",
    )
    add_common_args(parser, batch_size=32)
    parser.add_argument(
        "--server_startup_timeout",
        type=int,
        default=3600,
        help="Seconds to wait for the server's /health (first run includes the ~9 GB weight download).",
    )
    parser.add_argument(
        "--server_pid",
        type=int,
        default=None,
        help="PID of the backgrounded `vllm serve`; if it dies during startup the wait aborts at once.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=600.0,
        help="Per-request read timeout in seconds for /v1/audio/speech.",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
