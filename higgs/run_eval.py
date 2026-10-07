"""
TTS synthesis for the Open TTS Leaderboard (Higgs TTS 3 backend, stage 1).

Higgs TTS 3 is served by an SGLang server (`sgl-omni serve`, launched by higgs/submit_jobs.sh)
exposing an OpenAI-compatible `/v1/audio/speech` endpoint, so this script is an HTTP client: it
waits for the server, sends the target texts as synthesis requests, times the batched generation
for RTFx, saves the wavs, and writes a manifest with `text`, `duration`, `time` and an empty
`pred_text` (filled by stage 2 ASR; stage 3 scores SIM for voice-clone runs).

Nothing here is Higgs-specific (`/v1/audio/speech` is sgl-omni's model-agnostic route), so
submit_ttfa_jobs.sh also drives it against `fishaudio/s2-pro` (target `s2pro-sglang`). Keep
model-specific values in the CLI defaults / the target's flags, not in the request-building code.

Voice cloning (`--voice_clone`) conditions each request on the per-sample
`prompt_audio`/`prompt_text`; outputs get a `_voice_clone` suffix. The default (`--no-voice_clone`)
uses the model's default voice and skips SIM.
"""

import argparse
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest,
    output_paths, pending_batches, print_next_steps, warm_up, wav_rel_path, write_entry,
)


def print_model_size(model_id):
    """Best-effort parameter count from the safetensors headers in the local HF cache.

    The SGLang server has already downloaded the model; only shard headers are read.
    """
    try:
        import struct

        from huggingface_hub import hf_hub_download

        def header_numels(path):
            with open(path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            total = 0
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                numel = 1
                for d in meta.get("shape", []):
                    numel *= d
                total += numel
            return total

        try:
            # Sharded checkpoint: read the index, then each referenced shard's header.
            idx = hf_hub_download(model_id, "model.safetensors.index.json", local_files_only=True)
            with open(idx) as f:
                weight_map = json.load(f)["weight_map"]
            total = sum(header_numels(hf_hub_download(model_id, shard, local_files_only=True))
                        for shard in sorted(set(weight_map.values())))
        except Exception:
            # Single-file checkpoint.
            total = header_numels(hf_hub_download(model_id, "model.safetensors", local_files_only=True))

        print(f"TTS model size: {total / 1e9:.2f}B parameters")
    except Exception as e:
        print(f"Could not determine model size from cache: {e}")


def wait_for_server(base_url, timeout_s, heartbeat_s=60):
    """Poll the SGLang server's /health endpoint until it is ready (or time out).

    Prints a heartbeat while waiting: the server logs to its own file, so otherwise the job
    prints nothing until ready (weight download can take minutes).
    """
    started = time.time()
    deadline = started + timeout_s
    last_err = None
    next_beat = started + heartbeat_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print(f"SGLang server ready at {base_url} after {time.time() - started:.0f}s")
                return
        except requests.RequestException as e:
            last_err = e
        now = time.time()
        if now >= next_beat:
            print(f"  waiting for {base_url} ... {now - started:.0f}s elapsed "
                  f"(timeout {timeout_s}s; server output goes to its own log, dumped on failure)",
                  flush=True)
            next_beat = now + heartbeat_s
        time.sleep(3)
    raise RuntimeError(f"SGLang server at {base_url} not ready after {timeout_s}s (last error: {last_err})")


def main(args):
    base_url = f"http://{args.host}:{args.port}"
    wait_for_server(base_url, args.server_timeout)
    # Server is up => the model is in the HF cache; report its parameter count.
    print_model_size(args.model_id)

    def speech_payload(job):
        """Request body for one synthesis, shared by the batched path and the TTFA probe.

        `job` is a (text, reference) tuple, where reference is the per-sample voice-clone
        payload (a one-element list of {audio_path, text}) or None for the default voice.

        --top_p is omitted unless set, so the server keeps the served model's own default.
        """
        text, reference = job
        payload = {
            "input": text,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            # Fixed seed for reproducibility (harmless if the endpoint ignores it).
            "seed": args.seed,
        }
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        if reference is not None:
            payload["references"] = reference
        return payload

    def synthesize(job):
        """Send one synthesis request; return the raw WAV bytes."""
        resp = requests.post(f"{base_url}/v1/audio/speech", json=speech_payload(job),
                             timeout=args.request_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Synthesis request failed ({resp.status_code}): {resp.text[:500]}")
        return resp.content

    # The streaming dialect is server-version dependent, so it is negotiated on the first probe
    # request and then reused:
    #   {"stream": True, "response_format": "pcm"}  pinned sgl-omni: raw `audio/pcm` body, format
    #       in X-Sample-Rate / X-Channels / X-Bit-Depth (stream without pcm is rejected with 400).
    #   {"stream": True}  the model card's SSE example: base64 audio chunks in `data:` events.
    #   {}  no streaming — buffered whole utterance, first_ts None (whole-utterance clock).
    STREAM_MODES = ({"stream": True, "response_format": "pcm"}, {"stream": True}, {})
    stream_mode = [None]  # index into STREAM_MODES; pinned after the first request that works

    # Bytes per sample per channel, by soundfile subtype. Needed to length an SSE chunk that
    # carries no RIFF header of its own (see _read_sse).
    SUBTYPE_BYTES = {"PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8}

    def _read_pcm(resp, early_stop):
        """Raw PCM stream -> (audio_seconds, first_ts, n_chunks).

        The body is headerless samples, so the first byte received is a real first-audio time.
        """
        rate = int(resp.headers["X-Sample-Rate"])
        frame_bytes = (int(resp.headers.get("X-Channels", 1))
                       * int(resp.headers.get("X-Bit-Depth", 16)) // 8)
        first, total, n_chunks = None, 0, 0
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            n_chunks += 1
            total += len(chunk)
            if first is None:
                first = time.perf_counter()
                if early_stop:
                    # Closing the response stops the server sending; TTFA is already known.
                    break
        return total / frame_bytes / rate, first, n_chunks

    def _read_sse(resp, early_stop):
        """Server-Sent Events stream -> (audio_seconds, first_ts, n_chunks).

        Each `data: {...}` line carries base64 audio in `audio.data`; the terminal one has
        `finish_reason: "stop"`. Duration is summed per chunk, which is correct whether chunks are
        standalone WAVs or headerless PCM after a first WAV.
        """
        first, n_chunks, frames, rate, frame_bytes = None, 0, 0, None, None
        # chunk_size=None so each line surfaces as soon as its bytes arrive; the default
        # 512-byte read would add a buffer wait to the very measurement being taken.
        for line in resp.iter_lines(chunk_size=None):
            if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("finish_reason") == "stop":
                break
            data = (event.get("audio") or {}).get("data")
            if not data:
                continue
            chunk = base64.b64decode(data)
            n_chunks += 1
            if chunk[:4] == b"RIFF":
                info = sf.info(io.BytesIO(chunk))
                rate = info.samplerate
                frame_bytes = info.channels * SUBTYPE_BYTES.get(info.subtype, 2)
                frames += info.frames
            elif frame_bytes:
                frames += len(chunk) // frame_bytes
            else:
                raise RuntimeError(
                    "First streamed chunk carries no RIFF header, so its sample rate is unknown "
                    f"and the audio duration cannot be measured (first bytes: {chunk[:8]!r})"
                )
            # Timestamp the first chunk holding actual frames, not a bare WAV header.
            if first is None and frames > 0:
                first = time.perf_counter()
                if early_stop:
                    break
        if not rate:
            raise RuntimeError("Event stream carried no audio chunks")
        return frames / rate, first, n_chunks

    def synthesize_streaming(job, early_stop=False):
        """Like synthesize(), but incrementally: (audio_seconds, first_ts, n_chunks).

        Used only by the TTFA probe; the batched (RTFx) path keeps the buffered request.
        `first_ts` is None when the server did not stream, so the probe falls back to the
        whole-utterance clock.
        """
        payload = speech_payload(job)
        rejected = None
        for index in range(stream_mode[0] or 0, len(STREAM_MODES)):
            extra = STREAM_MODES[index]
            with requests.post(f"{base_url}/v1/audio/speech", json={**payload, **extra},
                               timeout=args.request_timeout, stream=True) as resp:
                # A 400 while the route is still unknown means this server does not accept this
                # streaming dialect: try the next one instead of failing the job.
                if resp.status_code == 400 and stream_mode[0] is None and index + 1 < len(STREAM_MODES):
                    rejected = resp.text[:300]
                    print(f"  [ttfa] server rejected {extra}: {rejected}\n"
                          f"        trying {STREAM_MODES[index + 1]}", flush=True)
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"Synthesis request failed ({resp.status_code}): {resp.text[:500]}")
                if stream_mode[0] is None:
                    stream_mode[0] = index
                    print(f"  [ttfa] streaming route: {extra or 'none (buffered)'} "
                          f"-> {resp.headers.get('content-type')}", flush=True)
                content_type = resp.headers.get("content-type", "")
                if "event-stream" in content_type:
                    return _read_sse(resp, early_stop)
                if "pcm" in content_type or "X-Sample-Rate" in resp.headers:
                    return _read_pcm(resp, early_stop)
                # Buffered mode, or the server ignored `stream`: no first-chunk time exists.
                info = sf.info(io.BytesIO(resp.content))
                return info.frames / info.samplerate, None, 1
        raise RuntimeError(f"No usable /v1/audio/speech streaming route (last rejection: {rejected})")

    # Layout: results/<model_safe>/  MODEL_<safe>_DATASET_<dsid>.jsonl  +  <dsid>/output_<id>.wav
    # where <dsid> = <dataset_safe>_<dataset>_<split><mode_suffix>. Manifest paths are relative to
    # model_dir so later stages resolve them wherever the folder is copied (local or HF bucket).
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    # Reference clips must live under results/ (not tempfiles) so stage 3 SIM can read them.
    prompt_dir = os.path.join(output_dir, "prompts")
    if args.voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    # Reuse one thread pool across batches; concurrency = batch_size.
    executor = ThreadPoolExecutor(max_workers=args.batch_size)

    def generate_tts(texts, ids, references):
        """Synthesize a batch of texts concurrently; time the wall-clock batch for RTFx."""
        minibatch_size = len(texts)

        # START TIMING (concurrent batch generation on the server)
        start = time.perf_counter()
        wav_bytes_list = list(executor.map(synthesize, zip(texts, references)))
        runtime = time.perf_counter() - start
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        per_sample_time = runtime / minibatch_size

        gen_paths, audio_length_s = [], []
        for wav_bytes, sample_id in zip(wav_bytes_list, ids):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            # Write the returned WAV bytes verbatim, then read back for duration/sample rate.
            with open(os.path.join(model_dir, rel_path), "wb") as f:
                f.write(wav_bytes)
            data, sr = sf.read(io.BytesIO(wav_bytes))
            gen_paths.append(rel_path)
            audio_length_s.append(len(data) / sr)

        return gen_paths, audio_length_s, minibatch_size * [per_sample_time]

    # Keep `id`, target text, and (when cloning) the per-sample reference.
    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio") if args.voice_clone else ())

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # TTFA is per-request, so the probe sends one streaming request at a time into a temp dir and
    # returns before the batched path; WER/RTFx runs never reach this branch.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        stream_chunks = []  # per-sample chunk counts, reported in the sidecar

        def _gen(job, early_stop):
            reference = None
            if args.voice_clone:
                ref_path = os.path.join(output_dir, "prompts", f"prompt_{job['id'][0]}.wav")
                pa = job["prompt_audio"][0]
                sf.write(ref_path, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
                reference = [{"audio_path": ref_path, "text": job["prompt_text"][0]}]
            audio_s, first, n_chunks = synthesize_streaming(
                (job[args.text_column][0], reference), early_stop)
            stream_chunks.append(n_chunks)
            return audio_s, None, first

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            # No "device": this backend is an HTTP client of sglang and has no --device argument.
            extra={"note": "driven at batch size 1, one request in flight",
                   # Audio chunks per response; 1 everywhere means the server buffered.
                   "response_chunks": stream_chunks},
        )
        return

    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def prepare_references(batch):
        """When cloning, persist each reference clip and build the per-request payload (the SGLang
        server + stage-3 SIM both read `prompt_audio_filepath`); (prompt_paths, references)."""
        ids = batch["id"]
        prompt_paths = [None] * len(ids)
        references = [None] * len(ids)
        if args.voice_clone:
            for j, sid in enumerate(ids):
                # Store the path relative to model_dir (the manifest's dir); write to the full path.
                prel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sid}.wav")
                pp = os.path.join(model_dir, prel)
                if not os.path.exists(pp):
                    pa = batch["prompt_audio"][j]
                    sf.write(pp, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
                prompt_paths[j] = prel
                # The SGLang server was launched from a different cwd than this client, so
                # give it an absolute path (the manifest keeps the relative path for stage 3).
                references[j] = [{"audio_path": os.path.abspath(pp), "text": batch["prompt_text"][j] or ""}]
        return prompt_paths, references

    def run_batch(batch):
        prompt_paths, references = prepare_references(batch)
        return prompt_paths, generate_tts(list(batch[args.text_column]), batch["id"], references)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    for batch_start in tqdm(starts, desc="Generating"):
        batch = dataset[batch_start : batch_start + args.batch_size]
        texts = list(batch[args.text_column])
        prompt_paths, (gen_paths, audio_length_s, gen_times) = run_batch(batch)

        # Append each sample to the JSONL immediately.
        for afp, ppath, alen, gtime, ref in zip(gen_paths, prompt_paths, audio_length_s, gen_times, texts):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref, ppath if args.voice_clone else None))
        manifest_file.flush()

    manifest_file.close()
    executor.shutdown()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="bosonai/higgs-tts-3-4b",
        help="Higgs TTS model id served by the SGLang server (for output-path naming).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="SGLang server host.")
    parser.add_argument("--port", type=int, default=8000, help="SGLang server port.")
    parser.add_argument(
        "--server_timeout", type=int, default=1200, help="Seconds to wait for the SGLang server to become ready."
    )
    parser.add_argument("--request_timeout", type=int, default=300, help="Per-request timeout (seconds).")
    add_common_args(parser, batch_size=16, voice_clone=False)
    # Generation params: the model card's /v1/audio/speech example values.
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max audio tokens per synthesis request (model-card default 1024 = ~40 s at the 25 Hz "
             "codec rate).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (model-card default 0.8). Do NOT set 0.0: greedy decoding makes "
             "this model fail to emit EOS on a large fraction of prompts, which loops to "
             "--max_new_tokens instead of producing a usable clip.",
    )
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (Higgs model-card default 50). S2-Pro accepts only -1 "
                             "or 1..30, so that target passes --top_k=30.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p sampling. Unset = leave it to the server's default for the "
                             "served model (see speech_payload).")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} (SGLang @ {args.host}:{args.port}) on "
          f"{args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
