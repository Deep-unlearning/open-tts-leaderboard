"""
TTS synthesis for the Open TTS Leaderboard (NeuTTS Nano backend, stage 1).

NeuTTS Nano (neuphonic/neutts-nano) is Neuphonic's compact zero-shot voice-cloning TTS: a small
Qwen-style LM backbone emitting NeuCodec speech tokens, decoded to 24 kHz audio. Text is phonemized
with espeak-ng and outputs are watermarked by default with resemble-perth. A `*-gguf` model id
loads the backbone through llama_cpp instead (the only path that supports streaming).

NeuTTS always synthesizes from a reference, so every sample uses the dataset's
`prompt_audio`/`prompt_text`. `--voice_clone` adds the `_voice_clone` suffix and the SIM fields;
the default (`--no-voice_clone`) only skips SIM.

No native batching: `NeuTTS.infer()` takes one text, so `generate_tts(batch)` loops per sample and
`--batch_size` only sets manifest-writing / resume granularity; the per-sample time is batch time /
batch size. Reference encoding is inside the timed region (part of zero-shot inference); writing
the prompt wavs to disk is not.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# CPU-built llama_cpp, installed alongside the CUDA one (see the Dockerfile).
LLAMA_CPP_CPU_PATH = os.environ.get("LLAMA_CPP_CPU_PATH", "/opt/llama-cpp-cpu")


# Keep dependency chatter (transformers, phonemizer, perth) out of the way so the outer
# "Generating" tqdm is the only progress output; warnings/errors still come through stderr.
warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
except Exception:
    pass

from neutts import NeuTTS  # noqa: E402

from run_eval_utils import (  # noqa: E402
    add_common_args, count_parameters, load_done_entries, load_tts_dataset, manifest_entry,
    open_manifest, output_paths, pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path,
    write_entry,
)

torch.set_float32_matmul_precision("high")


def main(args):
    # Set seed for reproducibility (NeuTTS samples internally: do_sample=True, top_k=50).
    seed = args.seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True

    # llama_cpp ships as two mutually incompatible builds (see the Dockerfile): the CUDA one is
    # the default, and its libllama.so links libcuda.so.1 — absent on a CPU-only machine, where
    # importing it raises "Failed to load shared library". Put the CPU build first on sys.path
    # for a CPU run, BEFORE NeuTTS() triggers the import.
    if str(args.device).lower() == "cpu" and os.path.isdir(LLAMA_CPP_CPU_PATH):
        sys.path.insert(0, LLAMA_CPP_CPU_PATH)
        print(f"CPU run: using the CPU-built llama_cpp from {LLAMA_CPP_CPU_PATH}")

    # Torch (non-GGUF) path: the backbone is loaded with AutoModelForCausalLM and moved to
    # `backbone_device`; the NeuCodec codec goes to `codec_device`. `language` resolves to
    # "en-us" automatically for neuphonic/neutts-nano (BACKBONE_LANGUAGE_MAP).
    model = NeuTTS(
        backbone_repo=args.model_id,
        backbone_device=args.device,
        codec_repo=args.codec_repo,
        codec_device=args.device,
        language=args.language,
    )
    sampling_rate = int(getattr(model, "sample_rate", 24_000))  # NeuCodec: 24 kHz

    # A *-gguf model_id makes the package load the backbone through llama_cpp instead of
    # transformers, which is what enables infer_stream(). That backbone is not an nn.Module.
    is_gguf = bool(getattr(model, "_is_quantized_model", False))

    # The `neutts` package loads the backbone in float32 (no dtype knob); optionally cast
    # the backbone LM (the codec stays in float32). A gguf backbone is already quantized.
    if args.dtype != "float32":
        if is_gguf:
            print(f"Ignoring --dtype={args.dtype}: the gguf backbone carries its own quantization.")
        else:
            model.backbone.to(getattr(torch, args.dtype))

    if args.no_watermark:
        # infer() skips resemble-perth watermarking when the watermarker is None.
        model.watermarker = None

    print(
        f"Loaded NeuTTS model {args.model_id} (codec={args.codec_repo}, sr={sampling_rate}, "
        f"dtype={args.dtype}, watermark={not args.no_watermark})"
    )

    # Report parameter count. NeuTTS is a wrapper around nn.Module components (backbone LM
    # + NeuCodec), so sum unique params over any nn.Module attributes (dedup by id).
    try:
        n_params = count_parameters(model)
        if n_params:
            print(f"TTS model size: {n_params / 1e9:.2f}B parameters (backbone + codec)")
    except Exception as e:
        print(f"Could not determine model size: {e}")

    # Layout: results/<model_safe>/  MODEL_<safe>_DATASET_<dsid>.jsonl  +  <dsid>/output_<id>.wav
    # where <dsid> = <dataset_safe>_<dataset>_<split><mode_suffix>. Manifest paths are relative to
    # model_dir so later stages resolve them wherever the folder is copied (local or HF bucket).
    # Keep the two modes' audio separate so voice-clone and non-cloning runs don't overwrite.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir

    # `encode_reference` takes a file path; prompts are written under `results/` so stage 3 can
    # read them. Created regardless of --voice_clone since NeuTTS always needs a reference.
    prompt_dir = os.path.join(output_dir, "prompts")
    os.makedirs(prompt_dir, exist_ok=True)

    n_watermark_skipped = [0]  # near-empty takes that skipped watermarking
    n_failed = [0]             # samples that produced nothing after every retry

    # ── Degenerate generations ───────────────────────────────────────────────
    # The gguf backbone intermittently yields near-empty audio (perth's watermarker then fails
    # with "Padding size should be less ...") or no speech tokens at all (ValueError "No valid
    # speech tokens"). Retry (the library draws a fresh seed per call) inside the timed region,
    # and count failures so they stay visible in the numbers.
    TTS_MAX_ATTEMPTS = 3

    def _infer_resilient(text, ref_codes, prompt_text):
        """model.infer with retries; returns short silence if every attempt fails."""
        last = None
        for attempt in range(TTS_MAX_ATTEMPTS):
            try:
                return model.infer(text, ref_codes, prompt_text)
            except ValueError as e:
                if "No valid speech tokens" not in str(e):
                    raise
                last = e
            except RuntimeError as e:
                if "Padding size should be less" not in str(e):
                    raise
                # Near-empty audio: keep the take but skip the watermark, which cannot run on it.
                n_watermark_skipped[0] += 1
                saved, model.watermarker = model.watermarker, None
                try:
                    return model.infer(text, ref_codes, prompt_text)
                except (ValueError, RuntimeError) as e2:
                    last = e2
                finally:
                    model.watermarker = saved
            if attempt + 1 < TTS_MAX_ATTEMPTS:
                print(f"  retry {attempt + 1}/{TTS_MAX_ATTEMPTS - 1} after {type(last).__name__}: "
                      f"'{text[:40]}...'")
        # Every attempt failed: write a short silence so the sample is scored (~100% WER).
        n_failed[0] += 1
        print(f"  FAILED after {TTS_MAX_ATTEMPTS} attempts ({last}); writing silence: '{text[:40]}...'")
        return np.zeros(int(0.5 * sampling_rate), dtype=np.float32)

    def generate_tts(batch):
        """Synthesize speech for a minibatch of target texts; time the generation for RTFx.

        Loops one utterance at a time (no batched API); the whole loop is timed and divided by
        the batch size. Returned paths are relative to model_dir.
        """
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)

        # `encode_reference` takes a file path, so write the prompt clips to wavs first —
        # outside the timed region (disk I/O is not part of inference).
        prompt_rel_paths, prompt_full_paths = [], []
        prompt_texts = list(batch["prompt_text"])
        for sample_id, prompt_audio in zip(batch["id"], batch["prompt_audio"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            prel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
            ppath = os.path.join(model_dir, prel)
            if not os.path.exists(ppath):
                sf.write(
                    ppath,
                    np.asarray(prompt_audio["array"], dtype=np.float32),
                    prompt_audio["sampling_rate"],
                )
            prompt_rel_paths.append(prel)
            prompt_full_paths.append(ppath)

        # START TIMING (TTS generation for the whole minibatch)
        # CUDA events time the torch stream; the gguf backbone runs in llama.cpp, so use a wall
        # clock there instead.
        if is_gguf:
            t_start = time.perf_counter()
        else:
            torch.cuda.synchronize(device=args.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        wavs = []
        for text, prompt_text, prompt_path in zip(texts_to_generate, prompt_texts, prompt_full_paths):
            # Re-seed per sample so generation is reproducible regardless of batch layout.
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Reference encoding is timed: it is part of zero-shot cloning inference.
            ref_codes = model.encode_reference(prompt_path)
            wav = _infer_resilient(text, ref_codes, prompt_text)
            wavs.append(np.asarray(wav, dtype=np.float32).reshape(-1))

        # END TIMING
        if is_gguf:
            runtime = time.perf_counter() - t_start
        else:
            end_event.record()
            torch.cuda.synchronize(device=args.device)
            runtime = start_event.elapsed_time(end_event) / 1000.0
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(wavs, batch["id"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        batch["gen_audio_filepath"] = gen_paths
        batch["prompt_audio_filepath"] = prompt_rel_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    # Keep the reference clip/transcript regardless of --voice_clone (NeuTTS always needs one).
    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio"))

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # TTFA is per-request, so the probe drives one-row batches and returns before the batched
    # path. `model_dir`/`dataset_dir_name` are rebound to a temp dir first; generate_tts closes
    # over them (late binding), so its writes land there instead of the results tree.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        chunk_counts = []
        enc_s = []          # NeuCodec reference encode, per sample (torch)
        first_chunk_s = []  # llama.cpp prefill + first chunk, per sample

        def _gen(job, early_stop):
            """One utterance, streaming when the backbone supports it.

            NeuTTS.infer_stream() only exists for the gguf backbone; the torch path falls back
            to the whole-utterance clock.
            """
            ref_path = os.path.join(output_dir, "prompts", f"prompt_{job['id'][0]}.wav")
            pa = job["prompt_audio"][0]
            sf.write(ref_path, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])

            if not is_gguf:
                result = generate_tts(job)
                chunk_counts.append(1)
                return float(result["audio_length_s"][0]), None, None

            # Reference encoding stays inside the measurement, matching generate_tts. It is also
            # timed separately (torch/NeuCodec vs llama.cpp) to attribute a slow first chunk.
            first, chunks = None, []
            _t_enc = time.perf_counter()
            ref_codes = model.encode_reference(ref_path)
            enc_s.append(time.perf_counter() - _t_enc)
            _t_llm = time.perf_counter()
            for chunk in model.infer_stream(
                job[args.text_column][0], ref_codes, job["prompt_text"][0]
            ):
                if first is None:
                    first = time.perf_counter()
                    first_chunk_s.append(first - _t_llm)
                chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                if early_stop:
                    break
            chunk_counts.append(len(chunks))
            if not chunks:
                return np.zeros(1, dtype=np.float32), sampling_rate, None
            return np.concatenate(chunks), sampling_rate, first

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            # Only a gguf backbone streams (~0.5 s chunks); torch rows are whole-utterance.
            extra={"device": args.device, "note": "driven at batch size 1",
                   "quantized_backbone": is_gguf,
                   "chunks_per_utterance": chunk_counts,
                   "ref_encode_s": enc_s,
                   "prefill_to_first_chunk_s": first_chunk_s},
        )
        return

    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(generate_tts, dataset, starts, 1, args.warmup_steps)  # serial model: warm up on single samples
    n_watermark_skipped[0] = n_failed[0] = 0  # count timed samples only

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    for batch_start in tqdm(starts, desc="Generating"):
        batch = generate_tts(dataset[batch_start : batch_start + args.batch_size])

        # Append each sample to the JSONL immediately.
        for afp, ppath, alen, gtime, ref in zip(
            batch["gen_audio_filepath"], batch["prompt_audio_filepath"], batch["audio_length_s"],
            batch["generation_time_s"], batch["references"],
        ):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref, ppath if args.voice_clone else None))
        manifest_file.flush()

    manifest_file.close()
    if n_watermark_skipped[0] or n_failed[0]:
        print(f"WARNING: {n_watermark_skipped[0]} near-empty generation(s) skipped watermarking; "
              f"{n_failed[0]} sample(s) produced NO speech tokens after {TTS_MAX_ATTEMPTS} attempts "
              "and were written as silence (they will score ~100% WER).")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="neuphonic/neutts-nano",
        help="NeuTTS backbone repo id, loadable with the `neutts` package (a *-gguf repo uses llama_cpp).",
    )
    parser.add_argument(
        "--codec_repo",
        type=str,
        default="neuphonic/neucodec",
        help="Audio codec repo id ('neuphonic/neucodec' or 'neuphonic/distill-neucodec').",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="eSpeak language code for the phonemizer; None auto-resolves ('en-us' for neutts-nano).",
    )
    # --batch_size is the manifest-writing chunk size (NeuTTS has no batched API; generation is per sample).
    add_common_args(parser, batch_size=32, voice_clone=False)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Backbone dtype. The `neutts` package loads float32 (its documented path); pass "
        "'bfloat16' to cast the backbone LM after loading (codec stays float32).",
    )
    parser.add_argument(
        "--no_watermark",
        action="store_true",
        help="Disable the resemble-perth output watermark (applied by default inside NeuTTS.infer()).",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
