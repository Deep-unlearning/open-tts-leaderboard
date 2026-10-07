"""
TTS synthesis for the Open TTS Leaderboard (AuK backend, stage 1).
"""

import argparse
import contextlib
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from auk.infer.infer_auk import AukInfer

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_next_steps, set_seed, wav_rel_path, warm_up, write_entry,
)

# Zero-shot TTS instruction; upstream uses this one template for both English and Chinese.
ZERO_SHOT_TEMPLATE = 'Say the following with the same voice: "{text}"'

# Instruct TTS (no reference audio), upstream template.
INSTRUCT_TEMPLATE = (
    'Generate speech based on the following description: "{description}". '
    'The content to speak is: "{text}".'
)
# Neutral default description for --instruct. Deliberately plain: this measures intelligibility,
# not prompt engineering.
DEFAULT_VOICE_DESCRIPTION = (
    "A clear, neutral adult voice reading at a natural pace in a quiet room, "
    "with standard pronunciation and no strong emotion."
)

# AuK-Flash's distilled sampling recipe, copied from upstream `AukInfer.generate()`. The DMD
# student bakes in its own guidance, so re-adding CFG blows up the amplitude (it clips hard).
FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]
FLASH_NFE = 4

# Marker upstream appends to the text turn when there is no reference audio (Instruct TTS).
NO_PROMPT_AUDIO_MARKER = "|<no_prompt_audio>|"


def _resolve_model_dir(args):
    """Locate the AuK release dir (config.yaml + *.safetensors + vae.safetensors).

    The weights are not baked into the image, so the normal path is a Hub snapshot cached under
    HF_HOME; `--checkpoint_dir` points at a local copy when there is one.
    """
    local = args.checkpoint_dir
    if local:
        if os.path.isfile(os.path.join(local, "config.yaml")):
            return local
        print(f"--checkpoint_dir {local!r} has no config.yaml; falling back to the Hub.")
    from huggingface_hub import snapshot_download

    print(f"Downloading {args.model_id} from the Hub (~6.8 GB, cached under HF_HOME) ...")
    return snapshot_download(args.model_id)


def _resolve_qwen_dir(args):
    """Locate the Qwen2.5-Omni snapshot used as AuK's text/audio encoder.

    The shipped config.yaml points at a repo-relative path that does not exist here, so this is
    always passed explicitly to AukInfer(qwen_path=...).
    """
    local = args.qwen_dir
    if local:
        if os.path.isdir(local):
            return local
        print(f"--qwen_dir {local!r} not found; falling back to the Hub.")
    from huggingface_hub import snapshot_download

    print(f"Downloading {args.qwen_id} from the Hub (~12 GB, cached under HF_HOME) ...")
    return snapshot_download(args.qwen_id)


def _resolve_ckpt_file(model_dir):
    """Pick the model checkpoint inside a release dir (`auk_base` / `auk_flash`, not the VAE)."""
    candidates = [
        p for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if os.path.basename(p) != "vae.safetensors"
    ]
    if not candidates:
        raise FileNotFoundError(f"No model .safetensors (besides vae.safetensors) found in {model_dir}")
    if len(candidates) > 1:
        print(f"Multiple checkpoints in {model_dir}; using {candidates[0]}. Others: {candidates[1:]}")
    return candidates[0]


def _trim_trailing_silence(audio, sample_rate, top_db=40.0, keep_ms=200.0, frame_ms=20.0):
    """Drop the silent tail of a fixed-length generation, keeping `keep_ms` of it.

    AuK emits exactly the requested duration, so an over-estimate would otherwise count as
    generated audio in RTFx. Only the tail is touched, to avoid clipping the first phoneme.
    """
    if audio.size == 0:
        return audio
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = len(audio) // frame
    if n_frames < 2:
        return audio
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    peak = rms.max()
    if peak <= 0:
        return audio
    threshold = peak * (10.0 ** (-top_db / 20.0))
    voiced = np.nonzero(rms > threshold)[0]
    if voiced.size == 0:
        return audio
    end = (voiced[-1] + 1) * frame + int(sample_rate * keep_ms / 1000.0)
    return audio[: min(len(audio), end)]


def _write_prompt_wav(path, prompt_audio, max_seconds=0.0):
    """Write a dataset `prompt_audio` column value to a 16-bit PCM wav.

    With `max_seconds > 0` the truncated clip is written, so the model's conditioning and stage 3's
    SIM reference stay identical.
    """
    array = np.asarray(prompt_audio["array"], dtype=np.float32).reshape(-1)
    sr = int(prompt_audio["sampling_rate"])
    if max_seconds and len(array) > int(max_seconds * sr):
        array = array[: int(max_seconds * sr)]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, array, sr, subtype="PCM_16")
    return len(array) / sr


def _build_messages(text, prompt_path, args):
    """Compose the ChatML turn for one sample.

    `prompt_path` is set in both leaderboard modes (per-sample or fixed reference); the
    reference-free Instruct branch is reached only under `--instruct`.
    """
    if prompt_path is not None:
        instruction = args.instruction_template.format(text=text)
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "audio", "audio": prompt_path},
            ],
        }]
    # Instruct TTS: upstream's generate() appends this marker when there is no reference audio;
    # this path bypasses generate(), so do it here. AuK often speaks the marker aloud.
    instruction = INSTRUCT_TEMPLATE.format(description=args.voice_description, text=text)
    return [{
        "role": "user",
        "content": [{"type": "text", "text": instruction + NO_PROMPT_AUDIO_MARKER}],
    }]


def _estimate_gen_seconds(text, prompt_text, ref_seconds, args):
    """Target duration in seconds for one sample (AuK generates exactly this much audio).

    `ratio` reproduces upstream `get_gen_duration()`: the reference duration scaled by the UTF-8
    byte ratio of target text to reference transcript. Without a reference (`--instruct`) it falls
    back to a flat bytes-per-second rate (upstream has no heuristic for that case).
    """
    if args.gen_seconds > 0:
        return args.gen_seconds
    n_bytes = max(1, len(text.encode("utf-8")))
    use_ratio = args.duration_mode == "ratio" and prompt_text and ref_seconds
    if use_ratio:
        est = ref_seconds * n_bytes / max(1, len(prompt_text.encode("utf-8"))) / args.speed
    else:
        est = n_bytes / args.bytes_per_second / args.speed
    est *= args.duration_scale
    return float(min(max(est, args.min_gen_seconds), args.max_gen_seconds))


def main(args):
    # Set seed for reproducibility: the flow-matching sampler draws y0 noise and the VAE encoder
    # samples from the posterior (mean + randn * exp(log_std)), so generation is stochastic.
    set_seed(args.seed)

    if not args.instruct:
        # Fail fast, before the weight download: CFMEdit.build_cond_inputs imports qwen_omni_utils
        # lazily on the reference-audio path, so a broken install would otherwise surface late.
        import qwen_omni_utils  # noqa: F401

    model_dir = _resolve_model_dir(args)
    ckpt_path = _resolve_ckpt_file(model_dir)
    config_path = os.path.join(model_dir, "config.yaml")
    qwen_path = _resolve_qwen_dir(args)

    engine = AukInfer(
        config_path,
        ckpt_path,
        device=args.device,
        dtype=args.dtype,
        qwen_path=qwen_path,
    )
    sampling_rate = int(engine.target_sample_rate)  # 24000
    latent_hz = sampling_rate / engine.downsample_rate  # 50 Hz

    # AuK-Flash only works under its fixed (t_grid, CFG-off) recipe; upstream's generate() enforces
    # it and this path bypasses generate(), so apply the same lock here.
    nfe, cfg_strength, sway, t_grid = args.nfe, args.cfg_strength, args.sway_sampling_coef, None
    if args.t_grid:
        t_grid = [float(x) for x in args.t_grid.split(",")]
    if engine.is_flash:
        nfe, cfg_strength, sway, t_grid = FLASH_NFE, 0.0, None, FLASH_T_GRID
        print("AuK-Flash detected: sampling locked to the distilled 4-step / CFG-off recipe.")

    n_params = sum(p.numel() for p in engine.model.transformer.parameters())
    print(
        f"Loaded AuK {args.model_id} from {model_dir} (sr={sampling_rate}, latent={latent_hz:.0f} Hz, "
        f"nfe={nfe}, cfg={cfg_strength}, dtype={args.dtype}, device={engine.device})"
    )
    print(f"TTS model size: {n_params / 1e9:.2f}B parameters (DiT backbone; + Qwen2.5-Omni text encoder)")

    is_cuda = torch.device(args.device).type == "cuda"

    def _autocast():
        """Fresh autocast context per call (matches upstream `_run`'s `torch.autocast("cuda", ...)`)."""
        return torch.autocast("cuda", dtype=engine.dtype) if is_cuda else contextlib.nullcontext()

    # ── Attention-memory budget, calibrated at runtime ───────────────────────
    # Cost model: `n * max_total_latent_len**2`, a proxy for the [B, heads, T, T] attention mask.
    # `None` = no budget yet: run the full batch, and on the first OOM derive the budget from the
    # failed cost. --max_attn_cost skips that probe.
    state = {"attn_budget": args.max_attn_cost or None}
    # The failed attempt died partway through, so its true peak was higher; halving adds margin.
    ATTN_BUDGET_SAFETY = 2

    def _plan_chunks(indices, total_lens):
        """Greedily chunk `indices` so each chunk's `len(chunk) * max_total_len**2` fits the budget.

        Chunks stay contiguous to preserve manifest order; an over-budget sample forms a chunk of one.
        """
        if state["attn_budget"] is None:
            return [list(indices)]
        chunks, cur, cur_max = [], [], 0
        for i in indices:
            new_max = max(cur_max, total_lens[i])
            if cur and (len(cur) + 1) * new_max**2 > state["attn_budget"]:
                chunks.append(cur)
                cur, cur_max = [i], total_lens[i]
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            chunks.append(cur)
        return chunks

    @torch.inference_mode()
    def _sample_chunk(refs, messages_batch, gen_latent_lens):
        """One batched `CFMEdit.sample()` call; returns a list of [T] float32 numpy waveforms.

        `refs` holds one [1, T] CPU waveform per sample (empty in Instruct mode). Mirrors upstream
        `AukInfer._run()` with B > 1: encode and right-pad references, sample once, then slice and
        decode each sample's generated span.
        """
        device = engine.device
        batch = len(refs)
        cloning = any(r.shape[-1] > 0 for r in refs)

        ref_lens, ref_latents_list = [], []
        if cloning:
            for wav in refs:
                # Per-sample VAE encode: exact, and cheap next to the sampler.
                a = wav.to(device).unsqueeze(0)  # [1, 1, T]
                rl = a.shape[-1] // engine.downsample_rate
                lat, enc_lens = engine.vae_model.encoding_and_normalization(
                    a,
                    sample_lengths=torch.tensor([rl * engine.downsample_rate], dtype=torch.long, device=device),
                )
                rl = min(rl, int(enc_lens[0].item()))
                ref_lens.append(rl)
                ref_latents_list.append(lat[0, :rl].float())
            max_ref = max(ref_lens)
            ref_latents = torch.zeros(batch, max_ref, engine.latent_dim, device=device, dtype=torch.float32)
            for i, lat in enumerate(ref_latents_list):
                ref_latents[i, : lat.shape[0]] = lat
        else:
            ref_lens = [0] * batch
            ref_latents = torch.zeros(batch, 0, engine.latent_dim, device=device, dtype=torch.float32)

        ref_lens_t = torch.tensor(ref_lens, dtype=torch.long, device=device)
        total_lens_t = torch.tensor(
            [r + g for r, g in zip(ref_lens, gen_latent_lens)], dtype=torch.long, device=device
        )

        with _autocast():
            cond_inputs = engine.model.build_cond_inputs(messages_batch, engine.model.text_processor)
            generated, _ = engine.model.sample(
                cond=ref_latents,
                text=cond_inputs,
                duration=total_lens_t,
                lens=ref_lens_t,
                steps=nfe,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway,
                t_grid=t_grid,
                no_ref_audio=False,
                seed=args.sample_seed if args.sample_seed >= 0 else None,
            )  # [B, T_total_max, D]

        audios = []
        for i in range(batch):
            rl, tl = ref_lens[i], int(total_lens_t[i].item())
            gen_latent = generated[i, rl:tl, :].unsqueeze(0)  # [1, T_new, D]
            if gen_latent.shape[1] == 0 or not torch.isfinite(gen_latent).all():
                audios.append(np.zeros(0, dtype=np.float32))
                continue
            gen_latent = engine.vae_model.denormalize(gen_latent).permute(0, 2, 1)  # [1, D, T_new]
            wav = engine.vae_model.inference_from_latents(gen_latent).cpu()
            if wav.ndim == 3:
                wav = wav.squeeze(0)
            wav = wav.reshape(-1).to(torch.float32).numpy()
            audios.append(wav if np.isfinite(wav).all() else np.zeros(0, dtype=np.float32))
        return audios

    def generate_tts(batch):
        """Synthesize a minibatch; time the whole batch generation for RTFx."""
        texts = list(batch[args.text_column])
        minibatch_size = len(texts)
        prompt_paths = batch.get("_prompt_path", [None] * minibatch_size)
        ref_seconds = batch.get("_ref_seconds", [None] * minibatch_size)
        prompt_texts = batch.get("prompt_text", [None] * minibatch_size)

        gen_seconds = [
            _estimate_gen_seconds(texts[i], prompt_texts[i], ref_seconds[i], args)
            for i in range(minibatch_size)
        ]
        gen_latent_lens = [max(1, int(np.ceil(s * latent_hz))) for s in gen_seconds]

        audios = [None] * minibatch_size
        wasted_s = 0.0  # GPU time burned by sub-batches that OOM'd and were retried smaller

        def _run_chunk(chunk):
            refs = []
            for i in chunk:
                if prompt_paths[i] is None:
                    refs.append(torch.zeros(1, 0))
                else:
                    # _load_audio downmixes to mono and resamples to the model's 24 kHz.
                    wav, _ = engine._load_audio(prompt_paths[i])
                    refs.append(wav)
            messages_batch = [_build_messages(texts[i], prompt_paths[i], args) for i in chunk]
            out = _sample_chunk(refs, messages_batch, [gen_latent_lens[i] for i in chunk])
            for i, audio in zip(chunk, out):
                audios[i] = audio

        def _run_chunk_with_oom_retry(chunk):
            """Run `chunk`; on OOM, tighten the budget and retry the two halves.

            The OOM'd attempt's GPU time is subtracted from the timed region so RTFx reflects only
            work that produced audio.
            """
            nonlocal wasted_s
            attempt_start = torch.cuda.Event(enable_timing=True) if is_cuda else None
            attempt_end = torch.cuda.Event(enable_timing=True) if is_cuda else None
            if is_cuda:
                attempt_start.record()
            try:
                _run_chunk(chunk)
                return
            except torch.OutOfMemoryError:
                if len(chunk) == 1:  # can't split further — a single sample OOMs on its own
                    raise
            if is_cuda:
                attempt_end.record()
                torch.cuda.synchronize(device=args.device)
                wasted_s += attempt_start.elapsed_time(attempt_end) / 1000.0
            torch.cuda.empty_cache()

            max_len = max(gen_latent_lens[i] + _ref_latent_len(i) for i in chunk)
            failed_cost = len(chunk) * max_len**2
            budget = max(1, failed_cost // ATTN_BUDGET_SAFETY)
            if state["attn_budget"] is None or budget < state["attn_budget"]:
                state["attn_budget"] = budget
            print(
                f"OOM on a sub-batch of {len(chunk)} (max {max_len} latent frames, cost {failed_cost}); "
                f"attention budget now {state['attn_budget']} — pass "
                f"--max_attn_cost={state['attn_budget']} to skip this probe on a rerun. "
                "Halving and retrying.",
                file=sys.stderr,
                flush=True,
            )
            mid = len(chunk) // 2
            _run_chunk_with_oom_retry(chunk[:mid])
            _run_chunk_with_oom_retry(chunk[mid:])

        def _ref_latent_len(i):
            """Reference length in latent frames, for the cost model (0 without cloning)."""
            if not ref_seconds[i]:
                return 0
            return int(ref_seconds[i] * latent_hz)

        total_lens = {i: gen_latent_lens[i] + _ref_latent_len(i) for i in range(minibatch_size)}

        # START TIMING (TTS batch generation)
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_wall = time.perf_counter()

        chunks = _plan_chunks(range(minibatch_size), total_lens)
        if len(chunks) > 1:
            print(
                f"Split a minibatch of {minibatch_size} into sub-batches {[len(c) for c in chunks]} "
                f"(max {max(total_lens.values())} latent frames, attention budget {state['attn_budget']})",
                file=sys.stderr,
                flush=True,
            )
        for chunk in chunks:
            _run_chunk_with_oom_retry(chunk)

        # END TIMING
        if is_cuda:
            end_event.record()
            torch.cuda.synchronize(device=args.device)
            runtime = start_event.elapsed_time(end_event) / 1000.0
        else:
            runtime = time.perf_counter() - start_wall
        # Exclude time spent on OOM'd attempts that produced no audio.
        runtime = max(runtime - wasted_s, 1e-6)
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(audios, batch["id"]):
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if args.trim_trailing_silence:
                audio = _trim_trailing_silence(
                    audio, sampling_rate, top_db=args.trim_top_db, keep_ms=args.trim_keep_ms
                )
            if audio.size == 0:  # degenerate generation; write a stub so the pipeline continues
                audio = np.zeros(int(0.1 * sampling_rate), dtype=np.float32)
            # Store the path relative to model_dir_out (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            sf.write(os.path.join(model_dir_out, rel_path), audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        batch["gen_audio_filepath"] = gen_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    # ── Flat, bucket-friendly layout, per model ──────────────────────────────
    # NOTE: must match resolve_mode() in submit_jobs.sh, which builds the same manifest names.
    mode_suffix = "_voice_clone" if args.voice_clone else ("_instruct" if args.instruct else "")
    paths = output_paths(args, mode_suffix)
    model_dir_out, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    # Reference clips live under results/ so stage 3 SIM can read them; fixed-voice mode writes its
    # single fixed_prompt.wav there too (the ChatML turn needs a path on disk).
    if not args.instruct:
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

    # The reference columns are kept in every mode (fixed-voice mode may take its one reference
    # from the split); they are dropped below when not cloning.
    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio"))

    # Dataset references are needed unless --instruct, or fixed-voice mode got an external clip.
    needs_dataset_prompt = not args.instruct and not args.prompt_audio_path
    missing = [c for c in ("prompt_text", "prompt_audio") if c not in dataset.column_names]
    if missing and needs_dataset_prompt:
        raise ValueError(
            f"Dataset {args.dataset_path}/{args.dataset}/{args.split} is missing {missing}. "
            "Pass --prompt_audio_path/--prompt_text to supply a fixed reference externally, or "
            "--instruct to run upstream's reference-free Instruct TTS."
        )

    # ── Fixed-voice mode: resolve the ONE reference used for the whole split ──
    # `fixed_prompt` stays None in the other two modes; _with_prompts and the TTFA probe branch on it.
    fixed_prompt = None
    if not args.voice_clone and not args.instruct:
        fixed_rel = os.path.join(dataset_dir_name, "prompts", "fixed_prompt.wav")
        fixed_full = os.path.join(model_dir_out, fixed_rel)
        if args.prompt_audio_path:
            if not args.prompt_text:
                raise ValueError(
                    "--prompt_audio_path requires --prompt_text (the duration estimate scales the "
                    "reference's length by the target/reference UTF-8 byte ratio)."
                )
            audio, sr = sf.read(args.prompt_audio_path, dtype="float32", always_2d=True)
            seconds = _write_prompt_wav(
                fixed_full, {"array": audio.mean(axis=1), "sampling_rate": sr},
                max_seconds=args.max_ref_seconds,
            )
            text = args.prompt_text
            src = args.prompt_audio_path
        else:
            # Default: the split's own sample #--fixed_prompt_index, which keeps the reference
            # language-matched.
            if not 0 <= args.fixed_prompt_index < len(dataset):
                raise ValueError(
                    f"--fixed_prompt_index={args.fixed_prompt_index} is out of range for a "
                    f"{len(dataset)}-sample split."
                )
            row = dataset[args.fixed_prompt_index]
            seconds = _write_prompt_wav(
                fixed_full, row["prompt_audio"], max_seconds=args.max_ref_seconds
            )
            text = row["prompt_text"]
            src = f"{args.dataset}/{args.split} sample #{args.fixed_prompt_index} (id={row.get('id')})"
        fixed_prompt = {"path": fixed_full, "text": text, "seconds": seconds}
        print(
            f"Fixed-voice mode: one {seconds:.2f}s reference for all {len(dataset)} samples, "
            f"from {src}. SIM is skipped (no per-sample reference)."
        )

    # Not cloning: drop the reference columns so nothing triggers audio decoding.
    if not args.voice_clone:
        dataset = dataset.remove_columns([c for c in ("prompt_text", "prompt_audio") if c in dataset.column_names])

    def _prepare_prompt(sample_id, prompt_audio):
        """Materialize a sample's reference clip; return (rel_path, abs_path, seconds).

        The wav is read both by AuK's Qwen encoder (via the ChatML `audio` path) and by stage 3 SIM.
        """
        rel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
        path = os.path.join(model_dir_out, rel)
        if os.path.exists(path) and args.resume:
            info = sf.info(path)
            return rel, path, info.frames / info.samplerate
        seconds = _write_prompt_wav(path, prompt_audio, max_seconds=args.max_ref_seconds)
        return rel, path, seconds

    probe_out_dir = model_dir_out  # real results dir, captured before the probe rebinds it

    # ── TTFA probe: latency only (see scripts/ttfa_probe.py) ─────────────────
    if args.ttfa_probe != 0:
        # Imported HERE, not at module scope: the probe module is injected only by
        # submit_ttfa_jobs.sh, so a top-level import would break every ordinary job.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir_out = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        os.makedirs(os.path.join(model_dir_out, dataset_dir_name, "prompts"), exist_ok=True)

        columns = dataset.column_names
        jobs = []
        for i in sample_indices(args.ttfa_probe, len(dataset)):
            job = {c: [dataset[i][c]] for c in columns}
            if args.voice_clone:
                _, path, seconds = _prepare_prompt(dataset[i]["id"], dataset[i]["prompt_audio"])
                job["_prompt_path"] = [path]
                job["_ref_seconds"] = [seconds]
            elif fixed_prompt is not None:
                job["_prompt_path"] = [fixed_prompt["path"]]
                job["_ref_seconds"] = [fixed_prompt["seconds"]]
                job["prompt_text"] = [fixed_prompt["text"]]
            jobs.append(job)

        def _gen(job):
            result = generate_tts(job)
            return float(result["audio_length_s"][0]), None, None

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "voice_clone": args.voice_clone,
                   "voice_mode": "clone" if args.voice_clone else ("instruct" if args.instruct else "fixed"),
                   "note": "driven at batch size 1; no streaming API (TTFA == full generation)"},
        )
        return

    # ── Manifest (written incrementally; resume skips rows already in it) ───
    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def _with_prompts(batch):
        """A dataset slice (dict of lists) with the reference clips materialized."""
        n = len(batch["id"])
        if args.voice_clone:
            rels, prompt_paths, seconds = [], [], []
            for j in range(n):
                rel, path, sec = _prepare_prompt(batch["id"][j], batch["prompt_audio"][j])
                rels.append(rel)
                prompt_paths.append(path)
                seconds.append(sec)
            batch["_prompt_rel"] = rels
            batch["_prompt_path"] = prompt_paths
            batch["_ref_seconds"] = seconds
        elif fixed_prompt is not None:
            # One clip for the whole split. `prompt_text` is needed for the duration estimate.
            batch["_prompt_path"] = [fixed_prompt["path"]] * n
            batch["_ref_seconds"] = [fixed_prompt["seconds"]] * n
            batch["prompt_text"] = [fixed_prompt["text"]] * n
        return batch

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    # Warm-up absorbs the first call's CUDA autotuning + lazy kernel loads, which would otherwise
    # all land on batch 0's RTFx.
    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(lambda b: generate_tts(_with_prompts(b)), dataset, starts, args.batch_size, args.warmup_steps)
    for batch_start in tqdm(starts, desc="Generating"):
        batch = generate_tts(_with_prompts(dataset[batch_start : batch_start + args.batch_size]))

        # Append each sample to the JSONL immediately.
        for j, (afp, alen, gtime, ref) in enumerate(zip(
            batch["gen_audio_filepath"], batch["audio_length_s"],
            batch["generation_time_s"], batch["references"],
        )):
            entry = manifest_entry(afp, alen, gtime, ref, batch["_prompt_rel"][j] if args.voice_clone else None)
            write_entry(manifest_file, entry)
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, default="tencent/AuK",
        help="Model id (also used for manifest naming). 'tencent/AuK' or 'tencent/AuK-Flash'.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="",
        help="OPTIONAL local release dir (config.yaml + auk_*.safetensors + vae.safetensors). "
             "Empty (default) downloads --model_id from the Hub into the HF cache.",
    )
    parser.add_argument(
        "--qwen_id", type=str, default="Qwen/Qwen2.5-Omni-3B",
        help="Text/audio encoder repo. The shipped config.yaml points at a repo-relative path "
             "that does not exist here, so this is always resolved and passed explicitly.",
    )
    parser.add_argument(
        "--qwen_dir", type=str, default="",
        help="OPTIONAL local Qwen2.5-Omni snapshot. Empty (default) downloads --qwen_id from the "
             "Hub into the HF cache.",
    )
    # --batch_size is NOMINAL: sub-batched down to fit --max_attn_cost when sequences get long.
    add_common_args(parser, batch_size=16, voice_clone=False)
    parser.set_defaults(device="cuda:0")
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"],
        help="Autocast dtype for the sampler (AukInfer keeps the weights in fp32).",
    )
    parser.add_argument(
        "--max_attn_cost", type=int, default=0,
        help="Initial cap on `n * max_total_latent_frames^2` per sample() call; a minibatch "
             "exceeding it is split. Default 0 = start unbounded and calibrate from the first OOM.",
    )

    # ── Sampling (ignored for AuK-Flash, which is locked to its distilled recipe) ──
    parser.add_argument("--nfe", type=int, default=32, help="ODE steps (upstream default 32).")
    parser.add_argument("--cfg_strength", type=float, default=2.0, help="CFG strength (upstream default 2.0).")
    parser.add_argument(
        "--sway_sampling_coef", type=float, default=-1.0,
        help="Sway sampling coefficient (upstream default -1.0).",
    )
    parser.add_argument(
        "--t_grid", type=str, default="",
        help="Explicit comma-separated sampling times, overriding --nfe/--sway_sampling_coef.",
    )

    # ── Duration control (AuK generates EXACTLY the requested length) ──────────
    parser.add_argument(
        "--duration_mode", type=str, default="ratio", choices=["ratio", "rate"],
        help="'ratio' (default) = upstream get_gen_duration(): reference duration scaled by the "
             "UTF-8 byte ratio of target text to prompt_text. 'rate' = a flat "
             "--bytes_per_second estimate. 'ratio' falls back to 'rate' when there is no "
             "reference (--instruct).",
    )
    parser.add_argument(
        "--gen_seconds", type=float, default=0.0,
        help="Fixed target duration for EVERY sample, overriding the estimate. 0 = estimate.",
    )
    parser.add_argument(
        "--bytes_per_second", type=float, default=13.0,
        help="Speaking rate for --duration_mode rate, in UTF-8 bytes of target text per second. "
             "~13 fits both English (~14 chars/s, 1 byte each) and Mandarin (~4.5 chars/s, 3 bytes).",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Divides the duration estimate (>1 = faster).")
    parser.add_argument(
        "--duration_scale", type=float, default=1.0,
        help="Global multiplier on the duration estimate. >1 buys headroom against truncated "
             "endings at the cost of more trailing silence (trimmed by default).",
    )
    parser.add_argument("--min_gen_seconds", type=float, default=0.5, help="Lower clamp on the estimate.")
    parser.add_argument(
        "--max_gen_seconds", type=float, default=30.0,
        help="Upper clamp on the estimate. Upstream notes a ~30 s reference-plus-target budget.",
    )
    parser.add_argument(
        "--max_ref_seconds", type=float, default=0.0,
        help="Truncate each reference clip to at most N seconds before use. 0 (default) = off. "
             "The truncated clip is what is written to prompts/, so the model's conditioning and "
             "stage 3's SIM reference stay identical.",
    )

    # ── Output post-processing ────────────────────────────────────────────────
    parser.add_argument(
        "--trim_trailing_silence", action=argparse.BooleanOptionalAction, default=True,
        help="Trim the silent tail of each fixed-length generation (default on), so an over-estimated "
             "duration does not inflate RTFx.",
    )
    parser.add_argument("--trim_top_db", type=float, default=40.0, help="Silence threshold below the clip's peak frame RMS.")
    parser.add_argument("--trim_keep_ms", type=float, default=200.0, help="Tail retained after the last voiced frame.")

    # --voice_clone clones each sample's OWN prompt_audio (+ SIM); --no-voice_clone (default) uses
    # ONE fixed reference for the whole split, comparable with the fixed-voice backends.
    parser.add_argument(
        "--fixed_prompt_index", type=int, default=0,
        help="--no-voice_clone only: which sample of the split supplies the single reference. "
             "Taking it from the split itself keeps the voice language-matched.",
    )
    parser.add_argument(
        "--prompt_audio_path", type=str, default=None,
        help="--no-voice_clone only: use this wav as the fixed reference instead of a dataset "
             "sample (requires --prompt_text). Lets one voice be held constant across languages.",
    )
    parser.add_argument(
        "--prompt_text", type=str, default=None,
        help="Transcript of --prompt_audio_path. Required with it: the duration estimate scales "
             "the reference's length by the target/reference byte ratio.",
    )
    parser.add_argument(
        "--instruct", action="store_true",
        help="DIAGNOSTIC, not the leaderboard path: upstream's reference-free Instruct TTS "
             "(--voice_description, no reference audio). Requires --no-voice_clone and writes "
             "'_instruct'-suffixed manifests. AuK often speaks the |<no_prompt_audio>| marker aloud.",
    )
    parser.add_argument(
        "--voice_description", type=str, default=DEFAULT_VOICE_DESCRIPTION,
        help="Natural-language voice description for --instruct.",
    )
    parser.add_argument(
        "--instruction_template", type=str, default=ZERO_SHOT_TEMPLATE,
        help="Zero-shot TTS instruction; must contain a {text} placeholder.",
    )

    parser.add_argument(
        "--sample_seed", type=int, default=-1,
        help="Seed re-applied inside CFMEdit.sample() for the y0 noise draw. -1 (default) = None, "
             "i.e. let the global seed drive it; note upstream re-seeds per sample, which makes "
             "every sample in a batch start from the SAME noise when this is set.",
    )

    args = parser.parse_args()

    # --instruct is a third mode layered under --no-voice_clone.
    if args.instruct and args.voice_clone:
        parser.error("--instruct requires --no-voice_clone (it is the reference-free mode).")
    for flag, value in (("--fixed_prompt_index", args.fixed_prompt_index != 0),
                        ("--prompt_audio_path", args.prompt_audio_path),
                        ("--prompt_text", args.prompt_text)):
        if value and (args.voice_clone or args.instruct):
            parser.error(f"{flag} only applies to fixed-voice mode (--no-voice_clone without --instruct).")

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
