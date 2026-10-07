"""
TTS synthesis for the Open TTS Leaderboard (Pocket TTS backend, stage 1).

Pocket TTS (kyutai/pocket-tts) is Kyutai's ~110M-parameter CPU-first streaming TTS (flow-matching
LM over Mimi latents, 24 kHz output), loaded in-process via the `pocket_tts` package.

`pocket_tts` ships one checkpoint per language; LANGUAGE_MAP translates the dataset language code
to the config name. `--variant` picks a non-default checkpoint by suffixing it (`24l` = 24-layer
preview models, `2026-01` / `2026-04` = dated English checkpoints); a non-empty variant is also
appended to the output suffix so variants never overwrite each other under the shared model id.

Voice cloning (`--voice_clone`) conditions each sample on the per-sample `prompt_audio`; outputs
get a `_voice_clone` suffix. The default (`--no-voice_clone`) uses a predefined voice (`--speaker`,
default per language) and skips SIM.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from pocket_tts import TTSModel

from run_eval_utils import (
    add_common_args, count_parameters, load_done_entries, load_tts_dataset, manifest_entry,
    open_manifest, output_paths, pending_batches, print_next_steps, set_seed, warm_up, wav_rel_path,
    write_entry,
)

LANGUAGE_MAP = {
    "en": "english",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "fr": "french_24l",  # no plain `french` config upstream
}

# Languages whose single checkpoint name already carries the variant: --variant does not change
# the config loaded, but still affects the output suffix (kept in lockstep with submit_jobs.sh).
_VARIANT_EXEMPT = {"fr"}

# (language, variant) pairs whose config is NOT `<base>_<variant>`. The 24-layer English model is
# only shipped under a dated name; the manifest suffix still reads `_24l` (the requested variant).
_VARIANT_ALIASES = {("en", "24l"): "english_2026-04_24l"}

# Default predefined voice per language, copied from pocket_tts.default_parameters
# (DEFAULT_VOICE_FOR_LANGUAGE + DEFAULT_VOICE_FALLBACK). Only used with --no-voice_clone.
DEFAULT_VOICE_FOR_LANGUAGE = {
    "it": "giovanni",
    "es": "lola",
    "de": "juergen",
    "pt": "rafael",
    "fr": "estelle",
}
DEFAULT_VOICE_FALLBACK = "alba"


def _resolve_config_name(code, variant):
    """Translate (dataset language code, variant) to a pocket_tts config name, validated against
    the configs installed in the image so a bad name fails fast with the real list."""
    base = LANGUAGE_MAP.get(code)
    if base is None:
        raise ValueError(
            f"Pocket TTS has no checkpoint for language '{code}'. Supported: "
            f"{sorted(LANGUAGE_MAP)}. Gate the combo out in submit_jobs.sh instead of running "
            f"it — there is no multilingual model to fall back on, so the WER would be meaningless."
        )
    name = base
    if variant and code not in _VARIANT_EXEMPT:
        name = _VARIANT_ALIASES.get((code, variant), f"{base}_{variant}")

    from pocket_tts.utils.config import CONFIGS_DIR

    available = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
    if name not in available:
        raise ValueError(
            f"Config '{name}' (language '{code}', variant '{variant}') is not available in the "
            f"installed pocket-tts. Available: {available}"
        )
    return name


def _resolve_speaker(args):
    """Predefined voice for --no-voice_clone: explicit --speaker, else the per-language default."""
    if args.speaker:
        return args.speaker
    return DEFAULT_VOICE_FOR_LANGUAGE.get(args.language, DEFAULT_VOICE_FALLBACK)


def main(args):
    # Set seed for reproducibility (the flow LM samples with temperature).
    seed = args.seed
    set_seed(seed)

    config_name = _resolve_config_name(args.language, args.variant)

    # `temp` is omitted when --temperature is unset, leaving the library's per-config default
    # (0.3 for the English models, 0.7 otherwise).
    load_kwargs = {
        "language": config_name,
        "sampler_decode_steps": args.sampler_decode_steps,
        "eos_threshold": args.eos_threshold,
        "quantize": args.quantize,
    }
    if args.temperature is not None:
        load_kwargs["temp"] = args.temperature
    model = TTSModel.load_model(**load_kwargs)
    # load_model() always returns a CPU model; there is no device argument.
    model.to(args.device)

    if args.num_threads:
        # pocket_tts pins torch to 1 thread at import; only override if explicitly asked.
        torch.set_num_threads(args.num_threads)

    sampling_rate = model.sample_rate  # 24000
    n_params = count_parameters(model)
    print(
        f"Loaded Pocket TTS {args.model_id} (config={config_name}, sr={sampling_rate}, "
        f"temp={model.temp}, device={model.device}, torch_threads={torch.get_num_threads()})"
    )
    print(f"TTS model size: {n_params / 1e6:.0f}M parameters")

    # Fail fast if pocket_tts silently fell back to the no-voice-cloning weights (gated download
    # failed); otherwise the run would only raise on the first sample.
    if args.voice_clone and not model.has_voice_cloning:
        raise RuntimeError(
            "pocket_tts fell back to the weights WITHOUT voice cloning (the Mimi encoder is zeroed "
            "out), so --voice_clone cannot work. The download of kyutai/pocket-tts failed: check "
            "network access, or accept the terms at https://huggingface.co/kyutai/pocket-tts and "
            "set HF_TOKEN. Use --no-voice_clone to run with a predefined voice instead."
        )
    print(f"Voice cloning available: {model.has_voice_cloning}")

    is_cuda = torch.device(args.device).type == "cuda"

    # In fixed-voice mode the state is fetched once. `generate_audio` mutates the state it is
    # given, so it must be reused with copy_state=True.
    fixed_state = None
    if not args.voice_clone:
        speaker = _resolve_speaker(args)
        fixed_state = model.get_state_for_audio_prompt(speaker)
        print(f"Using predefined voice {speaker!r} (no cloning, SIM stage skipped).")

    def synth_one_streaming(text, prompt_path):
        """Like synth_one, but timestamps the first chunk: (audio_1d_numpy, first_audio_ts).

        Used only by the TTFA probe (via generate_audio_stream). Reference encoding stays inside
        the measured region, matching synth_one.
        """
        state = (
            fixed_state
            if fixed_state is not None
            else model.get_state_for_audio_prompt(Path(prompt_path))
        )
        first, chunks = None, []
        for chunk in model.generate_audio_stream(
            state,
            text,
            max_tokens=args.max_tokens,
            frames_after_eos=args.frames_after_eos,
            copy_state=True,
        ):
            arr = np.asarray(chunk.reshape(-1).detach().to(torch.float32).cpu()
                             if hasattr(chunk, "detach") else chunk, dtype=np.float32).reshape(-1)
            if first is None:
                first = time.perf_counter()
            chunks.append(arr)
        if not chunks:
            return np.zeros(1, dtype=np.float32), None
        return np.concatenate(chunks), first

    def synth_one(text, prompt_path):
        """Synthesize one text; return (audio_1d_numpy, elapsed_seconds).

        Reference encoding (get_state_for_audio_prompt) is inside the timed region, as for the
        other cloning backends: it is per-utterance work for a zero-shot cloning model.
        """
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
        start = time.perf_counter()
        state = (
            fixed_state
            if fixed_state is not None
            # Pass a Path, not a str: get_state_for_audio_prompt treats a str as a possible URL /
            # hf:// ref / predefined-voice name before falling through to a local file.
            else model.get_state_for_audio_prompt(Path(prompt_path))
        )
        audio = model.generate_audio(
            state,
            text,
            max_tokens=args.max_tokens,
            frames_after_eos=args.frames_after_eos,
            copy_state=True,  # never let generation consume the state we may reuse
        )
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
        elapsed = time.perf_counter() - start
        # generate_audio returns a 1D float32 tensor of PCM samples.
        return audio.reshape(-1).detach().to(torch.float32).cpu().numpy(), elapsed

    # Flat, bucket-friendly layout, per model
    variant_suffix = f"_{args.variant}" if args.variant else ""
    mode_suffix = variant_suffix + ("_voice_clone" if args.voice_clone else "")
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    # Reference clips must live under results/ so stage 3 SIM can read them.
    prompt_dir = os.path.join(output_dir, "prompts")
    if args.voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    # Keep `id`, target text, and (when cloning) the per-sample reference.
    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio") if args.voice_clone else ())

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="ttfa_refs_")

        def _ttfa_ref(i):
            if not args.voice_clone:
                return None
            path = os.path.join(tmp_dir, f"prompt_{dataset[i]['id']}.wav")
            _write_prompt_wav(path, dataset[i]["prompt_audio"])
            return path

        jobs = [{"text": dataset[i][args.text_column], "prompt_path": _ttfa_ref(i)}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        # Always a full generation (no early_stop parameter): pocket_tts decodes in a daemon thread
        # that leaving the generator does NOT stop, so an early-stopped row left it decoding the
        # rest of the utterance into the next row's measurement (TTFA 10-18x too high).
        def _ttfa_gen(job):
            audio, first = synth_one_streaming(job["text"], job["prompt_path"])
            return audio, sampling_rate, first

        run_probe(
            _ttfa_gen,
            jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "voice_clone": args.voice_clone},
        )
        return

    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    def prompt_for(row):
        """When cloning, persist a row's reference clip; (prompt_rel_path, prompt_path) or Nones."""
        if not args.voice_clone:
            return None, None
        prompt_rel_path = os.path.join(dataset_dir_name, "prompts", f"prompt_{row['id']}.wav")
        prompt_path = os.path.join(model_dir, prompt_rel_path)
        if not os.path.exists(prompt_path):
            _write_prompt_wav(prompt_path, row["prompt_audio"])
        return prompt_rel_path, prompt_path

    # Serial generation: batches of one sample. Warm-up first: the first call is several times
    # slower (CUDA autotuning), which would otherwise all land on sample 0's RTFx.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda b: synth_one(b[args.text_column][0], prompt_for({c: b[c][0] for c in b})[1]),
            dataset, starts, 1, args.warmup_steps)

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    n_written = 0
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        sample_id = row["id"]
        text = row[args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        # When cloning, persist the reference clip to disk
        prompt_rel_path, prompt_path = prompt_for(row)

        # Re-seed per sample so generation is reproducible regardless of where a resume picked up.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        audio, elapsed = synth_one(text, prompt_path)
        sf.write(path, audio, sampling_rate)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, elapsed, text, prompt_rel_path))
        # --batch_size only sets the manifest-flush interval; generation is one sample at a time.
        n_written += 1
        if n_written % args.batch_size == 0:
            manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


def _write_prompt_wav(path, prompt_audio):
    """Write a dataset `prompt_audio` column value to a 16-bit PCM wav."""
    sf.write(
        path,
        np.asarray(prompt_audio["array"], dtype=np.float32),
        prompt_audio["sampling_rate"],
        subtype="PCM_16",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, default="kyutai/pocket-tts", help="Model id (also used for manifest naming)."
    )
    # --batch_size is the manifest-flush granularity ONLY: Pocket TTS has no batched generation API.
    add_common_args(parser, batch_size=32, voice_clone=False)
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Dataset language code, mapped to a pocket_tts config via LANGUAGE_MAP.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        help="Non-default checkpoint for the same language, suffixed onto the config name: "
             "'24l' for the 24-layer preview models, or '2026-01'/'2026-04' for the dated English "
             "checkpoints. Empty (default) = the distilled production model. Also appended to the "
             "output/manifest suffix.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="Override torch's thread count. pocket_tts pins it to 1 at import; 0 (default) leaves "
             "that as-is so the model is benchmarked as shipped. Only meaningful with --device=cpu.",
    )
    # Generation params. Defaults match the library's own (see load_model / MAX_TOKEN_PER_CHUNK).
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Unset (default) = the config's `default_temperature` (0.3 for "
             "the English models, 0.7 otherwise).",
    )
    parser.add_argument(
        "--sampler_decode_steps", type=int, default=1,
        help="Lagrangian Self Distillation decode steps (library default 1)."
    )
    parser.add_argument(
        "--eos_threshold", type=float, default=-4.0, help="End-of-sequence threshold (library default -4.0)."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Max text tokens per generated CHUNK (library default 50). generate_audio splits "
             "longer text into sentence chunks and concatenates — it does NOT truncate.",
    )
    parser.add_argument(
        "--frames_after_eos",
        type=int,
        default=None,
        help="Extra frames generated after EOS. None (default) = the library's per-text estimate.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic int8 quantization to the transformer (upstream: ~48%% less memory, ~27%% "
             "faster on x86, WER unchanged). Off by default — it is a CPU (FBGEMM) optimization.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="",
        help="Predefined voice for --no-voice_clone (26 available: alba, anna, ..., vera). Empty "
             "(default) picks the language's default: giovanni/it, lola/es, juergen/de, rafael/pt, "
             "estelle/fr, alba otherwise.",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
