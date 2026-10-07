"""
TTS synthesis for the Open TTS Leaderboard (Chatterbox backend, stage 1).

Chatterbox (ResembleAI/chatterbox) is a ~0.5B zero-shot TTS loaded in-process via the
`chatterbox-tts` package, synthesized one text at a time (no batched API). Two checkpoints,
selected with `--multilingual`:

  * `ChatterboxTTS` (default) — the English-only checkpoint; `generate()` takes no language, so
    use it for `en` only.
  * `ChatterboxMultilingualTTS` (`--multilingual`) — the 23-language checkpoint (needs
    `language_id`); outputs get a `_multilingual` suffix since it shares the model id. This also
    covers Resemble's single-language finetunes (see `_load_single_language_pack`).

Voice cloning (`--voice_clone`) conditions each sample on the per-sample `prompt_audio` via
`audio_prompt_path`; outputs get a `_voice_clone` suffix. The default (`--no-voice_clone`) uses the
built-in default voice and skips SIM.
"""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# The multilingual class is imported lazily in `_load_model`.
from chatterbox.tts import ChatterboxTTS

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# The repo whose weights upstream's ChatterboxMultilingualTTS.from_pretrained() hardcodes.
CHATTERBOX_BASE_REPO = "ResembleAI/chatterbox"


# T3 filenames the multilingual from_local() may hardcode, depending on the chatterbox-tts version
# (0.1.7 loads v2; newer builds default to v3). A single-language pack's T3 is staged under both.
_MULTILINGUAL_T3_NAMES = ("t3_mtl23ls_v2.safetensors", "t3_mtl23ls_v3.safetensors")


def _accepts(func, name):
    """Whether `func` takes a keyword argument `name` (chatterbox-tts's API varies by version)."""
    import inspect

    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _load_single_language_pack(model_id, device, t3_model_file=None, s3gen_source="pack"):
    """Load a Resemble 'single language pack' finetune (e.g. Chatterbox-Multilingual-zh-cmn).

    `from_pretrained()` hardcodes the base repo, so the pack is snapshotted and loaded with
    `from_local()`, which expects fixed filenames. A staging dir of symlinks (never mutating the HF
    cache) presents:
      * the pack's T3 (e.g. `t3_zh_cmn.safetensors`) under the multilingual T3 names;
      * a decoder as `s3gen.pt`: the pack's own, or the base repo's (the decoder is
        language-agnostic; 0.1.7 cannot load the V3 packs' `s3gen_v3.pt`). `s3gen_source="auto"`
        tries the pack's and falls back to the base repo's on a load error.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    ckpt_dir = Path(snapshot_download(model_id))

    if t3_model_file is None:
        # The pack contains exactly one T3 checkpoint, named after its language.
        candidates = sorted(
            p.name for p in ckpt_dir.glob("t3_*.safetensors") if p.name not in _MULTILINGUAL_T3_NAMES
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one language-specific t3_*.safetensors in {model_id}, found "
                f"{candidates}. Pass --t3_model_file explicitly."
            )
        t3_model_file = candidates[0]

    staged = Path("/tmp") / f"chatterbox_pack_{model_id.replace('/', '-')}"
    staged.mkdir(parents=True, exist_ok=True)

    def link(src, name):
        dst = staged / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src, dst)

    for src in ckpt_dir.iterdir():
        if not src.is_dir():
            link(src, src.name)
    # Present the pack's T3 under every name the loader might hardcode.
    for name in _MULTILINGUAL_T3_NAMES:
        link(ckpt_dir / t3_model_file, name)

    kwargs = {}
    if _accepts(ChatterboxMultilingualTTS.from_local, "t3_model"):
        kwargs["t3_model"] = t3_model_file   # newer API: name it explicitly

    def attempt(which):
        """Point staged/s3gen.pt at `which` decoder, then load."""
        if which == "pack":
            src = next((ckpt_dir / n for n in ("s3gen.pt", "s3gen_v3.pt") if (ckpt_dir / n).exists()), None)
            if src is None:
                return None  # pack ships no decoder; nothing to try
        else:
            src = Path(hf_hub_download(CHATTERBOX_BASE_REPO, "s3gen.pt"))
        link(src, "s3gen.pt")
        print(f"Loading single-language pack {model_id} via from_local "
              f"(t3={t3_model_file}, s3gen={which}:{src.name}, kwargs={list(kwargs)})")
        return ChatterboxMultilingualTTS.from_local(staged, device, **kwargs)

    if s3gen_source in ("pack", "base"):
        model = attempt(s3gen_source)
        if model is None:
            raise RuntimeError(f"{model_id} ships no s3gen checkpoint; use --pack_s3gen=base.")
        return model

    # auto: prefer the pack's own decoder, fall back to the base repo's.
    try:
        model = attempt("pack")
        if model is not None:
            return model
    except RuntimeError as e:
        print(f"WARNING: the pack's s3gen did not load ({e.__class__.__name__}: "
              f"{str(e).splitlines()[0]}); retrying with {CHATTERBOX_BASE_REPO}'s s3gen.pt.")
    return attempt("base")


def _load_model(args):
    """Load the checkpoint selected by --multilingual / --model_id."""
    if not args.multilingual:
        # English-only checkpoint; generate() has no language argument at all.
        return ChatterboxTTS.from_pretrained(device=args.device)

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    if args.model_id == CHATTERBOX_BASE_REPO:
        if _accepts(ChatterboxMultilingualTTS.from_pretrained, "t3_model"):
            return ChatterboxMultilingualTTS.from_pretrained(device=args.device, t3_model=args.t3_model)
        # Released 0.1.7 has no t3_model argument and always fetches the v2 multilingual T3.
        print(
            f"WARNING: installed chatterbox-tts has no t3_model argument — ignoring "
            f"--t3_model={args.t3_model} and loading the v2 multilingual checkpoint. Rebuild the "
            f"image against a newer chatterbox-tts for v3."
        )
        return ChatterboxMultilingualTTS.from_pretrained(device=args.device)
    return _load_single_language_pack(
        args.model_id, args.device, t3_model_file=args.t3_model_file, s3gen_source=args.pack_s3gen
    )


# ── chatterbox-tts attention-hook leak workaround ────────────────────────────────────────────────
# Each multilingual `generate()` builds a fresh `AlignmentStreamAnalyzer` that registers forward
# hooks on three `self_attn` modules and never removes them, so per-step cost grows with every
# sample generated (and RTFx would measure the leak). Dropping the stale hooks before each call
# keeps throughput flat; a no-op for the English-only checkpoint.
_ALIGNMENT_HOOK_NAME = "attention_forward_hook"


def _prune_alignment_hooks(model):
    """Remove AlignmentStreamAnalyzer hooks left behind by earlier generate() calls; return the count.

    Matches hooks by function name so unrelated hooks on the same modules are left alone, and keeps
    torch's parallel per-hook flag dicts in sync with `_forward_hooks`.
    """
    layers = getattr(getattr(getattr(model, "t3", None), "tfmr", None), "layers", None)
    if layers is None:
        return 0
    removed = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        hooks = getattr(attn, "_forward_hooks", None)
        if not hooks:
            continue
        stale = [k for k, fn in hooks.items() if getattr(fn, "__name__", "") == _ALIGNMENT_HOOK_NAME]
        for k in stale:
            del hooks[k]
            for flags in ("_forward_hooks_with_kwargs", "_forward_hooks_always_called"):
                getattr(attn, flags, {}).pop(k, None)
        removed += len(stale)
    return removed


def main(args):
    # Set seed for reproducibility (Chatterbox's T3 backbone samples autoregressively).
    set_seed(args.seed)

    model = _load_model(args)
    sampling_rate = model.sr  # 24000
    variant = f"multilingual, language_id={args.language}" if args.multilingual else "english-only"
    print(f"Loaded Chatterbox ({variant}, sr={sampling_rate})")
    print_model_size(model)

    is_cuda = torch.device(args.device).type == "cuda"
    pruning_logged = []  # one-shot flag so the log states the workaround engaged, without spamming

    def synth_one(text, prompt_path):
        """Synthesize one text (optionally cloning prompt_path); return (audio_1d_numpy, elapsed_seconds)."""
        # Drop the previous call's leaked alignment hooks before the timer starts.
        pruned = _prune_alignment_hooks(model)
        if pruned and not pruning_logged:
            pruning_logged.append(pruned)
            print(f"Dropping {pruned} leaked alignment hooks per sample (chatterbox-tts hook leak).")
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
        start = time.perf_counter()
        gen_kwargs = dict(
            audio_prompt_path=prompt_path,  # None → built-in default voice; else clone this clip
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
        )
        # Only the multilingual checkpoint takes a language; ChatterboxTTS.generate() would reject it.
        if args.multilingual:
            gen_kwargs["language_id"] = args.language
        wav = model.generate(text, **gen_kwargs)
        if is_cuda:
            torch.cuda.synchronize(device=args.device)
        elapsed = time.perf_counter() - start
        # generate() returns a (1, N) float tensor.
        return wav.squeeze().detach().to(torch.float32).cpu().numpy(), elapsed

    # Layout: results/<model_safe>/  MODEL_<safe>_DATASET_<dsid>.jsonl  +  <dsid>/output_<id>.wav
    # where <dsid> = <dataset_safe>_<dataset>_<split><mode_suffix>. Manifest paths are relative to
    # model_dir so later stages resolve them wherever the folder is copied (local or HF bucket).
    # `_multilingual` separates the multilingual checkpoint, which shares the English model id.
    mode_suffix = ("_multilingual" if args.multilingual else "") + ("_voice_clone" if args.voice_clone else "")
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    # Reference clips must live under results/ (not tempfiles) so stage 3 SIM can read them.
    prompt_dir = os.path.join(output_dir, "prompts")
    if args.voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    # Keep `id`, target text, and (when cloning) the per-sample reference.
    dataset = load_tts_dataset(args, ("prompt_text", "prompt_audio") if args.voice_clone else ())

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Same voice handling as the main loop; references go to a temp dir. No streaming API, so
    # `first` is None -> whole-utterance clock.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="ttfa_refs_")

        def _ttfa_ref(i):
            if not args.voice_clone:
                return None
            path = os.path.join(tmp_dir, f"prompt_{dataset[i]['id']}.wav")
            pa = dataset[i]["prompt_audio"]
            sf.write(path, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
            return path

        jobs = [{"text": dataset[i][args.text_column], "prompt_path": _ttfa_ref(i)}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            lambda job: (synth_one(job["text"], job["prompt_path"])[0], sampling_rate, None),
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
        """When cloning, persist a row's reference clip (stage 3 SIM reads it); (rel_path, path) or Nones."""
        if not args.voice_clone:
            return None, None
        prompt_rel_path = os.path.join(dataset_dir_name, "prompts", f"prompt_{row['id']}.wav")
        prompt_path = os.path.join(model_dir, prompt_rel_path)
        if not os.path.exists(prompt_path):
            pa = row["prompt_audio"]
            sf.write(prompt_path, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
        return prompt_rel_path, prompt_path

    # Serial generation: batches of one sample.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda b: synth_one(b[args.text_column][0], prompt_for({c: b[c][0] for c in b})[1]),
            dataset, starts, 1, args.warmup_steps)

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        text = row[args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, row["id"])
        path = os.path.join(model_dir, rel_path)

        # When cloning, condition on the persisted reference clip.
        prompt_rel_path, prompt_path = prompt_for(row)

        audio, elapsed = synth_one(text, prompt_path)
        sf.write(path, audio, sampling_rate)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / sampling_rate, elapsed, text, prompt_rel_path))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="ResembleAI/chatterbox", help="Model id (for manifest naming).")
    add_common_args(parser, voice_clone=False)
    parser.add_argument(
        "--multilingual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load ChatterboxMultilingualTTS (23 languages, needs --language) instead of the "
             "English-only ChatterboxTTS. Required for any non-English split: the English-only "
             "checkpoint has no language argument and renders other languages as gibberish.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code passed as `language_id` to the multilingual checkpoint (e.g. 'en', "
             "'zh'). Upstream validates it against its 23 SUPPORTED_LANGUAGES. Ignored when "
             "--no-multilingual.",
    )
    parser.add_argument(
        "--t3_model",
        type=str,
        default="v3",
        help="Multilingual T3 checkpoint for the base repo: 'v3' (default) or 'v2'.",
    )
    parser.add_argument(
        "--t3_model_file",
        type=str,
        default=None,
        help="For single-language packs only: the t3_*.safetensors filename to load. Auto-detected "
             "when the pack contains exactly one.",
    )
    parser.add_argument(
        "--pack_s3gen",
        type=str,
        choices=["auto", "pack", "base"],
        default="auto",
        help="For single-language packs only: whose speech decoder to use. 'auto' (default) tries "
             "the pack's own s3gen and falls back to ResembleAI/chatterbox's s3gen.pt if the "
             "installed chatterbox-tts can't load it. 'pack' or 'base' force one and fail loudly.",
    )
    # Generation params (defaults match the model's generate() defaults).
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--cfg_weight", type=float, default=0.5, help="Classifier-free guidance weight.")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration.")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
