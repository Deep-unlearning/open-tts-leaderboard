"""
TTS synthesis for the Open TTS Leaderboard (Fish Audio S2-Pro backend, stage 1).

S2-Pro (fishaudio/s2-pro) is a ~5B dual-AR TTS run in-process from the cloned `fish-speech` repo,
one sample at a time (no batched API). Output is 44.1 kHz mono.

With --voice_clone each sample's reference clip is encoded to VQ tokens and passed with its
transcript; the reference wav is saved for SIM and outputs get a `_voice_clone` suffix. By default
(--no-voice_clone) the default (seed-dependent) voice is used.
"""

import argparse
import os
import sys
import time
from functools import partial

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from tqdm import tqdm

from run_eval_utils import (
    add_common_args,
    count_parameters,
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    open_manifest,
    output_paths,
    pending_batches,
    print_next_steps,
    set_seed,
    warm_up,
    wav_rel_path,
    write_entry,
)

import fish_speech.models.text2semantic.inference as fs_inference  # noqa: E402
from fish_speech.conversation import Conversation  # noqa: E402
from fish_speech.models.text2semantic.inference import (  # noqa: E402
    init_model,
    generate_long,
    load_codec_model,
    decode_to_audio,
    encode_audio,
)

# Silence fish-speech's per-sample loguru INFO logs, per-token tqdm bar and prompt dump.
logger.remove()
logger.add(sys.stderr, level="WARNING")
fs_inference.tqdm = partial(tqdm, disable=True)
Conversation.visualize = lambda self, *args, **kwargs: None


def main(args):
    seed = args.seed
    set_seed(seed)

    device = args.device
    precision = getattr(torch, args.dtype)

    # Load text2semantic model (returns model + its single-token decode fn) and set up caches.
    model, decode_one_token = init_model(args.checkpoint_path, device, precision, compile=False)
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    codec = load_codec_model(os.path.join(args.checkpoint_path, "codec.pth"), device, precision)
    sampling_rate = int(codec.sample_rate)  # 44100
    print(f"Loaded S2-Pro ({args.model_id}, sr={sampling_rate})")
    n_params = count_parameters(model, codec)
    if n_params:
        print(f"TTS model size: {n_params / 1e9:.2f}B parameters (text2semantic + codec)")

    is_cuda = str(device).startswith("cuda")

    def synth_one_streaming(text, prompt_tokens=None, prompt_text=None, early_stop=False):
        """Like synth_one, but timestamps the first audio: (audio_1d_numpy, first_audio_ts).

        TTFA probe only. The first generate_long() segment is decoded as soon as it arrives (that
        decode counts toward TTFA). Segments split only on `<|speaker:N|>` tags, so eval texts are
        a single segment and `first` lands at end of generation; chunk-level streaming TTFA is
        measured via sglang-omni instead (the s2pro-sglang target in submit_ttfa_jobs.sh).
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        first, codes, pieces = None, [], []
        for resp in generate_long(
            model=model, device=device, decode_one_token=decode_one_token, text=text,
            num_samples=1, max_new_tokens=args.max_new_tokens, top_p=args.top_p,
            top_k=args.top_k, temperature=args.temperature, compile=False,
            prompt_text=prompt_text, prompt_tokens=prompt_tokens,
        ):
            if resp.action == "sample":
                codes.append(resp.codes)
                if first is None:
                    seg = decode_to_audio(resp.codes.to(device), codec)
                    pieces.append(np.asarray(seg.detach().to(torch.float32).cpu()).reshape(-1))
                    first = time.perf_counter()
                    if early_stop:
                        break
            elif resp.action == "next":
                break
        if not codes:
            return np.zeros(1, dtype=np.float32), None
        if len(codes) > 1:
            rest = decode_to_audio(torch.cat(codes[1:], dim=1).to(device), codec)
            pieces.append(np.asarray(rest.detach().to(torch.float32).cpu()).reshape(-1))
        return np.concatenate(pieces), first

    def synth_one(text, prompt_tokens=None, prompt_text=None):
        """Synthesize one text; return (audio_1d_numpy, elapsed_s).

        When prompt_tokens + prompt_text are given the generation is conditioned on that
        reference clip (voice cloning); otherwise the default (seed-dependent) voice is used.
        """
        # Re-seed per sample so the (seed-dependent) default voice is consistent + reproducible.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()

        gen = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            compile=False,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
        )
        codes = []
        for resp in gen:
            if resp.action == "sample":
                codes.append(resp.codes)
            elif resp.action == "next":
                break

        if codes:
            merged = torch.cat(codes, dim=1)  # (num_codebooks, T)
            audio = decode_to_audio(merged.to(device), codec)
            audio = np.asarray(audio.detach().to(torch.float32).cpu()).reshape(-1)
        else:
            audio = np.zeros(int(0.1 * sampling_rate), dtype=np.float32)

        if is_cuda:
            torch.cuda.synchronize()
        return audio, time.perf_counter() - start

    # Layout: results/<model_safe>/MODEL_<safe>_DATASET_<dsid>.jsonl + <dsid>/output_<id>.wav.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    # Persist reference clips under results/ so stage 3 (SIM) can read them from disk.
    prompt_dir = os.path.join(output_dir, "prompts")
    if args.voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    dataset = load_tts_dataset(args, extra_columns=("prompt_text", "prompt_audio") if args.voice_clone else ())

    # ── TTFA probe: writes ONLY a JSON sidecar (no wavs, no manifest) ─────────
    # Same voice handling as the eval; references go to a temp dir so results are untouched.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported lazily: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="ttfa_refs_")

        def _ttfa_ref(i):
            if not args.voice_clone:
                return None, None
            path = os.path.join(tmp_dir, f"prompt_{dataset[i]['id']}.wav")
            pa = dataset[i]["prompt_audio"]
            sf.write(path, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
            return [encode_audio(path, codec, device).cpu()], [dataset[i]["prompt_text"]]

        jobs = []
        for i in sample_indices(args.ttfa_probe, len(dataset)):
            job = {"text": dataset[i][args.text_column]}
            job["prompt_tokens"], job["prompt_text"] = _ttfa_ref(i)
            jobs.append(job)

        def _ttfa_gen(job, early_stop):
            audio, first = synth_one_streaming(
                job["text"], prompt_tokens=job["prompt_tokens"], prompt_text=job["prompt_text"],
                early_stop=early_stop,
            )
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

    def synth_row(row):
        """Synthesize one dataset row; return (audio, elapsed, prompt_rel_path)."""
        prompt_rel_path = prompt_tokens = prompt_text = None
        if args.voice_clone:
            # Save the reference clip (for SIM), then encode it to VQ tokens.
            pa = row["prompt_audio"]
            prompt_rel_path = os.path.join(dataset_dir_name, "prompts", f"prompt_{row['id']}.wav")
            prompt_path = os.path.join(model_dir, prompt_rel_path)
            if not os.path.exists(prompt_path):
                sf.write(prompt_path, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
            # generate_long expects lists (one entry per reference segment); a bare tensor fails
            # its `bool(prompt_tokens)` check.
            prompt_text = [row["prompt_text"]]
            prompt_tokens = [encode_audio(prompt_path, codec, device).cpu()]
        audio, elapsed = synth_one(row[args.text_column], prompt_tokens=prompt_tokens, prompt_text=prompt_text)
        return audio, elapsed, prompt_rel_path

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda b: synth_row({k: v[0] for k, v in b.items()}), dataset, starts, 1, args.warmup_steps)
    for i in tqdm(starts, desc="Generating"):
        row = dataset[i]
        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, row["id"])
        audio, elapsed, prompt_rel_path = synth_row(row)
        sf.write(os.path.join(model_dir, rel_path), audio, sampling_rate)

        # `pred_text` is filled by stage 2 (ASR).
        write_entry(manifest_file, manifest_entry(
            rel_path, len(audio) / sampling_rate, elapsed, row[args.text_column], prompt_rel_path
        ))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser, voice_clone=False)

    parser.add_argument("--model_id", type=str, default="fishaudio/s2-pro", help="Model id (for manifest naming).")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/opt/fish-speech/checkpoints/s2-pro",
        help="Local dir with the S2-Pro safetensors + codec.pth (downloaded at image build).",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model precision, e.g. 'bfloat16'.")
    parser.add_argument("--max_new_tokens", type=int, default=0, help="Max semantic tokens (0 = unbounded to context).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
