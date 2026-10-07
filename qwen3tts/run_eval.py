"""
TTS synthesis for the Open TTS Leaderboard (Qwen3-TTS backend, stage 1).

Runs in its own image because `qwen-tts` pins an older `transformers` than the Qwen3-ASR scorer.
Two variants (auto-detected from --model_id, overridable with --mode), both natively batched:
    * CustomVoice: built-in speaker via `generate_custom_voice` (--speaker broadcast over the batch).
    * Base: zero-shot cloning via `generate_voice_clone` from each sample's `prompt_audio`/`prompt_text`;
      reference clips are saved for SIM and outputs get a `_voice_clone` suffix.
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from qwen_tts import Qwen3TTSModel

from run_eval_utils import (
    add_common_args,
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    open_manifest,
    output_paths,
    pending_batches,
    print_model_size,
    print_next_steps,
    set_seed,
    warm_up,
    write_entry,
)

torch.set_float32_matmul_precision("high")

# Qwen3-TTS takes the English language NAME; map the ISO codes the eval scripts pass. Unknown
# values pass through so 'English'/'Chinese'/'Auto' still work.
LANGUAGE_CODE_TO_NAME = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def main(args):
    # Set seed for reproducibility (some models sample internally).
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Resolve variant: Base models do zero-shot voice cloning from the dataset's
    # prompt_audio/prompt_text; CustomVoice models use a built-in speaker.
    if args.mode == "auto":
        mode = "voice_clone" if "base" in args.model_id.split("/")[-1].lower() else "custom_voice"
    else:
        mode = args.mode

    torch_dtype = getattr(torch, args.dtype)
    model = Qwen3TTSModel.from_pretrained(
        args.model_id,
        device_map=args.device,
        dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    if mode == "voice_clone":
        print(f"Loaded Qwen3-TTS model {args.model_id} (mode=voice_clone, language={args.language})")
    else:
        print(f"Loaded Qwen3-TTS model {args.model_id} (mode=custom_voice, speaker={args.speaker}, language={args.language})")

    # Report parameter count (Qwen3TTSModel is a wrapper: also sum its nn.Module attributes).
    try:
        print_model_size(model)
    except Exception as e:
        print(f"Could not determine model size: {e}")

    # Layout: results/<model_safe>/MODEL_<safe>_DATASET_<dsid>.jsonl + <dsid>/output_<id>.wav.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    mode_suffix = "_voice_clone" if mode == "voice_clone" else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir

    # Voice-clone reference clips are saved under results/ for stage 3 (SIM).
    prompt_dir = os.path.join(output_dir, "prompts")
    if mode == "voice_clone":
        os.makedirs(prompt_dir, exist_ok=True)

    def generate_tts(batch):
        """Synthesize speech for a minibatch of target texts; time the batch generation for RTFx."""
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)

        # Voice cloning (Base): per-sample reference audio + transcript from the dataset.
        # qwen-tts accepts ref_audio as a list of (numpy_array, sampling_rate) tuples.
        prompt_paths = [None] * minibatch_size
        if mode == "voice_clone":
            ref_audio = [
                (np.asarray(a["array"], dtype=np.float32), a["sampling_rate"])
                for a in batch["prompt_audio"]
            ]
            ref_text = list(batch["prompt_text"])
            prompt_paths = []
            for sample_id, a in zip(batch["id"], batch["prompt_audio"]):
                prompt_rel_path = os.path.join(dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
                ppath = os.path.join(model_dir, prompt_rel_path)
                if not os.path.exists(ppath):
                    sf.write(ppath, np.asarray(a["array"], dtype=np.float32), a["sampling_rate"])
                prompt_paths.append(prompt_rel_path)

        # START TIMING (TTS batch generation)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # qwen-tts accepts lists for all per-sample args; broadcast the single language.
        if mode == "voice_clone":
            wavs, sr = model.generate_voice_clone(
                text=texts_to_generate,
                language=[args.language] * minibatch_size,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        else:
            wavs, sr = model.generate_custom_voice(
                text=texts_to_generate,
                language=[args.language] * minibatch_size,
                speaker=[args.speaker] * minibatch_size,
            )

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(wavs, batch["id"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = os.path.join(dataset_dir_name, f"output_{sample_id}.wav")
            audio = np.asarray(audio).reshape(-1)
            sf.write(os.path.join(model_dir, rel_path), audio, sr)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sr)

        batch["gen_audio_filepath"] = gen_paths
        batch["prompt_audio_filepath"] = prompt_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    dataset = load_tts_dataset(args, extra_columns=("prompt_text", "prompt_audio") if mode == "voice_clone" else ())

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: writes ONLY a JSON sidecar (no wavs, no manifest) ─────────
    # TTFA is per-request, so generate_tts() is driven with one-row batches. No streaming API, so
    # TTFA = whole-utterance time. `model_dir`/`dataset_dir_name` are rebound to a temp dir:
    # generate_tts closes over them (late binding), so its writes never touch the results tree.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported lazily: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        def _gen(job):
            result = generate_tts(job)
            duration = result["audio_length_s"][0]
            return float(duration), None, None

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
            extra={"device": args.device, "note": "driven at batch size 1"},
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

        # Append each sample to the JSONL immediately. `pred_text` is filled by stage 2 (ASR);
        # `sim` by stage 3 (speaker similarity, voice-clone only).
        for afp, ppath, alen, gtime, ref in zip(
            batch["gen_audio_filepath"], batch["prompt_audio_filepath"], batch["audio_length_s"],
            batch["generation_time_s"], batch["references"],
        ):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref, ppath if mode == "voice_clone" else None))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(mode == "voice_clone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser, batch_size=32)

    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Qwen3-TTS model id, loadable with the `qwen-tts` package (CustomVoice or Base variant).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "custom_voice", "voice_clone"],
        help=(
            "Generation mode. 'custom_voice' uses a built-in --speaker (CustomVoice models); "
            "'voice_clone' clones the dataset's prompt_audio/prompt_text per sample (Base models); "
            "'auto' picks voice_clone when the model id contains 'Base'."
        ),
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ryan",
        help="Built-in speaker voice (e.g. Ryan, Eric, Aiden, Dylan, Vivian, Serena). Ignored in voice_clone mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Target language: an English name ('English', 'Chinese', ...), an ISO code ('en', 'zh', "
             "...; mapped to the name), or 'Auto'.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype, e.g. 'bfloat16'.")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", help="Attention impl ('flash_attention_2'/'sdpa').")

    args = parser.parse_args()

    args.language = LANGUAGE_CODE_TO_NAME.get(args.language.lower(), args.language)

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print(f"Target language: {args.language}")
    print("*" * 100)

    main(args)
