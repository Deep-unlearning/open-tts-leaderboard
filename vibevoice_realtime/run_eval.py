"""
TTS synthesis for the Open TTS Leaderboard (VibeVoice-Realtime backend, stage 1).

The streaming API asserts batch_size == 1, so generation is a per-sample loop inside
`generate_tts`; `--batch_size` only chunks the dataset for manifest writing / resume. Each chunk
is timed with CUDA events and per-sample time = chunk time / chunk size.
"""

import argparse
import copy
import os
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm


from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
# Queue-backed streamer for observing first audio during generate(); TTFA probe only.
from vibevoice.modular.streamer import AudioStreamer

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

torch.set_float32_matmul_precision("high")

# Voice presets = cached prefill states (.pt) for the streaming model, hosted on the HF Hub.
VOICE_REPO = "bezzam/vibevoice_samples"
VOICE_DIR = "realtime_model/voices/streaming_model"

BASE_VOICES = [
    "en-Carter_man", "en-Davis_man", "en-Emma_woman", "en-Frank_man", "en-Grace_woman", "en-Mike_man",
    "in-Samuel_man",
    "de-Spk0_man", "de-Spk1_woman",
    "fr-Spk0_man", "fr-Spk1_woman",
    "it-Spk0_woman", "it-Spk1_man",
    "jp-Spk0_man", "jp-Spk1_woman",
    "kr-Spk0_woman", "kr-Spk1_man",
    "nl-Spk0_man", "nl-Spk1_woman",
    "pl-Spk0_man", "pl-Spk1_woman",
    "pt-Spk0_woman", "pt-Spk1_man",
    "sp-Spk0_woman", "sp-Spk1_man",
]

EXPERIMENTAL_VOICES = {
    "de": ["de-Spk2_woman", "de-Spk3_man", "de-Spk4_woman", "de-Spk5_man", "de-Spk6_man"],
    "fr": ["fr-Spk2_man", "fr-Spk3_woman", "fr-Spk4_woman", "fr-Spk5_man"],
    "jp": ["jp-Spk2_woman", "jp-Spk3_woman", "jp-Spk4_woman", "jp-Spk5_man"],
    "kr": ["kr-Spk2_woman", "kr-Spk3_man"],
    "pl": ["pl-Spk2_man", "pl-Spk3_woman"],
    "pt": ["pt-Spk2_woman", "pt-Spk3_man", "pt-Spk4_man", "pt-Spk5_woman"],
    "sp": ["sp-Spk2_woman", "sp-Spk3_man", "sp-Spk4_woman", "sp-Spk5_man"],
}

AVAILABLE_VOICES = BASE_VOICES + [v for voices in EXPERIMENTAL_VOICES.values() for v in voices]
SAMPLE_RATE = 24000


def voice_repo_path(speaker):
    """Path of `speaker`'s cached prefill states inside VOICE_REPO (experimental ones are nested per language)."""
    if speaker in BASE_VOICES:
        return f"{VOICE_DIR}/{speaker}.pt"
    return f"{VOICE_DIR}/experimental_voices/{speaker.split('-', 1)[0]}/{speaker}.pt"


def main(args):
    # Set seed for reproducibility (the diffusion head samples internally).
    seed = args.seed
    set_seed(seed)
    torch.backends.cudnn.deterministic = True

    torch_dtype = getattr(torch, args.dtype)
    processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_id)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)
    print(
        f"Loaded VibeVoice-Realtime model {args.model_id} "
        f"(speaker={args.speaker}, cfg_scale={args.cfg_scale}, ddpm_steps={args.ddpm_steps})"
    )

    # Report parameter count (dedup by id in case submodules share parameters).
    print_model_size(model)

    # ── Fixed voice preset: cached prefill states from the HF Hub ────────────
    if args.speaker not in AVAILABLE_VOICES:
        raise ValueError(f"Unknown speaker '{args.speaker}'. Choose from: {', '.join(AVAILABLE_VOICES)}")
    voice_path = hf_hub_download(
        repo_id=VOICE_REPO,
        filename=voice_repo_path(args.speaker),
        repo_type="dataset",
    )
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        all_prefilled_outputs = torch.load(voice_path, map_location=args.device, weights_only=True)
    print(f"Using voice preset '{args.speaker}': {voice_path}")

    # Layout: results/<model_safe>/MODEL_<safe>_DATASET_<dsid>.jsonl + <dsid>/output_<id>.wav.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    def prepare_inputs(text):
        """Seed and build one generate() input dict (shared by the eval and the TTFA probe)."""
        # Re-seed per sample so diffusion sampling is reproducible regardless of chunking.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        clean_text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        inputs = processor.process_input_with_cached_prompt(
            text=clean_text,
            cached_prompt=all_prefilled_outputs,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(args.device)
        return inputs

    def synth_one(text):
        """Synthesize one utterance, returning the finished waveform."""
        inputs = prepare_inputs(text)

        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=args.cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,  # keep the outer "Generating" tqdm as the only progress output
            # generate() consumes/mutates the cached prompt states, so give it a copy.
            all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
        )

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            audio = np.asarray(audio.detach().to(torch.float32).cpu()).reshape(-1)
        else:
            print(f"Warning: no audio generated for text: {text[:60]}...")
            audio = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
        return audio

    def generate_tts(batch):
        """Synthesize a chunk of target texts (per-sample loop); time the chunk for RTFx."""
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)

        # START TIMING (TTS chunk generation)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # No batched API (generate() asserts batch_size == 1): loop per sample.
        wavs = [synth_one(text) for text in texts_to_generate]

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(wavs, batch["id"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            path = os.path.join(model_dir, rel_path)
            sf.write(path, audio, SAMPLE_RATE)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / SAMPLE_RATE)

        batch["gen_audio_filepath"] = gen_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    if args.ttfa_probe != 0:
        # Imported lazily: ttfa_probe.py is only injected by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import threading

        chunk_counts = []  # chunks per utterance

        def _gen(job, early_stop):
            """One generation with an audio_streamer attached, timestamping the first chunk."""
            inputs = prepare_inputs(job["text"])
            streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            stop_event = threading.Event()
            errors = []

            def _run():
                try:
                    model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=args.cfg_scale,
                        tokenizer=processor.tokenizer,
                        generation_config={"do_sample": False},
                        audio_streamer=streamer,
                        # Honoured between diffusion steps, so an early-stopped row aborts the
                        # generation instead of leaving it running for the rest of the probe.
                        stop_check_fn=stop_event.is_set,
                        verbose=False,
                        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
                    )
                except Exception as exc:  # surfaced on the consuming thread below
                    errors.append(exc)
                finally:
                    # Ensure get_stream(0) never blocks forever if generate() crashes.
                    streamer.end()

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            first, chunks = None, []
            try:
                for chunk in streamer.get_stream(0):
                    if first is None:
                        first = time.perf_counter()
                    chunk = np.asarray(
                        chunk.detach().to(torch.float32).cpu() if torch.is_tensor(chunk) else chunk,
                        dtype=np.float32,
                    ).reshape(-1)
                    chunks.append(chunk)
                    if early_stop:
                        break
            finally:
                stop_event.set()
                streamer.end()
                thread.join()
            if errors:
                raise errors[0]

            chunk_counts.append(len(chunks))
            audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            return audio, SAMPLE_RATE, first

        run_probe(
            _gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            # chunks_per_utterance distinguishes "no incremental output" (1 chunk) from late delivery.
            extra={"device": args.device, "note": "driven at batch size 1, audio_streamer attached",
                   "chunks_per_utterance": chunk_counts},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    # Generation is per-sample, so warm up on one sample per step (the first of each pending chunk).
    warm_up(generate_tts, dataset, starts, 1, args.warmup_steps)

    # ── Main loop: TTS → write JSONL, one chunk at a time ───────────────────
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
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="VibeVoice-Realtime model id, loadable with the `vibevoice` package.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="en-Carter_man",
        choices=AVAILABLE_VOICES,
        help="Fixed voice preset (cached prefill states from bezzam/vibevoice_samples). Must match "
        "the dataset's language: the en-* presets read non-English text with an English voice, which "
        "makes the WER/CER meaningless. Non-en presets are upstream's EXPERIMENTAL voices.",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="Classifier-free guidance scale.")
    parser.add_argument("--ddpm_steps", type=int, default=5, help="DDPM inference steps for the diffusion head.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype, e.g. 'bfloat16'.")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", help="Attention impl ('sdpa'/'eager').")
    add_common_args(parser)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Chunk size for manifest writing/resume ONLY — generation itself is per-sample "
        "(the streaming API asserts batch_size == 1).",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
