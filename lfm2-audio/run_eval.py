"""
TTS synthesis for the Open TTS Leaderboard (LFM2.5-Audio backend, stage 1).

LFM2.5-Audio-1.5B (LiquidAI/LFM2.5-Audio-1.5B) is Liquid AI's end-to-end speech + text model: an
LFM2.5 backbone whose RQ-transformer ("depthformer") emits 8 Mimi-compatible codebooks per 12.5 Hz
frame, turned into 24 kHz audio by an LFM-based detokenizer (`processor.decode`). It is loaded
in-process via the `liquid-audio` package. TTS uses the package's *sequential* generation with one
of four fixed voices, picked by the system prompt (see VOICES); there is no voice cloning, and the
model is English only.

`generate_sequential` asserts a batch of one (`_prefill`: text.shape[0] == 1), so rows are
processed serially. Per row we generate audio frames until the end-of-audio frame (all codes
2048), detokenize them in one call, and time both together for RTFx.

Upstream TTS example: https://github.com/Liquid4All/liquid-audio#tts

Stage 2 (`transformers/transcribe.py`, run in the shared `bezzam/evals` image) transcribes the
wavs with Qwen3-ASR to fill `pred_text`, and stage 4 (`normalizer.eval_utils.score_results`, run
locally) computes WER + RTFx from the completed manifest. Fixed voice, so no stage 3 (SIM).
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

from run_eval_utils import (
    add_common_args, load_done_entries, load_tts_dataset, manifest_entry, open_manifest, output_paths,
    pending_batches, print_model_size, print_next_steps, set_seed, warm_up, wav_rel_path, write_entry,
)

# The detokenizer outputs 24 kHz mono (fixed constant, not returned by the API).
SAMPLING_RATE = 24_000
# Audio code that marks the end-of-audio frame (the depthformer's extra vocab entry).
END_OF_AUDIO = 2048

# The four built-in voices, selected by the system prompt (from the liquid-audio README).
VOICES = {
    "us_female": "Perform TTS. Use the US female voice.",
    "us_male": "Perform TTS. Use the US male voice.",
    "uk_female": "Perform TTS. Use the UK female voice.",
    "uk_male": "Perform TTS. Use the UK male voice.",
}


def main(args):
    set_seed(args.seed)

    # The package's detokenizer is hard-coded to `.cuda()` (LFM2AudioProcessor.audio_detokenizer).
    if not args.device.startswith("cuda"):
        raise ValueError(f"liquid-audio's detokenizer only runs on CUDA; got --device={args.device}.")

    processor = LFM2AudioProcessor.from_pretrained(args.model_id, device=args.device).eval()
    model = LFM2AudioModel.from_pretrained(args.model_id, device=args.device).eval()
    # The detokenizer is loaded lazily on first decode; load it now so it is counted and its load
    # never lands in a timed generation.
    detokenizer = processor.audio_detokenizer
    print(f"Loaded LFM2.5-Audio ({args.model_id}, voice={args.voice})")
    print_model_size(model, detokenizer)

    system_prompt = VOICES[args.voice]
    truncated = []  # rows that hit --max_new_tokens before the end-of-audio frame

    def audio_frames(text):
        """Yield the (8,) audio code frames for `text`, stopping at the end-of-audio frame."""
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(system_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text(text)
        chat.end_turn()
        chat.new_turn("assistant")

        for t in model.generate_sequential(
            **chat,
            max_new_tokens=args.max_new_tokens,
            audio_temperature=args.audio_temperature,
            audio_top_k=args.audio_top_k,
        ):
            if t.numel() == 1:  # text token (<|audio_start|> etc.)
                continue
            # The audio is complete at end-of-audio; what follows is only the closing text tokens.
            if t[0] == END_OF_AUDIO:
                return
            yield t
        truncated.append(text)

    def decode(frames):
        """(8,) code frames → 1-D float32 numpy waveform."""
        if not frames:
            return np.zeros(1, dtype=np.float32)
        waveform = processor.decode(torch.stack(frames, 1).unsqueeze(0))
        return waveform[0].detach().to(torch.float32).cpu().numpy()

    def synth_one(text):
        """Synthesize one text. Return (audio_1d_numpy, elapsed_s)."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        # decode() ends with .cpu(), so the clock stops only once the audio is materialised.
        audio = decode(list(audio_frames(text)))
        elapsed = time.perf_counter() - start
        return audio, elapsed

    def synth_one_streaming(text, early_stop=False):
        """TTFA probe only: return (audio_1d_numpy, first_audio_ts).

        liquid-audio has no streaming decode API, but the detokenizer is causal (sliding-window
        attention over the code frames), so decoding the first `--ttfa_chunk_frames` frames as soon
        as they exist yields the audio a streaming client could start playing. The whole utterance
        is still decoded in one call at the end, as in synth_one; the extra prefix decode is part of
        the timed generation, so it slightly understates batch-1 RTFx, never TTFA.
        """
        frames, first = [], None
        for frame in audio_frames(text):
            frames.append(frame)
            if first is None and len(frames) == args.ttfa_chunk_frames:
                chunk = decode(frames)  # ends with .cpu(): the audio is in hand
                first = time.perf_counter()
                if early_stop:
                    return chunk, first
        audio = decode(frames)
        if first is None:  # shorter than one chunk: first audio is the whole utterance
            first = time.perf_counter()
        return audio, first

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav}.
    # Manifest paths are relative to model_dir so later stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name, manifest_path = paths.model_dir, paths.dataset_dir_name, paths.manifest_path

    dataset = load_tts_dataset(args)

    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Timestamps the first decoded chunk (see synth_one_streaming).
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices

        def _ttfa_gen(job, early_stop):
            audio, first = synth_one_streaming(job["text"], early_stop)
            return audio, SAMPLING_RATE, first

        # The probe discards its first rows as warm-up but does not run one: compile the kernels
        # here so the first probed rows are not dominated by CUDA initialisation.
        for i in range(min(args.warmup_steps, len(dataset))):
            synth_one(dataset[i][args.text_column])
        set_seed(args.seed)

        run_probe(
            _ttfa_gen,
            [{"text": dataset[i][args.text_column]} for i in sample_indices(args.ttfa_probe, len(dataset))],
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=model_dir, n=args.ttfa_probe,
            # The chunk size is the floor under TTFA, so it is recorded with the numbers.
            extra={"device": args.device, "voice": args.voice, "ttfa_chunk_frames": args.ttfa_chunk_frames},
        )
        return

    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # Serial synthesis: one sample per "batch"; warm-up runs untimed on the first pending samples.
    starts = pending_batches(dataset, 1, done_entries, dataset_dir_name)
    warm_up(lambda batch: [synth_one(t) for t in batch[args.text_column]], dataset, starts, 1, args.warmup_steps)
    set_seed(args.seed)  # so warm-up does not shift the RNG stream of the timed run
    truncated.clear()

    # ── Main loop: synthesize one sample at a time → write JSONL ─────────────
    for i in tqdm(starts, desc="Generating"):
        sample_id = dataset[i]["id"]
        text = dataset[i][args.text_column]

        # Store the path relative to model_dir (the manifest's dir); write to the full path.
        rel_path = wav_rel_path(dataset_dir_name, sample_id)
        path = os.path.join(model_dir, rel_path)

        audio, elapsed = synth_one(text)
        sf.write(path, audio, SAMPLING_RATE)

        write_entry(manifest_file, manifest_entry(rel_path, len(audio) / SAMPLING_RATE, elapsed, text))
        manifest_file.flush()

    manifest_file.close()
    if truncated:
        print(f"WARNING: {len(truncated)} sample(s) hit --max_new_tokens={args.max_new_tokens} "
              "before the end-of-audio frame (audio cut off).")
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2.5-Audio-1.5B", help="LFM2-Audio repo id.")
    parser.add_argument("--voice", type=str, default="us_female", choices=sorted(VOICES),
                        help="Built-in voice (selected through the system prompt).")
    # Sampling defaults from the liquid-audio README's TTS example.
    parser.add_argument("--audio_temperature", type=float, default=0.8, help="Audio-code sampling temperature.")
    parser.add_argument("--audio_top_k", type=int, default=64, help="Audio-code top-k.")
    # 1 frame = 80 ms of audio, so 1024 frames is ~82 s: far beyond any eval utterance.
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max generated tokens (text + audio frames) per sample.")
    parser.add_argument("--ttfa_chunk_frames", type=int, default=1,
                        help="TTFA probe: frames (80 ms each) decoded for the first playable chunk.")
    add_common_args(parser)

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
