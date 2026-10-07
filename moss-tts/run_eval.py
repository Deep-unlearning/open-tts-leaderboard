"""
TTS synthesis for the Open TTS Leaderboard (MOSS-TTS backend, stage 1).

MOSS-TTS (OpenMOSS-Team/MOSS-TTS) is an ~8B Qwen3-based delay-codebook TTS loaded in-process
via transformers with `trust_remote_code=True`. Its API is custom: build conversation messages
with the processor, run a custom `generate`, then `processor.decode` back to waveforms. It
supports batched generation, so texts are synthesized in minibatches (the batch call is timed
for RTFx).

Voice cloning (`--voice_clone`): each message is built with the dataset's per-sample
`prompt_audio` as the reference clip (written to `results/<model_safe>/<dsid>/prompts/prompt_{id}.wav`,
which MOSS's `build_user_message(reference=[path])` and stage 3 both read). Outputs get a
`_voice_clone` suffix so cloning and non-cloning runs coexist. By default (`--no-voice_clone`)
the built-in default voice is used and speaker-similarity scoring is skipped.

Wavs + a manifest with `text`, `duration`, and generation `time` (empty `pred_text`) are written
for stage 2 (`transformers/transcribe.py`, Qwen3-ASR), stage 3 (`transformers/score_similarity.py`,
speaker SIM — voice-clone only), and stage 4 (local `normalizer.eval_utils.score_results`, WER + RTFx + mean SIM).
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from transformers import AutoModel, AutoProcessor

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


def main(args):
    # Set seed for reproducibility (MOSS-TTS samples audio tokens).
    set_seed(args.seed)

    # SDPA backend hygiene (from the upstream CLI).
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    # The audio (de)tokenizer runs the vocoder; move it to the compute device.
    processor.audio_tokenizer = processor.audio_tokenizer.to(args.device)
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True, dtype=dtype).to(args.device)
    model.eval()
    sampling_rate = int(processor.model_config.sampling_rate)  # 24000
    print(f"Loaded MOSS-TTS ({args.model_id}, sr={sampling_rate})")
    print_model_size(model)

    # Layout: results/<model_safe>/{MODEL_<safe>_DATASET_<dsid>.jsonl, <dsid>/output_<id>.wav},
    # <dsid> = <dataset_safe>_<dataset>_<split><mode_suffix>. Manifest paths are relative to
    # model_dir so later stages resolve them wherever it is copied.
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    # Reference clips must live under results/ (not tempfiles) so stage 3 SIM can read them.
    prompt_dir = os.path.join(output_dir, "prompts")
    if args.voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    def generate_tts(texts, ids, prompt_full_paths):
        """Synthesize a batch of texts (optionally cloning prompt_full_paths); time the batch for RTFx.

        `prompt_full_paths` must be on-disk paths (MOSS opens them); returned paths are relative.
        """
        minibatch_size = len(texts)
        if args.voice_clone:
            conversations = [
                [processor.build_user_message(text=t, language=args.language, reference=[pp])]
                for t, pp in zip(texts, prompt_full_paths)
            ]
        else:
            conversations = [
                [processor.build_user_message(text=t, language=args.language)] for t in texts
            ]

        # START TIMING (TTS batch generation)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        batch = processor(conversations, mode="generation")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                max_new_tokens=args.max_new_tokens,
                audio_temperature=args.audio_temperature,
                audio_top_p=args.audio_top_p,
                audio_top_k=args.audio_top_k,
                audio_repetition_penalty=args.audio_repetition_penalty,
            )
        messages = processor.decode(outputs)

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        per_sample_time = runtime / minibatch_size

        gen_paths, audio_length_s = [], []
        for msg, sample_id in zip(messages, ids):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = os.path.join(dataset_dir_name, f"output_{sample_id}.wav")
            path = os.path.join(model_dir, rel_path)
            if msg is None or not getattr(msg, "audio_codes_list", None):
                # Empty generation — write a short silence so the manifest stays aligned.
                audio = np.zeros(int(0.1 * sampling_rate), dtype=np.float32)
            else:
                audio = msg.audio_codes_list[0]
                audio = np.asarray(audio.detach().to(torch.float32).cpu()).reshape(-1)
            sf.write(path, audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        return gen_paths, audio_length_s, minibatch_size * [per_sample_time]

    # Keep `id`, target text, and (when cloning) the per-sample reference.
    dataset = load_tts_dataset(args, extra_columns=("prompt_text", "prompt_audio") if args.voice_clone else ())

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # TTFA is per-request, so generate_tts() is driven with one-row batches. model_dir and
    # dataset_dir_name are rebound to a temp dir (generate_tts reads them late-bound), so no
    # results are touched. No streaming API: `first` is None -> whole-utterance fallback.
    # 0 = off; N > 0 = N evenly spaced samples; N < 0 = the whole split (see scripts/ttfa_probe.py).
    if args.ttfa_probe != 0:
        # Imported here: ttfa_probe.py is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)

        def _gen(job):
            ref = None
            if args.voice_clone:
                ref = os.path.join(output_dir, "prompts", f"prompt_{job['id'][0]}.wav")
                pa = job["prompt_audio"][0]
                sf.write(ref, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
            _, audio_length_s, _ = generate_tts(
                [job[args.text_column][0]], [job["id"][0]], [ref]
            )
            return float(audio_length_s[0]), None, None

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

    def run_batch(batch):
        """Persist the batch's reference clips (when cloning), then generate_tts; adds prompt paths."""
        ids = batch["id"]
        # MOSS + stage-3 SIM read the clips from disk. Store the path relative to model_dir (the
        # manifest's dir); write to the full path.
        prompt_rel_paths = [None] * len(ids)
        prompt_full_paths = [None] * len(ids)
        if args.voice_clone:
            for j, sid in enumerate(ids):
                prel = os.path.join(dataset_dir_name, "prompts", f"prompt_{sid}.wav")
                pp = os.path.join(model_dir, prel)
                if not os.path.exists(pp):
                    pa = batch["prompt_audio"][j]
                    sf.write(pp, np.asarray(pa["array"], dtype=np.float32), pa["sampling_rate"])
                prompt_rel_paths[j] = prel
                prompt_full_paths[j] = pp
        return (*generate_tts(list(batch[args.text_column]), ids, prompt_full_paths), prompt_rel_paths)

    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(run_batch, dataset, starts, args.batch_size, args.warmup_steps)

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    for batch_start in tqdm(starts, desc="Generating"):
        batch = dataset[batch_start : batch_start + args.batch_size]
        gen_paths, audio_length_s, gen_times, prompt_rel_paths = run_batch(batch)

        # `pred_text` is filled by stage 2 (ASR); `sim` by stage 3 (voice-clone only).
        for afp, ppath, alen, gtime, ref in zip(
            gen_paths, prompt_rel_paths, audio_length_s, gen_times, batch[args.text_column]
        ):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref, ppath))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(args.voice_clone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser, batch_size=32, voice_clone=False)

    parser.add_argument("--model_id", type=str, default="OpenMOSS-Team/MOSS-TTS", help="MOSS-TTS model id.")
    parser.add_argument("--language", type=str, default=None, help="Language tag passed to build_user_message. Default: None, as in the model card, so the model infers it from the text.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype, e.g. 'bfloat16'.")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max audio tokens per generation.")
    # Audio-sampling params (MossTTSDelay recommended defaults).
    parser.add_argument("--audio_temperature", type=float, default=1.7, help="Audio-token sampling temperature.")
    parser.add_argument("--audio_top_p", type=float, default=0.8, help="Audio-token top-p.")
    parser.add_argument("--audio_top_k", type=int, default=25, help="Audio-token top-k.")
    parser.add_argument("--audio_repetition_penalty", type=float, default=1.0, help="Audio-token repetition penalty.")

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
