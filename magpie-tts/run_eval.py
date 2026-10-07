"""
TTS synthesis for the Open TTS Leaderboard (NVIDIA Magpie TTS Multilingual backend, stage 1).

Magpie TTS Multilingual is loaded via NeMo (`MagpieTTSModel`) with 5 baked speakers and no
zero-shot cloning; output is 22.05 kHz. The HF repo is gated, so set HF_TOKEN.

Batching: `do_tts()` is single-utterance, but its short-text path wraps `model.infer_batch()`,
which takes a padded token batch. This script mirrors that wrapper for a whole minibatch; texts
at/above the long-form threshold fall back to per-sample `do_tts()` inside the same timed block.

Memory: NeMo's attention materializes the full (B, heads, T, T) matrix and CFG doubles the batch,
so memory grows with the SQUARE of the padded length. `--batch_size` is therefore a nominal
maximum: minibatches are split to fit a budget on `n * max_text_tokens^2` (see
`_plan_chunk_size`), calibrated from the first OOM. Short texts keep the full batch, so their RTFx
stays comparable with the other backends.
"""

import argparse
import contextlib
import logging as pylogging
import os
import sys

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from nemo.collections.tts.models import MagpieTTSModel
from nemo.utils import logging as nemo_logging

from run_eval_utils import (
    add_common_args, count_parameters, load_done_entries, load_tts_dataset, manifest_entry, open_manifest,
    output_paths, pending_batches, print_next_steps, set_seed, wav_rel_path, warm_up, write_entry,
)

# NeMo is very chatty; errors only. infer_batch's raw prints are swallowed with redirect_stdout below.
nemo_logging.setLevel(pylogging.ERROR)

torch.set_float32_matmul_precision("high")

# Baked speakers, from the model card.
SPEAKER_MAP = {"John": 0, "Sofia": 1, "Aria": 2, "Jason": 3, "Leo": 4}

# Language -> tokenizer-name candidates; copied from MagpieTTSModel.do_tts (NeMo v2.7.3).
LANGUAGE_TOKENIZER_MAP = {
    "en": ["english_phoneme", "english"],
    "de": ["german_phoneme", "german"],
    "es": ["spanish_phoneme", "spanish"],
    "fr": ["french_chartokenizer", "french"],
    "it": ["italian_phoneme", "italian"],
    "vi": ["vietnamese_phoneme", "vietnamese"],
    "zh": ["mandarin_phoneme", "mandarin", "chinese"],
    # ja/hi are not in NeMo's do_tts map but the checkpoint ships these tokenizers. Keep them OUT of
    # LONGFORM_WORD_THRESHOLDS: the long-form path uses do_tts, which would pick the wrong tokenizer.
    "ja": ["japanese_phoneme", "japanese"],
    "hi": ["hindi_chartokenizer", "hindi"],
}

# Word-count thresholds above which do_tts switches to long-form (sentence-chunked)
# inference; copied from MagpieTTSModel._needs_longform_inference (NeMo v2.7.3).
# Standard mode generates at most ~20 s of speech.
LONGFORM_WORD_THRESHOLDS = {"en": 45, "es": 73, "fr": 69, "vi": 50, "it": 53, "de": 50}


def _resolve_checkpoint(model_id, revision):
    """Download the checkpoint's ``.nemo`` from the HF Hub at a pinned revision; return its path.

    The revision is pinned because newer upstream configs are incompatible with this NeMo build.
    A local ``.nemo`` path is returned unchanged.
    """
    if model_id.endswith(".nemo") and os.path.exists(model_id):
        return model_id
    from huggingface_hub import hf_hub_download, list_repo_files

    token = os.environ.get("HF_TOKEN")
    nemo_files = [f for f in list_repo_files(model_id, revision=revision, token=token) if f.endswith(".nemo")]
    if len(nemo_files) != 1:
        raise RuntimeError(f"Expected exactly one .nemo file in {model_id}@{revision}, found: {nemo_files}")
    print(f"Downloading {model_id}/{nemo_files[0]} at revision {revision} ...")
    return hf_hub_download(repo_id=model_id, filename=nemo_files[0], revision=revision, token=token)


def _admit_checkpoint_locales(nemo_path):
    """Whitelist any G2P locales the checkpoint declares but this NeMo build doesn't recognize.

    The checkpoint ships tokenizers for locales (e.g. ``pt-BR``) that ``validate_locale`` rejects,
    aborting model construction. That check is the only effect of ``locale`` here, so admitting
    them leaves every tokenizer's vocab (and thus the embedding shape and bos/eos ids) unchanged.
    ``SUPPORTED_LOCALES`` is mutated in place so every importer sees it.
    """
    from nemo.collections.common.tokenizers.text_to_speech import ipa_lexicon

    wanted = set()
    try:
        from omegaconf import OmegaConf

        cfg = MagpieTTSModel.restore_from(nemo_path, return_config=True)
        stack = [OmegaConf.to_container(cfg, resolve=False)]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, val in node.items():
                    if key == "locale" and isinstance(val, str):
                        wanted.add(val)
                    else:
                        stack.append(val)
            elif isinstance(node, list):
                stack.extend(node)
    except Exception as e:
        print(f"Could not pre-read checkpoint config to detect locales ({e}); "
              "falling back to a known-unsupported set.")
        wanted = {"pt-BR"}

    added = [loc for loc in sorted(wanted) if loc not in ipa_lexicon.SUPPORTED_LOCALES]
    ipa_lexicon.SUPPORTED_LOCALES.extend(added)
    if added:
        print(f"Admitted checkpoint G2P locales not in this NeMo build: {added}")


def main(args):
    # Set seed for reproducibility (generation samples audio tokens).
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.speaker not in SPEAKER_MAP:
        raise ValueError(f"Unknown speaker {args.speaker!r}; choose from {sorted(SPEAKER_MAP)}")
    speaker_idx = SPEAKER_MAP[args.speaker]

    nemo_path = _resolve_checkpoint(args.model_id, args.revision)
    # Must run before the real load, otherwise __init__ aborts building the tokenizers.
    _admit_checkpoint_locales(nemo_path)

    # Also pulls the NanoCodec vocoder referenced by the checkpoint config.
    model = MagpieTTSModel.restore_from(nemo_path)
    model = model.to(args.device)
    model.eval()
    if args.cfg_scale is not None:
        model.inference_parameters.cfg_scale = args.cfg_scale
    sampling_rate = int(model.output_sample_rate)  # 22050 (NanoCodec 22 kHz)
    print(
        f"Loaded Magpie TTS model {args.model_id} "
        f"(speaker={args.speaker}[{speaker_idx}], language={args.language}, sr={sampling_rate}, "
        f"use_cfg={args.use_cfg}, cfg_scale={model.inference_parameters.cfg_scale})"
    )

    # Report parameter count (deduped by id; MagpieTTSModel is a single nn.Module that
    # also registers the NanoCodec decoder as a submodule).
    try:
        n_params = count_parameters(model)
        if n_params:
            print(f"TTS model size: {n_params / 1e9:.2f}B parameters (incl. codec)")
    except Exception as e:
        print(f"Could not determine model size: {e}")

    # Resolve the tokenizer for the target language exactly like do_tts does.
    available_tokenizers = list(model.tokenizer.tokenizers.keys())
    tokenizer_name = None
    for candidate in LANGUAGE_TOKENIZER_MAP.get(args.language, []):
        if candidate in available_tokenizers:
            tokenizer_name = candidate
            break
    if tokenizer_name is None:
        tokenizer_name = available_tokenizers[0]
    pad_id = model.tokenizer.pad
    print(f"Using tokenizer '{tokenizer_name}' (available: {available_tokenizers})")

    longform_threshold = LONGFORM_WORD_THRESHOLDS.get(args.language)

    def _is_longform(text):
        if longform_threshold is None:
            return False
        word_count = len(list(text)) if args.language == "zh" else len(text.split())
        return word_count >= longform_threshold

    # Flat, bucket-friendly layout, per model:
    #   results/<model_safe>/  MODEL_<safe>_DATASET_<dsid>.jsonl  +  <dsid>/output_<id>.wav
    # Manifest paths are relative to model_dir, so downstream stages resolve them wherever it is copied.
    paths = output_paths(args)
    model_dir, dataset_dir_name = paths.model_dir, paths.dataset_dir_name

    # ── Attention-memory budget, calibrated at runtime ───────────────────────
    # Cost model: `n * max_text_tokens**2`, a proxy for the attention peak. `None` = no budget yet:
    # run the full batch, and on the first OOM derive the budget from the failed cost.
    # --max_attn_cost skips that probe.
    attn_budget = args.max_attn_cost or None
    # The failed attempt died partway through its decode, so its true peak was higher; halving adds margin.
    ATTN_BUDGET_SAFETY = 2

    def _plan_chunk_size(indices, token_lens):
        """Greedily chunk `indices` so each chunk's ``len(chunk) * max_text_tokens**2`` fits the budget.

        Chunks stay contiguous to preserve manifest order; an over-budget text forms a chunk of one.
        """
        if attn_budget is None:
            return [list(indices)]
        chunks, cur, cur_max = [], [], 0
        for i in indices:
            new_max = max(cur_max, token_lens[i])
            if cur and (len(cur) + 1) * new_max**2 > attn_budget:
                chunks.append(cur)
                cur, cur_max = [i], token_lens[i]
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            chunks.append(cur)
        return chunks

    def generate_tts(batch):
        """Synthesize speech for a minibatch of target texts; time the batch generation for RTFx."""
        texts = list(batch[args.text_column])
        minibatch_size = len(texts)

        # Short texts go through true batched infer_batch calls (the same call do_tts makes with
        # batch size 1); rare long-form texts fall back to do_tts.
        short_idx = [i for i, t in enumerate(texts) if not _is_longform(t)]
        long_idx = [i for i in range(minibatch_size) if i not in short_idx]

        audios = [None] * minibatch_size
        token_lists = {}  # index -> text token ids (populated inside the timed region below)
        wasted_s = 0.0  # GPU time burned by sub-batches that OOM'd and were retried smaller

        def _infer_chunk(chunk):
            """One batched infer_batch call over `chunk` (indices into `texts`); fills `audios`."""
            token_lens = [len(token_lists[i]) for i in chunk]
            text_tensor = torch.full((len(chunk), max(token_lens)), pad_id, dtype=torch.long)
            for row, i in enumerate(chunk):
                text_tensor[row, : token_lens[row]] = torch.tensor(token_lists[i], dtype=torch.long)
            nemo_batch = {
                "text": text_tensor.to(model.device),
                "text_lens": torch.tensor(token_lens, device=model.device, dtype=torch.long),
                "speaker_indices": speaker_idx,  # int broadcasts over the batch
            }
            out = model.infer_batch(
                nemo_batch,
                use_cfg=args.use_cfg,
                use_local_transformer_for_inference=True,
            )
            for row, i in enumerate(chunk):
                n = int(out.predicted_audio_lens[row].item())
                audios[i] = out.predicted_audio[row, :n].detach().float().cpu().numpy()

        def _infer_chunk_with_oom_retry(chunk):
            """Run `chunk`; on OOM, tighten the budget and retry the two halves.

            The aborted attempt's GPU time is excluded from the timed region so RTFx reflects only
            work that produced audio. Messages go to stderr (stdout is swallowed here).
            """
            nonlocal wasted_s, attn_budget
            attempt_start = torch.cuda.Event(enable_timing=True)
            attempt_end = torch.cuda.Event(enable_timing=True)
            attempt_start.record()
            try:
                _infer_chunk(chunk)
                return
            except torch.OutOfMemoryError:
                if len(chunk) == 1:  # can't split further — a single text OOMs on its own
                    raise
            attempt_end.record()
            torch.cuda.synchronize(device=args.device)
            wasted_s += attempt_start.elapsed_time(attempt_end) / 1000.0
            torch.cuda.empty_cache()

            max_len = max(len(token_lists[i]) for i in chunk)
            failed_cost = len(chunk) * max_len**2
            tightened = failed_cost // ATTN_BUDGET_SAFETY
            attn_budget = tightened if attn_budget is None else min(attn_budget, tightened)
            print(
                f"OOM on a sub-batch of {len(chunk)} (max {max_len} text tokens, cost {failed_cost}); "
                f"attention budget now {attn_budget} — pass --max_attn_cost={attn_budget} to skip this "
                f"probe on a rerun. Halving and retrying.",
                file=sys.stderr,
                flush=True,
            )
            mid = len(chunk) // 2
            _infer_chunk_with_oom_retry(chunk[:mid])
            _infer_chunk_with_oom_retry(chunk[mid:])

        # START TIMING (TTS batch generation)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=getattr(torch, args.dtype))
            if args.dtype != "float32"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx, open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            if short_idx:
                # Tokenization stays inside the timed region: the phonemizer is part of the cost.
                token_lists = {
                    i: model.tokenizer.encode(text=texts[i], tokenizer_name=tokenizer_name) + [model.eos_id]
                    for i in short_idx
                }
                token_lens = {i: len(t) for i, t in token_lists.items()}
                chunks = _plan_chunk_size(short_idx, token_lens)
                if len(chunks) > 1:
                    print(
                        f"Split a minibatch of {len(short_idx)} into sub-batches {[len(c) for c in chunks]} "
                        f"(max {max(token_lens.values())} text tokens, attention budget {attn_budget})",
                        file=sys.stderr,
                        flush=True,
                    )
                for chunk in chunks:
                    _infer_chunk_with_oom_retry(chunk)

            for i in long_idx:
                wav, wav_len = model.do_tts(
                    texts[i],
                    language=args.language,
                    apply_TN=False,
                    use_cfg=args.use_cfg,
                    speaker_index=speaker_idx,
                )
                n = int(wav_len[0].item())
                audios[i] = wav[0, :n].detach().float().cpu().numpy()

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        # Exclude time spent on OOM'd attempts that produced no audio (see _infer_chunk_with_oom_retry).
        runtime = max(start_event.elapsed_time(end_event) / 1000.0 - wasted_s, 1e-6)
        # per-sample generation time (RTFx is aggregated over the whole set at scoring time)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(audios, batch["id"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            path = os.path.join(model_dir, rel_path)
            audio = np.asarray(audio).reshape(-1)
            if audio.size == 0:  # degenerate generation; write a stub so the pipeline continues
                audio = np.zeros(int(0.1 * sampling_rate), dtype=np.float32)
            sf.write(path, audio, sampling_rate)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        batch["gen_audio_filepath"] = gen_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        return batch

    # Fixed baked voice (no cloning): only `id` and the target text are kept.
    dataset = load_tts_dataset(args)

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # Drives generate_tts() with one-row batches; `model_dir`/`dataset_dir_name` are rebound to a
    # temp dir (generate_tts closes over them) so nothing lands in the results tree. No streaming
    # API, so TTFA is the whole-utterance time. See scripts/ttfa_probe.py.
    if args.ttfa_probe != 0:
        # Imported here: the probe module is injected only by submit_ttfa_jobs.sh.
        from ttfa_probe import run_probe, sample_indices
        import tempfile

        model_dir = tempfile.mkdtemp(prefix="ttfa_out_")
        dataset_dir_name = "ttfa"
        output_dir = os.path.join(model_dir, dataset_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        def _gen(job):
            result = generate_tts(job)
            return float(result["audio_length_s"][0]), None, None

        columns = dataset.column_names
        jobs = [{c: [dataset[i][c]] for c in columns}
                for i in sample_indices(args.ttfa_probe, len(dataset))]

        run_probe(
            _gen, jobs,
            model_id=args.model_id, dataset_path=args.dataset_path,
            dataset_config=args.dataset, split=args.split,
            out_dir=probe_out_dir, mode_suffix="", n=args.ttfa_probe,  # fixed voice: no clone mode
            extra={"device": args.device, "note": "driven at batch size 1"},
        )
        return

    # ── Manifest (written incrementally; resume skips rows already in it) ───
    manifest_path = paths.manifest_path
    done_entries = load_done_entries(manifest_path, args.resume)
    manifest_file = open_manifest(manifest_path, done_entries)

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(generate_tts, dataset, starts, args.batch_size, args.warmup_steps)
    for batch_start in tqdm(starts, desc="Generating"):
        batch = generate_tts(dataset[batch_start : batch_start + args.batch_size])

        # Append each sample to the JSONL immediately.
        for afp, alen, gtime, ref in zip(
            batch["gen_audio_filepath"], batch["audio_length_s"],
            batch["generation_time_s"], batch["references"],
        ):
            write_entry(manifest_file, manifest_entry(afp, alen, gtime, ref))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/magpie_tts_multilingual_357m",
        help="Magpie TTS model id on the HF Hub (gated; needs HF_TOKEN), loaded with NeMo. "
        "May also be a local .nemo path (then --revision is ignored).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="34d7e40da85cabc97f92198889b65cea27bc7fd1",
        help="Pinned HF Hub revision (commit) of the checkpoint, known to load under NeMo 2.7.3. "
        "Use 'main' for latest.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Sofia",
        help="Baked speaker voice: John, Sofia, Aria, Jason, or Leo (no zero-shot cloning).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code: en, de, es, fr, it, vi, zh, ja, hi (see LANGUAGE_TOKENIZER_MAP). An "
        "unmapped language silently falls back to the first available tokenizer.",
    )
    parser.add_argument(
        "--use_cfg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Classifier-free guidance (default on, matching do_tts). Disable with --no-use_cfg.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="Override the checkpoint's CFG scale (model card batch example uses 2.5); default: checkpoint value.",
    )
    add_common_args(parser, batch_size=32)
    parser.add_argument(
        "--max_attn_cost",
        type=int,
        default=0,
        help="Initial cap on `n * max_text_tokens^2` per infer_batch call; a minibatch exceeding it "
        "is split into sub-batches. Default 0 = start unbounded and calibrate from the first OOM.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Generation precision. NeMo Magpie inference is float32 by default; other values "
        "(e.g. 'bfloat16') run generation under torch.autocast.",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
