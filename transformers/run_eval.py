"""
TTS synthesis for the Open TTS Leaderboard (Transformers backend, stage 1).

Synthesizes the target text in batches (generation is timed for RTFx). With `--voice_clone`
(off by default) the dataset's reference prompt audio/text condition the output speaker.

Stage 2 (`transformers/transcribe.py`) transcribes the wavs, stage 3 (`score_similarity.py`,
voice cloning only) scores SIM, and stage 4 (local scoring) computes WER + RTFx, as for every backend.

Voice cloning is supported for VOICE_CLONE_FAMILIES; other models fall back to their default voice
(and skip SIM). Prompt wavs are saved under `output_dir/prompts/` for the SIM stage.

Per-family notes: SpeechT5 needs a speaker embedding + HiFi-GAN vocoder (`--speecht5_*`).
VITS/MMS and FastSpeech2 synthesize one sample at a time (batch padding causes a hum).
SeamlessM4T-v2 stays batched and trims each row to its `waveform_lengths`. VibeVoice returns already-trimmed per-sample waveforms and needs `diffusers`.
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTextToWaveform,
    AutoProcessor,
    CompileConfig,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    FastSpeech2ConformerTokenizer
)

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
    wav_rel_path,
    write_entry,
)


torch.set_float32_matmul_precision("high")

# Families that support `--voice_clone` (substrings of `config.model_type`). Keep in sync with
# VOICE_CLONE_MODELS in transformers/submit_jobs.sh.
VOICE_CLONE_FAMILIES = ("dia", "higgs", "vibevoice")

# ISO 639-1 (as passed by the eval scripts) -> ISO 639-3 (SeamlessM4T) for the 36 languages
# SeamlessM4T v2 can SPEAK (fewer than it can translate). `cmn` is simplified Mandarin.
SEAMLESS_LANGUAGE_CODES = {
    "ar": "arb", "bn": "ben", "ca": "cat", "cs": "ces", "cy": "cym", "da": "dan",
    "de": "deu", "en": "eng", "es": "spa", "et": "est", "fa": "pes", "fi": "fin",
    "fr": "fra", "hi": "hin", "id": "ind", "it": "ita", "ja": "jpn", "ko": "kor",
    "mt": "mlt", "nl": "nld", "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus",
    "sk": "slk", "sv": "swe", "sw": "swh", "te": "tel", "th": "tha", "tl": "tgl",
    "tr": "tur", "uk": "ukr", "ur": "urd", "uz": "uzn", "vi": "vie", "zh": "cmn",
}
SEAMLESS_SPEECH_CODES = frozenset(SEAMLESS_LANGUAGE_CODES.values())


def _seamless_lang(language):
    """Translate the pipeline's language code to the one SeamlessM4T expects.

    Unmapped values pass through (a raw Seamless code is accepted), but the result must be speakable;
    otherwise `generate()` fails late with a misleading error.
    """
    code = SEAMLESS_LANGUAGE_CODES.get(language.lower(), language)
    if code not in SEAMLESS_SPEECH_CODES:
        raise ValueError(
            f"SeamlessM4T cannot synthesize speech for --language={language!r} (resolved to "
            f"{code!r}). It supports {len(SEAMLESS_SPEECH_CODES)} languages for speech output; pass "
            "one of these ISO 639-1 codes (or its ISO 639-3 equivalent): "
            f"{', '.join(sorted(SEAMLESS_LANGUAGE_CODES))}."
        )
    return code


def load_tts_model(args, torch_dtype):
    """Load the TTS model under evaluation, its processor, and (for SpeechT5) the vocoder + speaker embedding.

    Returns (config, model, processor, extras) where `extras` holds SpeechT5-specific objects
    (`vocoder`, `speaker_embeddings`) or is empty for other models.
    """
    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision)
    extras = {}

    # Validate the language before loading weights (otherwise it only fails inside generate()).
    if "seamless" in config.model_type:
        _seamless_lang(args.language)

    if "speecht5" in config.model_type:
        # SpeechT5 is an encoder-decoder TTS that needs a speaker embedding and a separate HiFi-GAN
        # vocoder, and generates via `generate_speech` (not the generic `generate`). Load in float32
        # for output quality — it is small and the stop-token threshold is sensitive to low precision.
        from datasets import load_dataset

        model = SpeechT5ForTextToSpeech.from_pretrained(args.model_id, revision=args.revision).to(args.device)
        vocoder = SpeechT5HifiGan.from_pretrained(args.speecht5_vocoder).to(args.device)
        xvectors = load_dataset("parquet", data_files=args.speecht5_speaker_embeddings, split="train")
        speaker = torch.tensor(xvectors[args.speecht5_speaker_index]["xvector"]).unsqueeze(0).to(args.device)
        extras = {"vocoder": vocoder, "speaker_embeddings": speaker}
    elif "dia" in config.model_type:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_id,
            dtype=torch_dtype,
            device_map=args.device,
            attn_implementation=args.attn_implementation,
        )
    else:
        # Bark, Seamless, FastSpeech2, and VITS only support eager attention.
        eager_only = any(f in config.model_type for f in ("bark", "seamless", "fastspeech2", "vits"))
        attn_impl = "eager" if eager_only else args.attn_implementation
        # FastSpeech2 and VITS require float32; lower precision causes dtype mismatches or numerical errors.
        model_dtype = torch.float32 if ("fastspeech2" in config.model_type or "vits" in config.model_type) else torch_dtype
        if "vibevoice" in config.model_type:
            # Only the LM backbone (`text_config`) has attention; the tokenizers reject anything but
            # eager, so route the requested implementation to the backbone only.
            attn_impl = {"": "eager", "text_config": args.attn_implementation}
        model = AutoModelForTextToWaveform.from_pretrained(
            args.model_id,
            dtype=model_dtype,
            device_map=args.device,
            attn_implementation=attn_impl,
        )
    model.eval()
    print_model_size(model)

    processor_kwargs = {"device_map": args.device} if "higgs" in config.model_type else {}
    if "fastspeech2" in config.model_type:
        # FastSpeech2ConformerWithHifiGan uses a tokenizer from a separate repo.
        processor = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
    else:
        processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision, **processor_kwargs)
    return config, model, processor, extras


def build_gen_kwargs(args, config, model):
    """Assemble generation kwargs per model family."""
    gen_kwargs = {}
    if model.can_generate():
        # Greedy decoding for reproducible WER. GREEDY_UNSAFE_FAMILIES opt out: under greedy they
        # stop emitting EOS and run to the token cap, so they use their checkpoint's sampling defaults.
        GREEDY_UNSAFE_FAMILIES = ("dia", "bark", "csm")
        if not any(family in config.model_type for family in GREEDY_UNSAFE_FAMILIES):
            gen_kwargs["do_sample"] = False
            gen_kwargs["temperature"] = 0.0
        # Bark sets max_new_tokens internally per stage and rejects it as an explicit kwarg.
        if "bark" not in config.model_type:
            gen_kwargs["max_new_tokens"] = args.max_new_tokens
        # `min_new_tokens` pins the length for families that do not stop on their own; for the
        # NO_MIN_TOKENS_FAMILIES `--max_new_tokens` is only a cap and they stop at EOS.
        NO_MIN_TOKENS_FAMILIES = ("higgs", "qwen3_omni", "qwen2_5_omni", "dia", "bark", "seamless", "vibevoice")
        if not any(family in config.model_type for family in NO_MIN_TOKENS_FAMILIES):
            gen_kwargs["min_new_tokens"] = args.max_new_tokens
        if "csm" in config.model_type:
            gen_kwargs["output_audio"] = True
            # Sampling follows the checkpoint's generation_config (do_sample=True, temperature=0.9);
            # the model card's greedy example is a latency benchmark, not the inference default.
        if "seamless" in config.model_type:
            # Output language; src_lang is a processor kwarg, set at tokenization below.
            gen_kwargs["tgt_lang"] = _seamless_lang(args.language)
        if "qwen3_omni" in config.model_type:
            # The bare greedy kwargs above only reach the "thinker" (text) stage. The "talker"
            # (audio) stage is deliberately left at its checkpoint sampling defaults: greedy talker
            # decoding fails to emit EOS on some prompts. Don't add talker_do_sample=False.
            gen_kwargs["speaker"] = args.qwen_omni_speaker
            # The bare `max_new_tokens` never overrides the thinker's default; only the prefixed
            # kwarg does. Audio length is bounded by the talker's own `talker_max_new_tokens`.
            del gen_kwargs["max_new_tokens"]
            gen_kwargs["thinker_max_new_tokens"] = args.max_new_tokens
        if "qwen2_5_omni" in config.model_type:
            # As with Qwen3-Omni, the talker keeps its sampling defaults; don't add talker_do_sample=False.
            # Qwen2.5-Omni uses title-case speaker names and only supports Ethan/Chelsie (no Aiden).
            speaker = args.qwen_omni_speaker.capitalize()
            if speaker not in ["Ethan", "Chelsie"]:
                print("Warning: Qwen2.5-Omni only supports Ethan/Chelsie; forcing speaker=Ethan.")
                speaker = "Ethan"
            gen_kwargs["speaker"] = speaker
            del gen_kwargs["max_new_tokens"]
    elif args.max_new_tokens:
        raise ValueError("`max_new_tokens` should only be set for auto-regressive models.")

    if args.torch_compile is not None:
        if model.can_generate():
            gen_kwargs["compile_config"] = CompileConfig(mode=args.torch_compile, fullgraph=args.compile_fullgraph)
            model.generation_config.cache_implementation = "static"
        else:
            model = torch.compile(model, mode=args.torch_compile, fullgraph=args.compile_fullgraph)
        if args.warmup_steps is None or args.warmup_steps < 1:
            print("`--torch_compile` is enabled; forcing `--warmup_steps=10` to trigger compilation.")
            args.warmup_steps = 10
    return gen_kwargs, model


def apply_tts_chat_template(processor, config, texts, prompt_texts=None, prompt_audio_paths=None):
    """Apply the processor chat template (if any), returning either texts or pre-tokenized inputs.

    When `prompt_texts`/`prompt_audio_paths` are provided (voice cloning), the dia, higgs and
    vibevoice families condition generation on the reference prompt audio (dia/higgs additionally
    take its transcript; vibevoice's template has no slot for one).
    """
    voice_clone = prompt_texts is not None and prompt_audio_paths is not None

    if "dia" in config.model_type:
        # Dia requires a speaker-turn tag prefix; it does not use a chat template.
        if voice_clone:
            # Prepend the prompt transcript to the target text; the prompt audio is passed as
            # `audio=` to the processor by the caller and stripped from the output via
            # `get_audio_prompt_len` at decode time.
            return [f"[S1] {pt} [S1] {text}" for pt, text in zip(prompt_texts, texts)], None
        return [f"[S1] {text}" for text in texts], None

    if getattr(processor, "chat_template", None) is None:
        return texts, None

    if "csm" in config.model_type:
        # CSM uses a speaker id as the "role" rather than user/assistant.
        return [
            processor.apply_chat_template(
                [{"role": "0", "content": [{"type": "text", "text": text}]}],
                tokenize=False,
                add_generation_prompt=True,
                return_dict=False,
            )
            for text in texts
        ], None
    if "higgs" in config.model_type:
        if voice_clone:
            # Zero-shot voice cloning: the prompt transcript + prompt audio are supplied as a
            # user/assistant turn pair before the target user turn (Higgs Audio v2 convention).
            conversations = [
                [
                    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
                    {"role": "user", "content": [{"type": "text", "text": pt}]},
                    {"role": "assistant", "content": [{"type": "audio", "url": pa}]},
                    {"role": "user", "content": [{"type": "text", "text": text}]},
                ]
                for text, pt, pa in zip(texts, prompt_texts, prompt_audio_paths)
            ]
        else:
            conversations = [
                [
                    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
                    {"role": "user", "content": text},
                ]
                for text in texts
            ]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={"return_tensors": "pt", "sampling_rate": 24000},
        )
        return texts, inputs
    if "vibevoice" in config.model_type:
        # The "role" is a speaker id (always "0" here). Cloning uses only the reference audio (no
        # transcript slot), loaded and resampled by apply_chat_template.
        conversations = [
            [
                {
                    "role": "0",
                    "content": [{"type": "text", "text": text}]
                    + ([{"type": "audio", "url": pa}] if voice_clone else []),
                }
            ]
            for text, pa in zip(texts, prompt_audio_paths or [None] * len(texts))
        ]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            # librosa is pinned because the "auto" audio backend picks torchcodec whenever it is
            # importable, even without usable FFmpeg libs. The prompts are wavs written above.
            processor_kwargs={"return_tensors": "pt", "padding": True, "load_audio_backend": "librosa"},
        )
        return texts, inputs
    if "qwen3_omni" in config.model_type or "qwen2_5_omni" in config.model_type:
        # Qwen Omni models (2.5 and 3) require this exact system prompt for audio output,
        # and the user turn must explicitly ask to read the text aloud verbatim.
        system_text = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
            "perceiving auditory and visual inputs, as well as generating text and speech."
        )
        inputs = processor.apply_chat_template(
            [
                [
                    {"role": "system", "content": [{"type": "text", "text": system_text}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Please read the following text aloud exactly as written, with no "
                                    f"additional commentary:\n\n{text}"
                                ),
                            }
                        ],
                    },
                ]
                for text in texts
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            # `padding` is required for --batch_size > 1 (batched audio generation needs a real
            # attention_mask; see transformers#47186).
            processor_kwargs={"return_tensors": "pt", "padding": True},
        )
        return texts, inputs
    return [
        processor.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
        )
        for text in texts
    ], None


def main(args):
    # Set seed due to randomness in some models (e.g. VibeVoice's acoustic tokenizer sampling)
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    torch_dtype = getattr(torch, args.dtype)
    config, model, processor, extras = load_tts_model(args, torch_dtype)

    # Voice cloning is only wired up for the families in VOICE_CLONE_FAMILIES.
    voice_clone = args.voice_clone
    if voice_clone and not any(family in config.model_type for family in VOICE_CLONE_FAMILIES):
        print(
            f"Warning: --voice_clone is not supported for model_type '{config.model_type}'; "
            "falling back to the model's default voice (--no-voice_clone)."
        )
        voice_clone = False
    mode_suffix = "_voice_clone" if voice_clone else ""

    is_speecht5 = "speecht5" in config.model_type
    is_fastspeech2 = "fastspeech2" in config.model_type
    is_vits = "vits" in config.model_type
    is_seamless = "seamless" in config.model_type
    is_vibevoice = "vibevoice" in config.model_type
    # SpeechT5, FastSpeech2, and VITS use direct forward passes, not `generate`, so skip gen-kwargs.
    gen_kwargs = {}
    if not is_speecht5 and not is_fastspeech2 and not is_vits:
        gen_kwargs, model = build_gen_kwargs(args, config, model)

    sampling_rate = 16_000
    if "qwen3_omni" in config.model_type or "qwen2_5_omni" in config.model_type:
        # feature_extractor.sampling_rate is 16kHz (input audio), but talker output is fixed 24kHz.
        sampling_rate = 24_000
    elif "fastspeech2" in config.model_type:
        # FastSpeech2ConformerWithHifiGan outputs at 22050 Hz.
        sampling_rate = 22_050
    elif "vits" in config.model_type:
        # VITS/MMS sampling rate is stored in model config.
        sampling_rate = getattr(config, "sampling_rate", 16_000)
    elif "bark" in config.model_type:
        # Bark outputs at 24kHz; its processor has no feature_extractor with a sampling_rate.
        sampling_rate = 24_000
    elif "seamless" in config.model_type:
        # SeamlessM4Tv2 outputs at 16kHz.
        sampling_rate = 16_000
    elif getattr(processor, "feature_extractor", None) is not None and hasattr(
        processor.feature_extractor, "sampling_rate"
    ):
        sampling_rate = processor.feature_extractor.sampling_rate

    # Layout: results/<model_safe>/  MODEL_<safe>_DATASET_<dsid>.jsonl  +  <dsid>/output_<id>.wav
    # Manifest paths are relative to model_dir so downstream stages resolve them wherever it is copied.
    paths = output_paths(args, mode_suffix)
    model_dir, dataset_dir_name, output_dir = paths.model_dir, paths.dataset_dir_name, paths.output_dir
    # Reference prompt wavs are persisted here so the SIM stage (score_similarity.py) can read them.
    prompt_dir = os.path.join(output_dir, "prompts")
    if voice_clone:
        os.makedirs(prompt_dir, exist_ok=True)

    def generate_tts(batch):
        """Synthesize speech for a minibatch of target texts; time the batch generation for RTFx."""
        texts_to_generate = list(batch[args.text_column])
        minibatch_size = len(texts_to_generate)

        # Voice cloning: persist each reference prompt wav (once) and collect per-sample
        # prompt transcripts, paths (higgs / vibevoice) and raw arrays (dia) aligned with the batch.
        prompt_texts = prompt_audio_paths = prompt_arrays = None
        if voice_clone:
            prompt_texts = list(batch["prompt_text"])
            prompt_audio_paths, prompt_arrays = [], []
            for sample_id, prompt_audio in zip(batch["id"], batch["prompt_audio"]):
                arr = np.asarray(prompt_audio["array"], dtype=np.float32)
                sr = int(prompt_audio["sampling_rate"])
                pp = os.path.join(prompt_dir, f"prompt_{sample_id}.wav")
                if not os.path.exists(pp):
                    sf.write(pp, arr, sr)
                prompt_audio_paths.append(os.path.abspath(pp))
                if "dia" in config.model_type:
                    # Dia's processor expects the reference audio at 44100 Hz.
                    if sr != 44_100:
                        import librosa

                        arr = librosa.resample(arr, orig_sr=sr, target_sr=44_100)
                    prompt_arrays.append(arr)

        # START TIMING (TTS batch generation)
        torch.cuda.synchronize(device=args.device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        if args.torch_compile is not None:
            sdpa_backends = [SDPBackend.MATH]
        else:
            sdpa_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

        if is_vits:
            # VITS / MMS-TTS: process each sample individually to avoid padding-induced hum.
            outputs = []
            for text in texts_to_generate:
                inp = processor([text], return_tensors="pt").to(args.device)
                with torch.no_grad(), sdpa_kernel(sdpa_backends):
                    outputs.append(model(**inp).waveform[0])
        elif is_fastspeech2:
            # FastSpeech2: process each sample individually to avoid padding-induced hum.
            outputs = []
            for text in texts_to_generate:
                inp = processor([text], return_tensors="pt").to(args.device)
                with torch.no_grad(), sdpa_kernel(sdpa_backends):
                    outputs.append(model(inp["input_ids"], return_dict=True)["waveform"][0])
        elif is_speecht5:
            # SpeechT5: batched mel-spectrogram generation + HiFi-GAN vocoding in one call.
            # A single speaker embedding is broadcast to the whole batch by generate_speech.
            inputs = processor(text=texts_to_generate, padding=True, return_tensors="pt").to(args.device)
            with sdpa_kernel(sdpa_backends):
                waveforms, lengths = model.generate_speech(
                    inputs["input_ids"],
                    extras["speaker_embeddings"],
                    attention_mask=inputs["attention_mask"],
                    vocoder=extras["vocoder"],
                    return_output_lengths=True,
                )
            outputs = [waveforms[i, : lengths[i]] for i in range(len(lengths))]
        else:
            # 1. Pre-processing. Pad to full batch size under torch.compile to avoid recompilations.
            padding_size = None
            if minibatch_size != args.batch_size and args.torch_compile is not None:
                padding_size = args.batch_size - minibatch_size
                texts_to_generate.extend([texts_to_generate[-1]] * padding_size)

            texts_to_generate, inputs = apply_tts_chat_template(
                processor, config, texts_to_generate, prompt_texts, prompt_audio_paths
            )
            if inputs is None:  # not pre-tokenized by the chat template (higgs / vibevoice / Qwen-Omni are)
                # BarkProcessor passes padding internally; passing it explicitly causes a conflict.
                # SeamlessM4T requires src_lang to tokenize text input.
                proc_kwargs = {}
                if "bark" not in config.model_type:
                    proc_kwargs["padding"] = True
                if is_seamless:
                    proc_kwargs["src_lang"] = _seamless_lang(args.language)
                if voice_clone and "dia" in config.model_type:
                    # Dia conditions on the reference audio via the `audio=` arg; the prompt tokens
                    # are stripped from the output by `get_audio_prompt_len` at decode time.
                    proc_kwargs["audio"] = prompt_arrays
                inputs = processor(text=texts_to_generate, **proc_kwargs, return_tensors="pt")
            if is_vibevoice:
                # The reference waveform (`input_values`) comes out of the feature extractor in
                # float32 while the acoustic tokenizer runs in the model's dtype, so the cast is
                # part of the move here (what the model card does with `.to(device, model.dtype)`).
                # BatchFeature.to() only casts floating-point tensors, so input_ids stay integral.
                inputs = inputs.to(args.device, model.dtype)
            else:
                inputs = inputs.to(args.device)

            # 2. Model inference
            with sdpa_kernel(sdpa_backends):
                if "qwen3_omni" in config.model_type or "qwen2_5_omni" in config.model_type:
                    # generate() returns (text_ids, audio) — we only need the audio.
                    _, pred_waveform = model.generate(**inputs, **gen_kwargs)
                else:
                    pred_waveform = model.generate(**inputs, **gen_kwargs)

            # 3. Post-processing
            seamless_lengths = None
            if is_seamless:
                # generate() returns (waveform, waveform_lengths); split before the padding trim.
                # `.reshape(-1)` handles batch size 1, where lengths is 0-dim.
                pred_waveform, seamless_lengths = pred_waveform[0], pred_waveform[1].reshape(-1)

            if padding_size is not None:
                # Batched Qwen-Omni returns a list of per-sample waveforms, which slices like a
                # sequence; every other backend returns a tensor, which needs the ellipsis.
                pred_waveform = (
                    pred_waveform[:-padding_size]
                    if isinstance(pred_waveform, (list, tuple))
                    else pred_waveform[:-padding_size, ...]
                )
                if seamless_lengths is not None:
                    seamless_lengths = seamless_lengths[:-padding_size]

            if config.model_type == "dia":
                prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
                outputs = processor.batch_decode(pred_waveform, audio_prompt_len=prompt_len)
            elif "higgs" in config.model_type:
                outputs = processor.batch_decode(pred_waveform)
            elif "qwen3_omni" in config.model_type or "qwen2_5_omni" in config.model_type:
                # A batch yields a list of already-trimmed waveforms; a single sample a `(1, 1, N)`
                # tensor. Normalize both to a list of 1-D tensors.
                waveforms = pred_waveform if isinstance(pred_waveform, (list, tuple)) else [pred_waveform]
                outputs = [w.reshape(-1) for w in waveforms]
            elif is_vibevoice:
                # One already-trimmed (1, num_samples) tensor per row; flatten so len() counts samples.
                # None means no audio was generated: keep the row as short silence (scores 100% WER).
                outputs = []
                for sample_id, waveform in zip(batch["id"], pred_waveform):
                    if waveform is None:
                        print(f"Warning: VibeVoice generated no audio for sample {sample_id}.")
                        waveform = torch.zeros(sampling_rate // 10)
                    outputs.append(waveform.reshape(-1))
            elif is_seamless:
                # Rows are padded to the longest; pad units vocode to an audible buzz that would
                # also inflate `audio_length_s`/RTFx, so cut each row to its reported length.
                outputs = [
                    pred_waveform[i, : min(int(seamless_lengths[i]), pred_waveform.shape[-1])]
                    for i in range(minibatch_size)
                ]
            else:
                outputs = pred_waveform

        # END TIMING
        end_event.record()
        torch.cuda.synchronize(device=args.device)
        runtime = start_event.elapsed_time(end_event) / 1000.0
        # per-sample generation time (RTFx is aggregated over the whole set at the end)
        batch["generation_time_s"] = minibatch_size * [runtime / minibatch_size]

        gen_paths, audio_length_s = [], []
        for audio, sample_id in zip(outputs, batch["id"]):
            # Store the path relative to model_dir (the manifest's dir); write to the full path.
            rel_path = wav_rel_path(dataset_dir_name, sample_id)
            path = os.path.join(model_dir, rel_path)
            if is_vibevoice or not hasattr(processor, "save_audio"):
                # Processors without save_audio, plus VibeVoice (different save_audio signature).
                sf.write(path, audio.reshape(-1).detach().to(torch.float32).cpu().numpy(), sampling_rate)
            else:
                processor.save_audio(audio, saving_path=path)
            gen_paths.append(rel_path)
            audio_length_s.append(len(audio) / sampling_rate)

        batch["gen_audio_filepath"] = gen_paths
        batch["audio_length_s"] = audio_length_s
        batch["references"] = list(batch[args.text_column])  # raw; normalized at scoring time
        if voice_clone:
            # Store the path relative to model_dir (the manifest's dir); the full paths handed to the
            # model as reference audio are the absolute `prompt_audio_paths` written above.
            batch["prompt_audio_filepath"] = [
                os.path.join(dataset_dir_name, "prompts", f"prompt_{sid}.wav") for sid in batch["id"]
            ]
        return batch

    dataset = load_tts_dataset(args, extra_columns=("prompt_text", "prompt_audio") if voice_clone else ())

    probe_out_dir = model_dir  # real results dir, captured before the probe rebinds model_dir
    # ── TTFA probe: latency only, writing nothing but a JSON sidecar ─────────
    # TTFA is per-request, so the probe drives generate_tts() with one-row batches and returns
    # before the main loop. `model_dir`/`dataset_dir_name` are rebound to a temp dir (generate_tts
    # closes over them late-bound) so nothing lands in the results tree. No streaming API, so
    # `first` is None (whole-utterance fallback). 0 = off; N > 0 = N evenly spaced; N < 0 = whole split.
    if args.ttfa_probe != 0:
        # Imported lazily: ttfa_probe is only injected by submit_ttfa_jobs.sh.
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

    # Only batches with a sample missing from the manifest are (re)generated.
    starts = pending_batches(dataset, args.batch_size, done_entries, dataset_dir_name)
    warm_up(generate_tts, dataset, starts, args.batch_size, args.warmup_steps)

    # ── Main loop: TTS → write JSONL, one batch at a time ───────────────────
    for batch_start in tqdm(starts, desc="Generating"):
        batch = generate_tts(dataset[batch_start : batch_start + args.batch_size])
        for i, (afp, alen, gtime, ref) in enumerate(zip(
            batch["gen_audio_filepath"], batch["audio_length_s"], batch["generation_time_s"], batch["references"],
        )):
            write_entry(manifest_file, manifest_entry(
                afp, alen, gtime, ref,
                prompt_audio_filepath=batch["prompt_audio_filepath"][i] if voice_clone else None,
            ))
        manifest_file.flush()

    manifest_file.close()
    print("Results saved at path:", os.path.abspath(manifest_path))
    print_next_steps(voice_clone)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="TTS model id, loadable with 🤗 Transformers.")
    add_common_args(parser, batch_size=32, voice_clone=False)
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language of the text being synthesized, as an ISO 639-1 code ('en', 'zh', 'ru'). "
             "Used by SeamlessM4T (mapped via SEAMLESS_LANGUAGE_CODES).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Max tokens for TTS generation.")
    parser.add_argument("--torch_compile", type=str, default=None, help="torch.compile mode, e.g. 'max-autotune'.")
    parser.add_argument("--compile_fullgraph", action="store_true", help="Full-graph compilation.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype, e.g. 'bfloat16'.")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", help="Attention impl ('sdpa'/'eager'/...).")
    parser.add_argument("--revision", type=str, default=None, help="TTS model revision.")

    # ── SpeechT5-specific (ignored for other models) ─────────────────────
    parser.add_argument(
        "--speecht5_vocoder", type=str, default="microsoft/speecht5_hifigan", help="HiFi-GAN vocoder for SpeechT5."
    )
    parser.add_argument(
        "--speecht5_speaker_embeddings",
        type=str,
        default="https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
        help="Parquet of x-vector speaker embeddings for SpeechT5.",
    )
    parser.add_argument(
        "--speecht5_speaker_index", type=int, default=7306, help="Row index of the speaker x-vector to use."
    )

    # ── Qwen3-Omni-specific (ignored for other models) ────────────────────
    parser.add_argument(
        "--qwen_omni_speaker",
        type=str,
        default="aiden",
        choices=["ethan", "chelsie", "aiden"],
        help="Talker voice for Qwen3-Omni's audio output. Aiden not available for Qwen2.5-Omni (only Ethan/Chelsie).",
    )

    args = parser.parse_args()

    print("*" * 100)
    print(f"Evaluating TTS model {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
