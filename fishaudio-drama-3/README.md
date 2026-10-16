# Fish Audio Drama 3

This backend evaluates the open checkpoint **[`fishaudio/drama-3`](https://huggingface.co/fishaudio/drama-3)**
through a local SGLang Omni server. It supports default-voice synthesis, per-sample
voice cloning, concurrent offline generation and streaming TTFA.

It uses Drama 3's `FishOmniModel` MoE backbone and codebook-MTP decoder. Existing
`fishaudio/s2-pro`, hosted `fish/s2-pro` and hosted `fish/s2.1-pro` results are different
models. The pinned S2-Pro **codec dependency** below does not change the Drama 3
backbone's identity. The int4 Drama checkpoint is a separate model and is rejected.

Public references:

- [Official model card](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/README.md)
- [Official SGLang Omni cookbook](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)
- [Pinned request preprocessor](https://github.com/sgl-project/sglang-omni/blob/12f65fa60fbb5217ad47f58b2e58b18092c818a1/sglang_omni/models/fishaudio_drama3/speech_request.py)
- [Pinned speech protocol](https://github.com/sgl-project/sglang-omni/blob/12f65fa60fbb5217ad47f58b2e58b18092c818a1/sglang_omni/serve/openai_api.py)
- [Pinned license](https://huggingface.co/fishaudio/drama-3/blob/e0a9218aa48888270f06ef1ed9ea67622c0e17d2/LICENSE)

## Completed evaluation

The full evaluation completed 58 jobs and scored 15,216 samples: 7,608 each for
default voice and voice cloning, covering both Seed-TTS languages and all nine
CV3 languages. Generation and TTFA used H200; ASR and speaker similarity used L4.
The six result CSVs are proposed in
[results PR #58](https://huggingface.co/datasets/hf-audio/tts_leaderboard_results/discussions/58)
(the results dataset is private).

| Metric | Default voice | Voice clone |
| --- | ---: | ---: |
| Seed English WER (%) | 0.95 | 1.55 |
| Seed Chinese WER (%) | 0.96 | 0.84 |
| CV3 mean WER (%) | 3.55 | 5.40 |
| CV3 mean RTFx | 7.15 | 6.63 |

Warm H200 loopback/server TTFA was 91.89 ms p50 and 94.73 ms p95, with batch-1
RTFx 7.54. The benchmark excludes the first three requests, leaving 497 warm TTFA
samples; the first excluded cold request took 5.59 seconds. All 28 valid outputs
that reached the fixed 2,048-token cap were retained and scored, without rerolls
or output filtering. The recorded pins reproduce the configuration; generation
remains stochastic.

The evaluated implementation was `3970a19f503657d3770ef9de32cd7741017a8253`.
This backend PR is based on upstream `main` and leaves the shared normalizer
unchanged. The scores use the audited normalizer revisions below; the related
normalization change is reviewed separately in
[PR #12](https://github.com/huggingface/open-tts-leaderboard/pull/12).

## Revisions and request settings

| Component | Revision |
| --- | --- |
| Original evaluated implementation base | `e2a95394d6a14b6a103deb7eb540bcd5c1bf6826` |
| Normalizer tree at that base | `ccf2fe24596d17b6cb45e49ead3eef336bf914a1` |
| Evaluated normalizer source revision | `99e259cd461e7821989cc5dfbd998ba3f47485d4` |
| Same normalizer file/aggregate in PR #12 | `8ade67e276659f263d9d094747d8e32cd867d0a5` |
| Validated normalizer tree | `d028cc45cdbd2d98c067969b27a32c0eb02056dd` |
| Validated `normalizer.py` SHA256 | `5ba67c837110c53b1e9bbc82984a7ca4a35e9a4ed52f158730447a43f802d032` |
| Drama 3 model | `e0a9218aa48888270f06ef1ed9ea67622c0e17d2` |
| SGLang Omni | `12f65fa60fbb5217ad47f58b2e58b18092c818a1` |
| Codec (`fishaudio/s2-pro`, pinned by upstream) | `1de9996b6be38b745688de084d87a5633f714e4e` |

Defaults are `temperature=1.0`, `top_p=0.9`, `top_k=30` and
`max_new_tokens=2048`, with one request in flight. Model startup and two warm-up
batches are excluded from offline RTFx. Concurrent generation records batch wall
time divided by the number of samples. Valid audio with an explicit `stop` or
`length` finish reason is retained from its single request and scored. A `length`
response is flagged as hitting the token cap in the row and run receipt; it is
never retried or removed from headline WER/SIM. Missing or unknown terminal states,
server errors and malformed audio still fail the request. No silence placeholders
are produced. This follows the repository's LFM2 and Higgs behavior of scoring
capped output rather than selecting only successful natural stops.

Drama 3's pinned preprocessor **rejects request-level `seed`**, `language` and
`instructions`; these fields are omitted. The inherited common `--seed` argument
does not seed this remote model. Recorded revisions and sampling settings reproduce
the experiment configuration, but do not guarantee identical stochastic audio.
Target text, including brackets and parentheses, is sent unchanged.

## Offline validation

From the repository root, using Python 3.10 or later:

```bash
python3 -m unittest discover -s fishaudio-drama-3/tests -v
bash -n fishaudio-drama-3/submit_jobs.sh fishaudio-drama-3/run_job.sh
python3 fishaudio-drama-3/run_eval.py --help
bash fishaudio-drama-3/submit_jobs.sh
```

Tests use synthetic local audio and mocked network/runtime dependencies. They do
not download weights, call a paid API, submit jobs or measure model quality.

## Runtime image prerequisite

Use an explicitly selected public Docker Space image with a compatible NVIDIA
CUDA environment and the pinned SGLang source installed with its `fishaudio` extra,
or the immutable base and ephemeral bootstrap below.
Follow upstream's [installation instructions](https://github.com/sgl-project/sglang-omni/blob/12f65fa60fbb5217ad47f58b2e58b18092c818a1/docs/get_started/installation.md).
Retain the source checkout's `.git` directory. Within the chosen compatible image,
the source-install steps are:

```bash
git clone https://github.com/sgl-project/sglang-omni.git /opt/sglang-omni
git -C /opt/sglang-omni checkout 12f65fa60fbb5217ad47f58b2e58b18092c818a1
cd /opt/sglang-omni
uv pip install --prerelease=allow -e '.[fishaudio]'
```

Use Python 3.12 for the GPU environment. The upstream dependencies include CUDA
extensions and audio decoding libraries; the commands above are source-install
steps, not a complete CUDA base-image recipe. Record the resolved environment and
image digest before evaluation. The completed evaluation validated the immutable
upstream image plus ephemeral bootstrap below; no new image was published.
Existing S2-Pro or Higgs images must not be assumed compatible.
Publishing or modifying a Space requires separate approval.

An alternative avoids image publication by installing the runtime inside each
ephemeral single-H200 Job. Select the upstream base's immutable Linux amd64 image:

```bash
TTS_IMAGE=lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75 \
BOOTSTRAP_RUNTIME=true \
RESULTS_BUCKET=OWNER/PRIVATE_BUCKET RESULTS_PREFIX=fish-drama-3/validation \
DATASET_SCOPE=seed_en MAX_EVAL_SAMPLES=8 JOB_TIMEOUT=30m \
bash fishaudio-drama-3/submit_jobs.sh
```

This is a preview. `TTS_IMAGE` and `TTS_SPACE` are mutually exclusive. Bootstrap
verifies Python 3.12, CUDA 13.0.3, Torch 2.13.0, SGLang 0.5.21 and FlashInfer
0.6.18, installs the clean pinned Omni source and Fish extras, and pins
`datasets==3.6.0` for the evaluation audio interface. It saves private runtime
receipts and dependency versions beneath the results prefix. It supports the
pinned single-node GPU0 CUDA IPC/SHM path; it does not establish cross-node UCX
compatibility. Installation, downloads and validation share the Job timeout.
The completed H200 evaluation used this bootstrap configuration.

`run_job.sh` checks that the imported server package belongs to the clean retained
Git checkout at the pinned revision before downloading the pinned model snapshot.
It starts that same Python interpreter's server with the snapshot path and verifies
the served model ID through `/v1/models`. `PYTHON_BIN` can select an image's virtual
environment, for example `/opt/omni/bin/python`.

## Private prepared reference inputs

The launcher can read prepared local Parquet repositories from an existing private
bucket, avoiding publication of a dataset repository. Set `INPUTS_BUCKET` and
optional `INPUTS_PREFIX`; the launcher mounts the subfolder read-only at `/inputs`
and passes `--dataset_local_root /inputs`. Retain canonical dataset IDs in
`DATASET_PATH=OWNER/seed_tts_eval` and `CV3_EVAL_PATH=OWNER/cv3_eval` for output names.
The receipt explicitly records the mounted input source and provenance, rather
than claiming those IDs were downloaded as Hub repositories.

The mount must contain `seed_tts_eval/` and `cv3_eval/` load-dataset-compatible
Parquet directories with embedded prompt bytes, README config mappings, and
`input_provenance.json` format version 1. Each dataset entry records `source`,
`readme_sha256` and `splits`; each `config/language` split records `rows`,
`parquet_file`, `parquet_sha256` and `ordered_id_text_sha256`. The last field hashes
each ordered JSON `[str(id), text]` record plus a newline, with `ensure_ascii=False`.
Require full canonical counts: Seed EN 1,088 / ZH 2,020; CV3 nine languages 500 each.
The client verifies config and selected Parquet checksums, row counts and full-run
ID/text order. Smoke subsampling happens after source verification. References and
private preparation metadata must stay outside Git and public result locations.

## Proposed bounded smoke test

The following is a **preview** and submits nothing:

```bash
TTS_SPACE=OWNER/APPROVED_DRAMA_IMAGE \
RESULTS_BUCKET=OWNER/PRIVATE_DRAMA_SMOKE_BUCKET \
RESULTS_PREFIX=fish-drama-3/validation \
DATASET_NAMESPACE=OWNER \
DATASET_SCOPE=seed_en MAX_EVAL_SAMPLES=8 \
JOB_TIMEOUT=30m BATCH_SIZE=1 WARMUP_STEPS=2 \
bash fishaudio-drama-3/submit_jobs.sh
```

After approval of that exact run, append `--execute` to the same command. This
requests **one H200 job capped at 30 minutes**, eight measured samples and two
warm-up requests, default voice only. It starts no ASR, SIM or scoring jobs and
has no automatic retry. `VOICE_CLONE=true` is a separate run requiring approval
for one additional job. A generation timeout can include model download/startup;
30 minutes is a spending bound, not a promise the run will finish.

Before execution, verify existing jobs and the selected bucket to avoid duplicate
work, confirm current HF hardware pricing and the billing namespace, and approve
the applicable billing arrangement. Keep smoke results in their own private bucket
or a separate `RESULTS_PREFIX` inside an existing private bucket. The prefix is a
relative subfolder, and the launcher rejects traversal and shell metacharacters.
No credentials
should appear in commands or tracked files; the launcher passes the existing
`HF_TOKEN` to the job through HF's secret mechanism.

The launcher selects exactly one dataset by default. `DATASET_SCOPE=cv3_en` selects
only CV3 English. `DATASET_SCOPE=full` selects two Seed-TTS splits and nine CV3
splits, serially. This avoids the common `ONLY_LANGS=en` behavior of starting both
English datasets at once.

## Full evaluation, after smoke review and compute approval

The completed evaluation used the validated normalizer revision above, verified
by its exact file and aggregate SHA256. This backend PR does not include that
normalizer change. Merging PR #12 supplies the exact evaluated normalizer hashes.
Before rescoring or using `prepare_submission.py`, use that audited normalizer;
a later revision requires review and revalidation of the helper's allowlisted
aggregate as well as its caller-supplied file hash. Runtime
and smoke validation have completed for the recorded configuration; a new run
still needs an explicit compute bound and its own results prefix.

Preview one mode's full evaluation:

```bash
TTS_IMAGE=lmsysorg/sglang@sha256:2dbe4c7f53230b09aaa8fa41891c13c51d9d2e253f486a70822aaf08783fff75 \
BOOTSTRAP_RUNTIME=true \
RESULTS_BUCKET=OWNER/PRIVATE_DRAMA_FULL_BUCKET \
RESULTS_PREFIX=fish-drama-3/full \
DATASET_NAMESPACE=OWNER \
DATASET_SCOPE=full MAX_EVAL_SAMPLES=-1 \
JOB_TIMEOUT=2h BATCH_SIZE=1 VOICE_CLONE=false \
bash fishaudio-drama-3/submit_jobs.sh
```

This plan has 11 serial H200 jobs, each capped at two hours: at most 22 allocated
GPU hours per mode. Cloning adds an 11-job plan with `VOICE_CLONE=true`.
Authorize the namespace, billing arrangement and total cap before
adding `--execute`. A later concurrency comparison must use a separate results
location and record its batch size.

Generation does not import the normalizer. It writes a manifest plus `.run.json`
receipt with exact model/server/codec revisions, sampling settings, corpus hash,
concurrency, expected/completed rows and `metrics_ready=false`. This is generation
coverage, not a scored leaderboard result. Results already present locally require
`--resume` with identical settings; the runner verifies completed audio and skips
completed rows individually, including partially completed batches.
The Job exit handler copies available output to the private mounted results
folder on success or failure. Failed generation receipts are marked `failed` and
retain their completed-row count; they cannot pass full-split submission checks.
Bucket resume remains a manual recovery task: the launcher refuses existing
targets rather than overwriting or silently rerunning their completed rows.

HF jobs receive the launcher's source-checkout revision and normalizer file/tree
hashes explicitly, since the injected job environment has no repository `.git`
directory. The receipt distinguishes this source provenance from actual scoring.

The launcher does not run the shared scoring pipeline automatically. Once the audit
has established the normalizer revision and all compute stages are approved, use
the existing shared ASR/SIM helpers against the private bucket. For example, this
explicit **single-split** recipe runs ASR and optional cloning SIM; prepare its
current image and cost bounds before executing:

```bash
RESULTS_BUCKET=OWNER/PRIVATE_DRAMA_FULL_BUCKET/fish-drama-3/full \
DATASET_PATH=OWNER/seed_tts_eval VOICE_CLONE=false bash <<'BASH'
set -e
# The shared injectors expect GNU base64 -w0; support macOS as well.
base64() {
  if [[ "${1:-}" == -w0 ]]; then
    shift
    if [[ $# -gt 0 ]]; then command base64 < "$1"; else command base64; fi | tr -d '\r\n'
  else
    command base64 "$@"
  fi
}
source scripts/tts_jobs_common.sh
NAMESPACE_ARG=""
MODEL_SAFE=fishaudio-drama-3
DATASET_SAFE="${DATASET_PATH//\//-}"
MODE_SUFFIX=""
[[ "$VOICE_CLONE" != true ]] || MODE_SUFFIX=_voice_clone
BUCKET_MANIFEST="/results/$MODEL_SAFE/MODEL_${MODEL_SAFE}_DATASET_${DATASET_SAFE}_tts_en${MODE_SUFFIX}.jsonl"
ASR_LANG=en
transcribe_stage
if [[ "$VOICE_CLONE" == true ]]; then
  SIM_BACKEND=wavlm_seed_tts sim_stage
  # Seed-TTS publication also has a legacy xvector SIM column.
  SIM_BACKEND=xvector sim_stage
fi
BASH
```

Shared ASR/SIM helpers use L4 jobs with a four-hour per-job timeout at this base;
the example is one ASR job, plus two SIM jobs for cloning. It does not score WER.
The shared helpers do not read `RESULTS_PREFIX`: include the same subfolder in
their `RESULTS_BUCKET` value as shown above, so their `/results` mount matches
generation. Confirm each remote job reaches `COMPLETED`; older HF CLI versions
can exit successfully after a failed remote job.
For each additional split, set the exact manifest, dataset config and ASR language;
CV3 cloning needs only `wavlm_seed_tts`. Preserve the private dataset locations.
Sync completed manifests and receipts locally using the established HF bucket
workflow; verify full source-split coverage before scoring. Keep generated audio,
references and transcripts out of commits.

## Streaming TTFA

Preview a small probe against CV3 English:

```bash
TTS_SPACE=OWNER/APPROVED_DRAMA_IMAGE \
RESULTS_BUCKET=OWNER/PRIVATE_DRAMA_TTFA_SMOKE_BUCKET \
RESULTS_PREFIX=fish-drama-3/ttfa-validation \
DATASET_NAMESPACE=OWNER DATASET_SCOPE=cv3_en \
MAX_EVAL_SAMPLES=8 TTFA_PROBE=8 BATCH_SIZE=1 \
JOB_TIMEOUT=30m VOICE_CLONE=false \
bash fishaudio-drama-3/submit_jobs.sh
```

After separate approval, `--execute` requests one H200 job capped at 30 minutes.
The probe uses the pinned SSE protocol (`stream=true`, `response_format=pcm`,
`stream_format=sse`), timestamps the first complete PCM frame, and checks the
terminal finish reason on full responses. It has no dialect fallback requests.
Each probe records its finish reason, whether it hit the cap, and whether the
client deliberately stopped after obtaining first audio. A valid capped full
response remains in the latency distribution and is reported as capped.
Cloning probes use temporary references under the allowed results root and clean
them up without altering evaluation references.

For a full CV3 English latency distribution, preview
`MAX_EVAL_SAMPLES=-1 TTFA_PROBE=500 DATASET_SCOPE=cv3_en` with an approved timeout.
The completed evaluation used this 500-request configuration. The eight-sample
probe is not a publication result. The shared probe's initial
warm-up exclusion and first-50-full-generation policy are retained. The launcher
adds `__sglang__h200` to the sidecar filename so the publisher knows its engine and
hardware. TTFA sidecars alone do not supply WER, SIM or full-batch RTFx.

## Local results submission preparation

Use `prepare_submission.py --help` to prepare a local plan after full generation,
ASR and required SIM have completed. It requires the audited normalizer's exact
file SHA256, checks full-split receipts, rejects smoke/partial results and
records a command for the existing `scripts/open_results_pr.py` dry run. It does
not run scoring or make a network request.

```bash
python3 fishaudio-drama-3/prepare_submission.py \
  --results_dir /ABSOLUTE/LOCAL/results/fishaudio-drama-3 \
  --output_dir /ABSOLUTE/LOCAL/drama-submission-standard \
  --normalizer_sha256 5ba67c837110c53b1e9bbc82984a7ca4a35e9a4ed52f158730447a43f802d032
```

Prepare cloning in a separate output directory with `--voice_clone`. The emitted
publisher command uses `--skip_sync` and `--out_dir`, and omits `--open_pr`.
Running that later command scores locally and reads the existing private published
CSVs; it does not create a PR. Review generated CSV differences and record the
scoring revision before requesting publication approval.

### Completed ASR receipts for empty predictions

An empty ASR transcript is a valid deletion error after successful transcription;
the same empty string can also mean the generated audio was never transcribed.
The local submission preflight distinguishes these cases using a completed ASR
receipt from the scorer wrapper, stored privately at
`scoring_receipts/<plain-manifest-stem>.asr.json`. A SIM-fork manifest uses the
plain stem without `_wavlm_seed_tts`.

The shared `transformers/transcribe.py` does not emit these receipts. The completed
evaluation used an external pinned scoring wrapper that wrote them only after
the scorer completed, with this required contract:

| Field | Required value |
| --- | --- |
| `status`, `stage` | `complete`, `asr` |
| `manifest` | Plain manifest filename, including `.jsonl` |
| `rows` | Exact completed row count |
| `asr_model_id` | `Qwen/Qwen3-ASR-1.7B-hf` |
| `asr_model_resolved_sha` | Actual resolved 40-character ASR model revision |
| `source_revision` | Actual 40-character evaluation-source revision |
| `asr_source_sha256`, `source_sha256` | SHA256 of the executed `transformers/transcribe.py` |
| `ordered_prediction_sha256` | Prediction digest described below |

For each manifest row, in manifest order, the digest consumes
`json.dumps([row["audio_filepath"], row["pred_text"]], ensure_ascii=False).encode("utf-8")`
followed by a newline byte. Receipt source hashes must also match the submission
checkout's scorer source. A wrapper must verify successful scorer completion and
bind its actual model/source identities before writing a receipt; synthesizing
one from a manifest alone does not prove that transcription ran. Without this
evidence, the helper refuses empty predictions rather than silently dropping
them or accepting an unscored run. This PR documents the receipt interface and
does not change the shared scorer or ship the external wrapper.

Only the aggregate score summary is included here. Submission CSVs, audio,
references, transcripts and private scoring receipts remain outside this backend.
