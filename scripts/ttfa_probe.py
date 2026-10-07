"""
Shared time-to-first-audio (TTFA) probe for the Open TTS Leaderboard backends.

TTFA answers "how long after the request does the caller have audio it can start playing?".
It is measured as:

    ttfa = (first_audio_timestamp - request_start)   when the backend streams
           (generation_finished    - request_start)   when it does not

A model that returns a whole utterance cannot start playback until generation ends, so the
fallback is the honest number for it. The sidecar records `streaming` so the two clocks (which
differ by ~100x) are never silently mixed.

The probe writes ONLY a JSON sidecar (no wavs, no manifest), so it is safe to run against a
results tree that already holds completed evals.

Backends integrate by building a list of jobs and one `gen` callable; see `run_probe`.

IMPORTANT — batch size 1. TTFA is a per-request latency, so `gen` must synthesize exactly one
utterance per call even when the backend's normal eval path batches. A minibatch of N makes
"time to first audio" meaningless for any single request in it.
"""

import inspect
import json
import os
import time

# Rows excluded from the headline summary: the first generations pay residual warm-up and, on
# backends that encode a reference clip, one-off kernel selection per first-seen audio shape.
WARMUP_DISCARD = 3

# Default probe size: the whole split (500 rows for CV3-Eval zero_shot/en). The median settles in
# a few dozen rows, but a p95 needs the full split: ~25 rows above it, not the 2-3 of a 50-row run.
DEFAULT_SAMPLES = -1

# A first-chunk timestamp is NOT sufficient evidence of streaming: a buffered HTTP response can
# arrive in several chunks that all land at the end. Real streaming means first audio arrives
# meaningfully before generation ends, so the verdict is taken from ttfa/gen, not the plumbing.
STREAMING_MAX_FRAC = 0.95

# Rows generated to completion (for ttfa_frac_of_gen and batch-1 throughput) before the probe
# starts aborting after first audio to buy a cheap p90/p95 tail. Only backends whose gen() accepts
# an `early_stop` argument can abort; blocking ones are always full-generation.
FULL_GEN_SAMPLES = 50


def resolve_n(n, total):
    """Number of samples to probe, given the flag and the split size; negative means the whole split."""
    return total if n < 0 else min(n, total)


def sample_indices(n, total):
    """Which rows to probe: evenly spaced across the split, NOT the first n.

    Some splits (e.g. CV3-Eval zero_shot/en) are ordered short-first, so a prefix would understate
    batch-1 RTFx. Deterministic, so every model is probed on the same utterances.
    """
    if n < 0 or n >= total:
        return list(range(total))
    if n == 0:
        return []
    # Float step so the picks stay spread over the whole split when total % n != 0.
    step = total / n
    return [min(total - 1, int(i * step)) for i in range(n)]


def _summary(values):
    """p50/p90/p95/min/max over the non-None values, or None if there are none."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None

    def pct(q):
        return values[min(len(values) - 1, int(round(q / 100.0 * (len(values) - 1))))]

    return {
        "n": len(values),
        "p50": round(pct(50), 2),
        "p90": round(pct(90), 2),
        "p95": round(pct(95), 2),
        "min": round(values[0], 2),
        "max": round(values[-1], 2),
    }


def probe_filename(model_safe, dataset_safe, dataset_config, split, mode_suffix=""):
    """Sidecar name, mirroring the manifest convention so the two sort together."""
    config = str(dataset_config).replace("/", "-")
    return f"TTFA_MODEL_{model_safe}_DATASET_{dataset_safe}_{config}_{split}{mode_suffix}.json"


def run_probe(gen, jobs, *, model_id, dataset_path, dataset_config, split, out_dir,
              mode_suffix="", sample_rate=None, n=DEFAULT_SAMPLES, extra=None, verbose=True):
    """Measure TTFA over the first `n` jobs and write a JSON sidecar. Returns the payload.

    `gen(job)` — or `gen(job, early_stop)` for a backend that can abort once it has the first
    audio — must synthesize ONE utterance and return either:
        (audio_1d, first_ts)             — `sample_rate` then comes from this function's argument
        (audio_1d, sample_rate, first_ts)

    where `first_ts` is a `time.perf_counter()` reading taken the moment the first audio became
    available, or None if the backend has no streaming API (the whole-utterance fallback above).
    `audio_1d` is any float array-like (only its length is used, for the audio duration), or a
    plain float/int giving that duration in SECONDS directly.

    `jobs` is whatever `gen` accepts — this function only slices and counts it. `n` follows
    resolve_n(): negative means every job supplied.
    """
    total = resolve_n(n, len(jobs))
    # A gen() taking a second parameter can stop as soon as it has the first audio.
    supports_early = len(inspect.signature(gen).parameters) >= 2
    rows = []
    for index in range(total):
        early = supports_early and index >= FULL_GEN_SAMPLES
        start = time.perf_counter()
        result = gen(jobs[index], early) if supports_early else gen(jobs[index])
        end = time.perf_counter()

        if len(result) == 3:
            audio, row_sample_rate, first = result
        else:
            audio, first = result
            row_sample_rate = sample_rate
        if not row_sample_rate and not isinstance(audio, (int, float)):
            raise ValueError("sample_rate must be provided by gen() or passed to run_probe()")

        reported = first is not None
        audio_s = float(audio) if isinstance(audio, (int, float)) else len(audio) / float(row_sample_rate)
        gen_s = end - start
        ttfa_ms = ((first if reported else end) - start) * 1000.0
        rows.append({
            "index": index,
            "ttfa_ms": ttfa_ms,
            # Did the backend hand back a first-chunk timestamp at all...
            "first_reported": reported,
            # False => generation was aborted once the first audio arrived, so gen_s/audio_s are
            # partial and ttfa_frac_of_gen is undefined. Only ttfa_ms is meaningful on these rows.
            "full_generation": not early,
            # ...and did that first audio actually arrive early? 1.00 means it landed only when
            # generation finished, i.e. nothing was really incremental. Full rows only.
            "ttfa_frac_of_gen": (ttfa_ms / 1000.0 / gen_s) if (gen_s and not early) else None,
            "gen_s": gen_s,
            "audio_s": audio_s,
        })
        if verbose:
            r = rows[-1]
            frac = r["ttfa_frac_of_gen"]
            if early:
                kind = "early-stop"
            else:
                kind = "stream" if (reported and frac is not None and frac < STREAMING_MAX_FRAC) else "whole-utterance"
            # frac is None on early-stopped rows (and when gen_s is 0), so it cannot use a float format.
            frac_s = f"{frac:.2f}" if frac is not None else " n/a"
            print(f"  [{index + 1}/{total}] ttfa {r['ttfa_ms']:8.1f} ms ({kind})  "
                  f"gen {r['gen_s']:5.2f}s  ttfa/gen {frac_s}  audio {r['audio_s']:5.2f}s", flush=True)

    if not rows:
        raise ValueError("TTFA probe ran zero samples — is the dataset split empty?")

    warm = rows[WARMUP_DISCARD:] or rows
    # Diagnostics that need a COMPLETE generation come only from the full rows; folding an
    # early-aborted row into gen_s or rtfx would corrupt the very numbers this split preserves.
    warm_full = [r for r in warm if r["full_generation"]]
    # Verdict from the measurements, not from whether a timestamp was handed back.
    fracs = [r["ttfa_frac_of_gen"] for r in warm_full if r["ttfa_frac_of_gen"] is not None]
    median_frac = sorted(fracs)[len(fracs) // 2] if fracs else None
    any_streaming = (
        any(r["first_reported"] for r in warm_full)
        and median_frac is not None
        and median_frac < STREAMING_MAX_FRAC
    )
    audio_s = sum(r["audio_s"] for r in warm_full)
    gen_s = sum(r["gen_s"] for r in warm_full)
    payload = {
        "model_id": model_id,
        "dataset_path": dataset_path,
        "dataset": dataset_config,
        "split": split,
        "mode_suffix": mode_suffix,
        # TTFA is a single-request latency; batching would make it meaningless (see the module docstring).
        "batch_size": 1,
        "n_samples": len(rows),
        "warmup_excluded": WARMUP_DISCARD if len(rows) > WARMUP_DISCARD else 0,
        # False => first audio is not available before generation ends, so these values are the
        # whole-utterance clock. Do NOT compare them against real streaming values naively.
        "streaming": any_streaming,
        # Median ttfa/gen over the summarised rows. ~1.0 is the signature of a buffered path,
        # however many chunks its transport happened to split the response into.
        "ttfa_frac_of_gen": round(median_frac, 3) if median_frac is not None else None,
        # True when a backend DID report first-chunk timestamps that turned out not to be early:
        # the code path looks streaming but the server is not. Worth investigating, not trusting.
        "first_reported_but_not_early": bool(
            any(r["first_reported"] for r in warm_full) and not any_streaming
        ),
        # TTFA over EVERY warm row — that is the point of early-aborting the tail.
        "ttfa_ms": _summary([r["ttfa_ms"] for r in warm]),
        # Throughput/diagnostics over the fully-generated rows only.
        "gen_s": _summary([r["gen_s"] for r in warm_full]),
        "full_generation_rows": len(warm_full),
        "early_stopped_rows": len(warm) - len(warm_full),
        # NOT the leaderboard RTFx: this is throughput at batch size 1, whereas the eval measures
        # RTFx at each backend's real batch size. The two can differ by an order of magnitude.
        "rtfx_batch1": round(audio_s / gen_s, 2) if gen_s else None,
        "samples": rows,
    }
    if extra:
        payload.update(extra)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, probe_filename(
        model_id.replace("/", "-"), dataset_path.replace("/", "-"),
        dataset_config, split, mode_suffix))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)

    if verbose and payload["early_stopped_rows"]:
        print(f"  ({payload['full_generation_rows']} rows generated fully for the diagnostics, "
              f"{payload['early_stopped_rows']} aborted after first audio)")
    if any_streaming:
        kind = f"streaming, first audio at {median_frac:.0%} of generation"
    elif payload["first_reported_but_not_early"]:
        # Either a single chunk was emitted (first audio IS last audio) or a buffered response
        # arrived in chunks only at the end; the backend's chunk count in the sidecar tells which.
        kind = (f"NOT streaming: first audio arrived at {median_frac:.0%} of generation — "
                "either a single chunk was emitted, or chunks only landed once it had finished")
    else:
        kind = "whole-utterance (no streaming API)"
    # One greppable line carrying both headline numbers, so a caller driving many probes can
    # report them without re-reading the sidecar. submit_ttfa_jobs.sh matches on "TTFA SUMMARY".
    t = payload["ttfa_ms"] or {}
    rtfx = payload["rtfx_batch1"]
    print(f"\nTTFA SUMMARY [{kind}] n={t.get('n', 0)} (warm-up excluded)"
          f" | TTFA p50 {t.get('p50', float('nan')):.1f} ms, p90 {t.get('p90', float('nan')):.1f} ms,"
          f" p95 {t.get('p95', float('nan')):.1f} ms"
          f" | RTFx(batch1) {rtfx if rtfx is not None else 'n/a'}")
    print("TTFA probe written to:", os.path.abspath(path))
    return payload
