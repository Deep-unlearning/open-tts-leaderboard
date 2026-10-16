"""Local SGLang Omni client for the released Fish Audio Drama 3 checkpoint.

Protocol source: sglang-omni/serve/openai_api.py and
models/fishaudio_drama3/speech_request.py at SERVER_REVISION below.
Only Python's standard library is needed for offline contract tests.
"""

import base64
import binascii
import io
import json
import math
import time
import urllib.error
import urllib.request
import wave
from dataclasses import asdict, dataclass

MODEL_ID = "fishaudio/drama-3"
MODEL_REVISION = "e0a9218aa48888270f06ef1ed9ea67622c0e17d2"
SERVER_REVISION = "12f65fa60fbb5217ad47f58b2e58b18092c818a1"
CODEC_ID = "fishaudio/s2-pro"
CODEC_REVISION = "1de9996b6be38b745688de084d87a5633f714e4e"
OUTPUT_HANDLING_POLICY = "retain_valid_audio_on_stop_or_length_v1"


@dataclass(frozen=True)
class SynthesisResult:
    audio: bytes
    finish_reason: str

    @property
    def generation_token_limit_reached(self):
        return self.finish_reason == "length"


@dataclass(frozen=True)
class StreamResult:
    duration: float
    first_audio_timestamp: float
    response_chunks: int
    finish_reason: str | None
    early_stopped: bool = False

    @property
    def generation_token_limit_reached(self):
        # The terminal reason is deliberately not observed after a first-frame abort.
        return None if self.early_stopped else self.finish_reason == "length"


def validate_finish_reason(reason):
    if reason not in ("stop", "length"):
        raise RuntimeError(f"Drama 3 returned a missing or unsupported terminal reason: {reason!r}")
    return reason


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 30
    max_new_tokens: int = 2048

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 1 <= self.top_k <= 2147483647 or not 1 <= self.max_new_tokens <= 2048:
            raise ValueError("Expected positive int32 top_k and max_new_tokens within the server's 2048 cap")


def speech_payload(text, reference=None, *, sampling=None, streaming=False):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Synthesis needs nonempty text")
    payload = {"model": MODEL_ID, "input": text, **asdict(sampling or Sampling())}
    if reference is not None:
        payload["references"] = reference
    # Drama 3 rejects seed, language and instructions. Do not inherit the S2/Higgs fields.
    if streaming:
        payload.update(stream=True, response_format="pcm", stream_format="sse")
    else:
        payload.update(stream=False, response_format="wav")
    return payload


def wav_duration(data):
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            frames, rate = audio.getnframes(), audio.getframerate()
            expected = frames * audio.getnchannels() * audio.getsampwidth()
            if frames <= 0 or rate <= 0 or len(audio.readframes(frames)) != expected:
                raise ValueError("Empty or truncated WAV response")
            return frames / rate
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid PCM WAV response") from exc


class DramaClient:
    def __init__(self, host="127.0.0.1", port=8000, timeout=300, sampling=None):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("This benchmark expects a local Drama 3 server")
        if not 1 <= port <= 65535 or timeout <= 0:
            raise ValueError("Invalid server port or request timeout")
        self.base_url = f"http://{'[::1]' if host == '::1' else host}:{port}"
        self.timeout = timeout
        self.sampling = sampling or Sampling()
        # Never send loopback benchmark references through an HTTP proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _open(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            return self.opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise RuntimeError(f"Drama 3 HTTP request failed ({status}); no retry was sent") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError("Drama 3 server connection failed; no retry was sent") from None

    def wait_for_server(self, timeout=1200):
        deadline = time.monotonic() + timeout
        while True:
            try:
                with self._open("/health") as response:
                    ready = response.status == 200
            except RuntimeError:
                ready = False
            if ready:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Drama 3 server did not become healthy before timeout")
            time.sleep(3)
        with self._open("/v1/models") as response:
            models = json.load(response)
        if [item.get("id") for item in models.get("data", [])] != [MODEL_ID]:
            raise RuntimeError("Server model identity is not fishaudio/drama-3")

    def synthesize(self, text, reference=None):
        """Compatibility API returning audio bytes without discarding capped outputs."""
        return self.synthesize_result(text, reference).audio

    def synthesize_result(self, text, reference=None):
        """Keep one valid response, including explicit token-limit termination.

        A length termination is a model outcome, not a transport failure. Return its
        audio once and carry the reason with it so callers can score and flag it.
        """
        payload = speech_payload(text, reference, sampling=self.sampling)
        with self._open("/v1/audio/speech", payload) as response:
            reason = validate_finish_reason(response.headers.get("X-Finish-Reason"))
            data = response.read()
        wav_duration(data)
        return SynthesisResult(data, reason)

    def stream(self, text, reference=None, early_stop=False):
        """Compatibility API returning duration, first timestamp and event count."""
        result = self.stream_result(text, reference, early_stop=early_stop)
        return result.duration, result.first_audio_timestamp, result.response_chunks

    def stream_result(self, text, reference=None, early_stop=False):
        """Return audio measurements and an observed stop/length terminal reason.

        The pinned server's SSE endpoint provides an explicit terminal reason. A raw
        PCM EOF alone cannot distinguish a completed utterance from a failed stream.
        An intentional first-frame abort leaves the terminal reason unknown.
        There are no fallback requests or automatic retries.
        """
        payload = speech_payload(text, reference, sampling=self.sampling, streaming=True)
        with self._open("/v1/audio/speech", payload) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise RuntimeError("Expected the pinned SGLang Omni SSE speech protocol")
            try:
                rate = int(response.headers["X-Sample-Rate"])
                channels = int(response.headers["X-Channels"])
                bits = int(response.headers["X-Bit-Depth"])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError("Missing PCM format headers") from None
            if rate <= 0 or channels != 1 or bits != 16:
                raise RuntimeError("Expected positive-rate mono PCM16")
            pending = bytearray()
            data_lines = []
            total = chunks = 0
            first = None
            # read1 returns available bytes without waiting to fill a large read buffer.
            while True:
                block = response.read1(65536)
                if not block:
                    break
                pending.extend(block)
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    line = line.rstrip(b"\r")
                    if line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip(b" "))
                    elif not line and data_lines:
                        try:
                            event = json.loads(b"\n".join(data_lines))
                        except (ValueError, UnicodeDecodeError):
                            raise RuntimeError("Malformed speech stream event") from None
                        data_lines = []
                        if not isinstance(event, dict):
                            raise RuntimeError("Malformed speech stream event")
                        if event.get("type") == "speech.audio.delta":
                            try:
                                chunk = base64.b64decode(event["audio"], validate=True)
                            except (KeyError, ValueError, TypeError, binascii.Error):
                                raise RuntimeError("Invalid base64 speech audio") from None
                            if chunk:
                                chunks += 1
                                total += len(chunk)
                            if first is None and total >= 2:
                                first = time.perf_counter()
                                if early_stop:
                                    return StreamResult((total // 2) / rate, first, chunks, None, True)
                        elif event.get("type") == "speech.audio.done":
                            reason = validate_finish_reason(event.get("finish_reason"))
                            if not total or total % 2:
                                raise RuntimeError("Empty or incomplete PCM speech stream")
                            return StreamResult(total / 2 / rate, first, chunks, reason)
                        elif event.get("type") == "error":
                            raise RuntimeError("Drama 3 speech stream returned an error")
            raise RuntimeError("Speech stream ended without a completed terminal event")
