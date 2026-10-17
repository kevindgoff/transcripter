"""
Voice Recorder - Real-time Audio Transcription Tool

A CLI tool that records audio from your microphone and transcribes it
to markdown using local Whisper models. Optionally labels who said what
(Speaker 1, Speaker 2, ...) using local speaker diarization.

Every recording is saved next to its transcript, sharing one base name:

    2026-10-16_22-48-48.wav   the audio (streamed to disk while recording)
    2026-10-16_22-48-48.md    the transcript
    2026-10-16_22-48-48.srt   closed captions (with --srt)

Usage:
    voice-record [--output-dir DIR] [--model SIZE] [--diarize] [--speakers N]
                 [--srt] [--no-audio] [--no-ui] [--from-file WAV]

Examples:
    voice-record                          # Record with default settings
    voice-record --output-dir ./notes     # Save to custom directory
    voice-record --model medium           # Use larger model for better accuracy
    voice-record --diarize --speakers 2   # Label two speakers
    voice-record --srt                    # Also write closed captions
    voice-record --from-file meeting.wav  # Transcribe an existing recording
"""

import argparse
import os
import queue
import signal
import struct
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

TARGET_RATE = 16000  # Whisper and the diarizer both expect 16 kHz mono

# Long recordings are transcribed in chunks so memory stays flat no matter how
# long the meeting is. Each chunk is cut at the quietest moment near its end.
CHUNK_SECONDS = 600.0          # target chunk length
CHUNK_THRESHOLD_SECONDS = 900.0  # recordings up to this length go through in one piece
CHUNK_SEARCH_SECONDS = 20.0    # look this far back from the target for a quiet cut point

SIGNAL_PEAK = 0.002  # below this (about -54 dBFS) a block counts as "no signal"

MODEL_DIR = Path(os.environ.get("TRANSCRIPTER_MODEL_DIR", Path.home() / ".cache" / "transcripter" / "models"))
_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_ARCHIVE_URL = f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_MODEL = MODEL_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING_URL = f"{_RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx"  # (sic) upstream typo
EMBEDDING_MODEL = MODEL_DIR / "wespeaker_en_voxceleb_resnet34.onnx"

ProgressFn = Callable[[float], None]  # called with "seconds of audio done so far"


def _no_progress(_done: float) -> None:
    pass


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

@dataclass
class Word:
    start: float
    end: float
    text: str  # may carry a leading space, as emitted by Whisper


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Turn:
    """A span of time attributed to one speaker by the diarizer."""
    start: float
    end: float
    speaker: int  # raw diarizer id, arbitrary


@dataclass
class LabeledWord:
    """A word with its speaker (1-based, by first appearance; None = unlabelled)."""
    speaker: int | None
    start: float
    end: float
    text: str


@dataclass
class Utterance:
    """Consecutive words by one speaker, ready for output."""
    speaker: int  # 1-based, numbered by first appearance
    start: float
    text: str


@dataclass
class Result:
    """Everything produced by transcribing (and optionally diarizing) a recording."""
    body: str
    words: list[LabeledWord]
    diarized: bool
    duration: float  # seconds of audio


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def sec_to_ms(seconds: float) -> int:
    return int(round(seconds * 1000))


def format_clock(seconds: float) -> str:
    """MM:SS, switching to HH:MM:SS from one hour on."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def unique_stem(directory: Path, stem: str, suffixes=(".wav", ".md", ".srt")) -> str:
    """`stem`, or `stem_1`, `stem_2`... so we never overwrite an earlier recording."""
    candidate, n = stem, 1
    while any((directory / f"{candidate}{s}").exists() for s in suffixes):
        candidate = f"{stem}_{n}"
        n += 1
    return candidate


@dataclass
class Outputs:
    """The files belonging to one recording: same directory, same base name."""
    directory: Path
    stem: str

    @property
    def wav(self) -> Path:
        return self.directory / f"{self.stem}.wav"

    @property
    def md(self) -> Path:
        return self.directory / f"{self.stem}.md"

    @property
    def srt(self) -> Path:
        return self.directory / f"{self.stem}.srt"


def write_text_atomic(path: Path, text: str, newline: str | None = None) -> None:
    """Write via temp file + rename so a crash never leaves a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------
# WAV I/O (16-bit PCM)
# --------------------------------------------------------------------------

def _wav_header(data_size: int, sample_rate: int, channels: int) -> bytes:
    data_size = min(data_size, 0xFFFFFFFF - 36)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16,
        b"data", data_size,
    )


class WavWriter:
    """Streams 16-bit PCM to a WAV file as it arrives.

    The size fields in the header are re-written every couple of seconds. If the
    process dies mid-recording (crash, power cut) the file on disk is still a
    valid WAV containing everything up to the last patch -- an hour-long meeting
    is not lost to a late failure.
    """

    PATCH_INTERVAL = 2.0

    def __init__(self, path: Path, sample_rate: int = TARGET_RATE, channels: int = 1):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.data_bytes = 0
        self._f = open(self.path, "wb")
        self._f.write(_wav_header(0, sample_rate, channels))
        self._f.flush()
        self._last_patch = time.monotonic()

    def write(self, pcm: bytes) -> None:
        self._f.write(pcm)
        self.data_bytes += len(pcm)
        if time.monotonic() - self._last_patch >= self.PATCH_INTERVAL:
            self.patch()

    def patch(self) -> None:
        n = min(self.data_bytes, 0xFFFFFFFF - 36)
        self._f.seek(4)
        self._f.write(struct.pack("<I", 36 + n))
        self._f.seek(40)
        self._f.write(struct.pack("<I", n))
        self._f.seek(0, os.SEEK_END)
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except OSError:
            pass
        self._last_patch = time.monotonic()

    def close(self) -> None:
        if self._f is not None:
            self.patch()
            self._f.close()
            self._f = None


class LinearResampler:
    """Streaming linear-interpolation resampler (block by block, no seams).

    Deliberately simple: no anti-alias filter, so downsampling folds a little
    high-frequency energy back in. Fine for speech recognition, and only used
    when a device or input file is not already 16 kHz.
    """

    def __init__(self, src_rate: int, dst_rate: int):
        import numpy as np

        self._np = np
        self.step = src_rate / dst_rate
        self._buf = np.zeros(0, dtype=np.float64)
        self._consumed = 0  # global index of _buf[0] (or of the next input sample when _buf is empty)
        self._skip = 0      # upcoming input samples to drop (when downsampling skips past a block's end)
        self._k = 0         # next output sample index

    def process(self, block):
        np = self._np
        block = np.asarray(block, dtype=np.float64)
        if self._skip:
            d = min(self._skip, len(block))
            block, self._consumed, self._skip = block[d:], self._consumed + d, self._skip - d
        buf = np.concatenate((self._buf, block))
        base = self._consumed
        last_idx = base + len(buf) - 1
        n_out = int(np.ceil(last_idx / self.step)) - self._k
        if len(buf) < 2 or n_out <= 0:
            self._buf = buf
            return np.zeros(0, dtype=np.float64)
        pos = (self._k + np.arange(n_out)) * self.step - base
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0
        out = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac
        self._k += n_out
        keep = max(0, int(np.floor(self._k * self.step - base)))
        if keep >= len(buf):
            # The next sample we need has not arrived yet: drop everything held
            # and skip the remainder when it does.
            self._skip = keep - len(buf)
            self._consumed = base + len(buf)
            self._buf = np.zeros(0, dtype=np.float64)
        else:
            self._buf = buf[keep:]
            self._consumed = base + keep
        return out


def read_wav_as_16k_mono(path: Path):
    """Load a 16-bit PCM WAV as a mono 16 kHz int16 numpy array.

    Handles any channel count (averaged to mono) and any sample rate (linear
    resampling). Reads in blocks so a long file never needs a second full copy.
    """
    import numpy as np

    try:
        w = wave.open(str(path), "rb")
    except (wave.Error, EOFError) as e:
        raise ValueError(f"{path}: not a readable PCM WAV file ({e})")
    with w:
        channels, width, rate, n_frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported (got {width * 8}-bit)")
        resampler = LinearResampler(rate, TARGET_RATE) if rate != TARGET_RATE else None
        out_len = n_frames if resampler is None else int(n_frames * TARGET_RATE / rate) + 2
        out = np.zeros(out_len, dtype=np.int16)
        pos = 0
        block_frames = 1 << 20
        while True:
            raw = w.readframes(block_frames)
            if not raw:
                break
            data = np.frombuffer(raw, dtype=np.int16)
            if channels > 1:
                data = data[: len(data) - len(data) % channels].reshape(-1, channels).mean(axis=1)
            if resampler is not None:
                data = resampler.process(data)
            data = np.clip(np.rint(data), -32768, 32767).astype(np.int16)
            out[pos:pos + len(data)] = data
            pos += len(data)
        return out[:pos]


def write_wav(path: Path, samples, sample_rate: int = TARGET_RATE) -> None:
    w = WavWriter(path, sample_rate, 1)
    try:
        step = 1 << 20
        for i in range(0, len(samples), step):
            w.write(samples[i:i + step].tobytes())
    finally:
        w.close()


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

@dataclass
class LevelSnapshot:
    """What the recorder is hearing right now (for a live display)."""
    elapsed: float       # seconds since recording started
    rms: float           # 0..1 of full scale, over the last ~128 ms
    peak: float          # 0..1
    samples: object      # last ~128 ms as float32 in [-1, 1] (numpy array)
    silent_for: float    # seconds since the signal last rose above SIGNAL_PEAK
    dropouts: int        # audio-device over/underruns reported so far
    clipped: bool


class AudioRecorder:
    """Records mono int16 audio from the default input device into a WAV file.

    The audio callback only enqueues the block; a writer thread streams it to
    disk and tracks levels. Memory use is constant however long the meeting is.
    """

    RING = 2048  # samples kept for the live level/spectrum display

    def __init__(self, wav_path: Path, sample_rate: int = TARGET_RATE, channels: int = 1,
                 on_error: Callable[[], None] | None = None):
        import numpy as np

        self.wav_path = Path(wav_path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_name = ""
        self.error: str | None = None
        self._on_error = on_error
        self._np = np
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stream = None
        self._thread: threading.Thread | None = None
        self._writer: WavWriter | None = None
        self._actual_rate = sample_rate
        self._lock = threading.Lock()
        self._ring = np.zeros(self.RING, dtype=np.int16)
        self._t0 = time.monotonic()
        self._last_signal = self._t0
        self._dropouts = 0
        self._stopped = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd

        device_info = sd.query_devices(kind="input")  # raises if no input device
        self.device_name = device_info["name"]

        def callback(indata, frames, time_info, status):
            if status:
                self._dropouts += 1
            self._q.put(indata.tobytes())

        # Prefer 16 kHz directly; fall back to the device's native rate and
        # convert the finished file once recording stops.
        rates = [self.sample_rate]
        native = int(device_info["default_samplerate"])
        if native != self.sample_rate:
            rates.append(native)

        last_err = None
        for rate in rates:
            try:
                stream = sd.InputStream(samplerate=rate, channels=self.channels, dtype="int16", callback=callback)
                stream.start()
            except sd.PortAudioError as e:
                last_err = e
                continue
            self._stream = stream
            self._actual_rate = rate
            break
        else:
            raise RuntimeError(f"Could not open input stream: {last_err}")

        self._t0 = self._last_signal = time.monotonic()
        self._writer = WavWriter(self.wav_path, self._actual_rate, self.channels)
        self._thread = threading.Thread(target=self._drain, name="wav-writer", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        np = self._np
        failed = False
        while True:
            block = self._q.get()
            if block is None:
                break
            if failed:
                continue  # keep draining so the queue cannot grow without bound
            try:
                self._writer.write(block)
                self._track(np.frombuffer(block, dtype=np.int16))
            except Exception as e:  # disk full, device unplugged filesystem, ...
                failed = True
                self.error = str(e)
                if self._on_error:
                    self._on_error()

    def _track(self, x) -> None:
        np = self._np
        if self.channels > 1:
            x = x[: len(x) - len(x) % self.channels].reshape(-1, self.channels)[:, 0]
        n = len(x)
        if n == 0:
            return
        with self._lock:
            if n >= self.RING:
                self._ring = x[-self.RING:].copy()
            else:
                self._ring = np.concatenate((self._ring[n:], x))
            if int(np.abs(x).max()) / 32768.0 >= SIGNAL_PEAK:
                self._last_signal = time.monotonic()

    def stop(self) -> float:
        """Stop recording and finalise the WAV. Returns seconds recorded."""
        if self._stopped:
            return self.seconds_written()
        self._stopped = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._q.put(None)
        if self._thread is not None:
            self._thread.join()
        seconds = self.seconds_written()
        if self._writer is not None:
            self._writer.close()
        if self._actual_rate != TARGET_RATE and seconds > 0:
            convert_wav_to_16k_mono(self.wav_path)
        return seconds

    # -- live status -------------------------------------------------------

    def seconds_written(self) -> float:
        if self._writer is None:
            return 0.0
        return self._writer.data_bytes / (2 * self.channels * self._actual_rate)

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def snapshot(self) -> LevelSnapshot:
        np = self._np
        with self._lock:
            ring = self._ring.copy()
            last_signal = self._last_signal
        x = ring.astype(np.float32) / 32768.0
        peak = float(np.abs(x).max()) if len(x) else 0.0
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        return LevelSnapshot(
            elapsed=self.elapsed(), rms=rms, peak=peak, samples=x,
            silent_for=time.monotonic() - last_signal, dropouts=self._dropouts,
            clipped=peak >= 0.99,
        )


def convert_wav_to_16k_mono(path: Path) -> None:
    """Rewrite `path` in place as 16 kHz mono (used when the mic refused 16 kHz)."""
    samples = read_wav_as_16k_mono(path)
    tmp = path.with_suffix(".part")
    write_wav(tmp, samples)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def int16_to_float32(samples):
    import numpy as np

    return np.asarray(samples).astype(np.float32) / 32768.0


def plan_chunks(samples, sample_rate: int = TARGET_RATE, chunk_s: float = CHUNK_SECONDS,
                search_s: float = CHUNK_SEARCH_SECONDS,
                threshold_s: float = CHUNK_THRESHOLD_SECONDS) -> list[tuple[int, int]]:
    """Split a long recording into (start, end) sample ranges for transcription.

    Short recordings are returned whole. Long ones are cut near every `chunk_s`
    seconds, at the quietest 50 ms frame within the preceding `search_s` seconds,
    so a cut almost never lands in the middle of a word.
    """
    import numpy as np

    n = len(samples)
    if n <= threshold_s * sample_rate:
        return [(0, n)]

    frame = int(0.05 * sample_rate)
    chunk, search = int(chunk_s * sample_rate), int(search_s * sample_rate)
    chunks, start = [], 0
    while n - start > chunk * 1.25:
        target = start + chunk
        lo = max(start + frame, target - search)
        window = np.asarray(samples[lo:target], dtype=np.float32)
        usable = (len(window) // frame) * frame
        if usable == 0:
            cut = target
        else:
            energy = (window[:usable].reshape(-1, frame) ** 2).mean(axis=1)
            cut = lo + int(np.argmin(energy)) * frame + frame // 2
        chunks.append((start, cut))
        start = cut
    chunks.append((start, n))
    return chunks


class Transcriber:
    """Local transcription using faster-whisper."""

    VALID_SIZES = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size.lower()
        self._model = None

    def load_model(self):
        if self.model_size not in self.VALID_SIZES:
            raise ValueError(f"Invalid size. Must be one of: {', '.join(self.VALID_SIZES)}")
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def transcribe(self, samples, word_timestamps: bool = False,
                   progress: ProgressFn = _no_progress,
                   chunk_s: float = CHUNK_SECONDS, threshold_s: float = CHUNK_THRESHOLD_SECONDS) -> list[Segment]:
        """Transcribe 16 kHz mono int16 samples (a numpy array).

        The array goes straight to faster-whisper (no temp file, no PyAV decode).
        `progress` is called with seconds of audio finished, as segments arrive.
        Long audio is processed in chunks; timestamps are made absolute.
        """
        if not self._model:
            raise RuntimeError("Model not loaded")
        try:
            out: list[Segment] = []
            for a, b in plan_chunks(samples, TARGET_RATE, chunk_s, threshold_s=threshold_s):
                offset = a / TARGET_RATE
                raw_segments, _info = self._model.transcribe(
                    int16_to_float32(samples[a:b]),
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True,
                    word_timestamps=word_timestamps,
                )
                # `raw_segments` is a lazy generator: iterating it *is* the work.
                for s in raw_segments:
                    out.append(Segment(
                        start=s.start + offset,
                        end=s.end + offset,
                        text=s.text.strip(),
                        words=[Word(w.start + offset, w.end + offset, w.word) for w in (s.words or [])],
                    ))
                    progress(out[-1].end)
                progress(b / TARGET_RATE)
            return out
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")


def plain_text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments if s.text).strip()


# --------------------------------------------------------------------------
# Diarization (who spoke when) via sherpa-onnx — ONNX only, no torch
# --------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...", file=sys.stderr)
    # Download to a temp name, then rename: an interrupted download must not
    # leave a truncated file that later passes the exists() check.
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def ensure_diarization_models() -> None:
    """Fetch the two ONNX models on first use (~30 MB total)."""
    if not EMBEDDING_MODEL.exists():
        print("First use of --diarize: fetching speaker models.", file=sys.stderr)
        _download(EMBEDDING_URL, EMBEDDING_MODEL)
    if not SEGMENTATION_MODEL.exists():
        archive = MODEL_DIR / "segmentation.tar.bz2"
        _download(SEGMENTATION_ARCHIVE_URL, archive)
        try:
            with tarfile.open(archive, "r:bz2") as tf:
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(MODEL_DIR, filter="data")
                else:  # older patch releases without extraction filters
                    tf.extractall(MODEL_DIR)
        finally:
            archive.unlink(missing_ok=True)
    if not (EMBEDDING_MODEL.exists() and SEGMENTATION_MODEL.exists()):
        raise RuntimeError(f"Diarization models missing under {MODEL_DIR}")


class Diarizer:
    """Finds speaker turns in 16 kHz mono audio."""

    def __init__(self, num_speakers: int | None = None, threshold: float = 0.5, num_threads: int | None = None):
        # num_speakers=None -> let the clusterer guess, controlled by `threshold`
        # (lower = more speakers, higher = fewer).
        self.num_speakers = num_speakers
        self.threshold = threshold
        self.num_threads = num_threads or min(4, os.cpu_count() or 1)
        self._sd = None

    def load(self) -> None:
        import sherpa_onnx

        ensure_diarization_models()
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(SEGMENTATION_MODEL),
                    # Window stride as a fraction of the window. Benchmarked on a Pi
                    # (4 cores): 0.1 (upstream default) was 2-3x slower than 0.25, scaled
                    # worse than linearly with length, and found *fewer* speakers on a
                    # 55 s test clip. 0.5 is faster still but merges short turns.
                    window_shift_ratio=0.25,
                ),
                num_threads=self.num_threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(EMBEDDING_MODEL), num_threads=self.num_threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=self.num_speakers or -1, threshold=self.threshold
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError("Invalid diarization config (check model files)")
        self._sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        if self._sd.sample_rate != TARGET_RATE:
            raise RuntimeError(f"Diarizer expects {self._sd.sample_rate} Hz, not {TARGET_RATE}")

    def diarize(self, samples, progress: ProgressFn = _no_progress) -> list[Turn]:
        """Speaker turns for 16 kHz mono int16 `samples`."""
        if self._sd is None:
            self.load()
        duration = len(samples) / TARGET_RATE

        def callback(done_chunks: int, total_chunks: int) -> int:
            # Called from native code: never let an exception escape into it.
            try:
                if total_chunks > 0:
                    progress(duration * done_chunks / total_chunks)
            except Exception:
                pass
            return 0  # 0 = keep going

        result = self._sd.process(int16_to_float32(samples), callback=callback).sort_by_start_time()
        progress(duration)
        return [Turn(r.start, r.end, r.speaker) for r in result]


def _speaker_at(turns: list[Turn], start: float, end: float) -> int | None:
    """Speaker whose turns overlap [start, end] the most; if none overlap
    (the word fell in a gap), the nearest turn in time."""
    overlap: dict[int, float] = {}
    for t in turns:
        o = min(end, t.end) - max(start, t.start)
        if o > 0:
            overlap[t.speaker] = overlap.get(t.speaker, 0.0) + o
    if overlap:
        return max(overlap, key=overlap.get)
    if not turns:
        return None
    mid = (start + end) / 2
    return min(turns, key=lambda t: 0.0 if t.start <= mid <= t.end else min(abs(mid - t.start), abs(mid - t.end))).speaker


def segment_words(seg: Segment) -> list[Word]:
    """The segment's words, or -- if Whisper gave no word timing -- the text
    spread across the segment's duration in proportion to word length."""
    if seg.words:
        return seg.words
    tokens = seg.text.split()
    if not tokens:
        return []
    total_chars = sum(len(t) for t in tokens)
    duration = max(seg.end - seg.start, 0.0)
    out, t = [], seg.start
    for tok in tokens:
        d = duration * len(tok) / total_chars
        out.append(Word(t, t + d, " " + tok))
        t += d
    return out


def label_words(segments: list[Segment], turns: list[Turn] | None) -> list[LabeledWord]:
    """Give every word a speaker (or None when there are no turns).

    Done per *word*, not per Whisper segment, because a segment can straddle
    a change of speaker. Speakers are renumbered 1, 2, ... by first appearance
    so the output reads naturally regardless of the diarizer's internal ids.
    """
    words = [w for seg in segments for w in segment_words(seg) if w.text.strip()]
    if not turns:
        return [LabeledWord(None, w.start, w.end, w.text) for w in words]
    renumber: dict[int, int] = {}
    out = []
    for w in words:
        raw = _speaker_at(turns, w.start, w.end)
        if raw not in renumber:
            renumber[raw] = len(renumber) + 1
        out.append(LabeledWord(renumber[raw], w.start, w.end, w.text))
    return out


def group_utterances(words: list[LabeledWord]) -> list[Utterance]:
    """Merge runs of consecutive words by the same speaker."""
    utterances: list[Utterance] = []
    for w in words:
        speaker = w.speaker or 0
        if not utterances or utterances[-1].speaker != speaker:
            utterances.append(Utterance(speaker, w.start, w.text))
        else:
            utterances[-1].text += w.text if w.text.startswith(" ") else " " + w.text
    for u in utterances:
        u.text = u.text.strip()
    return [u for u in utterances if u.text]


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[Utterance]:
    return group_utterances(label_words(segments, turns))


def format_utterances(utterances: list[Utterance]) -> str:
    return "\n\n".join(f"**Speaker {u.speaker}** ({format_clock(u.start)}): {u.text}" for u in utterances)


# --------------------------------------------------------------------------
# Captions (SubRip .srt)
# --------------------------------------------------------------------------

SRT_MAX_CHARS = 42            # per line: the usual broadcast / streaming limit
SRT_MAX_LINES = 2
SRT_MAX_DURATION_MS = 7000    # longest a single cue stays on screen
SRT_MIN_DURATION_MS = 1000    # shortest we try to leave a cue up (if the next cue allows)
SRT_MIN_SPAN_MS = 200         # hard floor: a cue is never shorter than this
SRT_GAP_BREAK_MS = 1500       # a pause this long always starts a new cue
_SENTENCE_END = (".", "?", "!", "…")
_CLAUSE_END = (",", ";", ":", "—")


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    lines: list[str]


def format_srt_timestamp(ms: int) -> str:
    """HH:MM:SS,mmm. Takes integer milliseconds, so rounding can never produce
    an invalid '00:00:60,000'."""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_caption(text: str, max_chars: int = SRT_MAX_CHARS, max_lines: int = SRT_MAX_LINES) -> list[str] | None:
    """Split `text` into at most `max_lines` balanced lines of at most `max_chars`.
    Returns None when it cannot fit."""
    words = text.split()
    if not words:
        return []
    if len(" ".join(words)) <= max_chars:
        return [" ".join(words)]
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > max_lines or any(len(line) > max_chars for line in lines):
        return None
    if len(lines) == 2:  # greedy fills line 1; rebalance so the two lines are similar in length
        best = None
        for i in range(1, len(words)):
            a, b = " ".join(words[:i]), " ".join(words[i:])
            if len(a) <= max_chars and len(b) <= max_chars and (best is None or max(len(a), len(b)) < best[0]):
                best = (max(len(a), len(b)), [a, b])
        if best:
            lines = best[1]
    return lines


def build_cues(words: list[LabeledWord], max_chars: int = SRT_MAX_CHARS, max_lines: int = SRT_MAX_LINES,
               max_duration_ms: int = SRT_MAX_DURATION_MS, min_duration_ms: int = SRT_MIN_DURATION_MS,
               gap_break_ms: int = SRT_GAP_BREAK_MS, max_end_ms: int | None = None) -> list[Cue]:
    """Turn timed words into caption cues.

    A cue ends at a change of speaker, a long pause, a sentence end (once it has a
    reasonable amount of text), a clause end (once it is mostly full), or when
    adding another word would exceed the line / duration limits. When speakers are
    known, the first cue of each turn is prefixed with "Speaker N: ". Cues never
    overlap and are never shorter than SRT_MIN_SPAN_MS.
    """
    unset = object()
    state = {"last_speaker": unset}
    raw: list[list] = []  # [start_ms, end_ms, lines]

    def clean(w: LabeledWord) -> str:
        return w.text.strip().replace("-->", "->")

    def prefix_for(group: list[LabeledWord]) -> str:
        sp = group[0].speaker
        return f"Speaker {sp}: " if sp is not None and sp != state["last_speaker"] else ""

    def text_of(group: list[LabeledWord]) -> str:
        return " ".join(t for t in (clean(w) for w in group) if t)

    def fits(group: list[LabeledWord]) -> bool:
        return len(group) == 1 or wrap_caption(prefix_for(group) + text_of(group), max_chars, max_lines) is not None

    cur: list[LabeledWord] = []

    def flush() -> None:
        if not cur:
            return
        text = prefix_for(cur) + text_of(cur)
        lines = wrap_caption(text, max_chars, max_lines) or [text]
        raw.append([sec_to_ms(cur[0].start), sec_to_ms(cur[-1].end), lines])
        state["last_speaker"] = cur[0].speaker
        cur.clear()

    for w in words:
        if not clean(w):
            continue
        if cur:
            prev = cur[-1]
            hard = (
                w.speaker != prev.speaker
                or sec_to_ms(w.start) - sec_to_ms(prev.end) > gap_break_ms
                or sec_to_ms(w.end) - sec_to_ms(cur[0].start) > max_duration_ms
                or not fits(cur + [w])
            )
            soft = False
            if not hard:
                size, tail = len(text_of(cur)), clean(prev)
                soft = (tail.endswith(_SENTENCE_END) and size >= 20) or (
                    tail.endswith(_CLAUSE_END) and size >= 0.6 * max_chars * max_lines)
            if hard or soft:
                flush()
        cur.append(w)
    flush()

    # Timing clean-up, in integer milliseconds.
    for i, c in enumerate(raw):
        nxt = raw[i + 1][0] if i + 1 < len(raw) else None
        if nxt is not None and c[1] > nxt:          # never run into the next cue
            c[1] = nxt
        want = c[0] + min_duration_ms               # leave it up long enough to read, if there is room
        if c[1] < want:
            c[1] = min(want, nxt) if nxt is not None else want
    prev_end = 0
    for c in raw:                                   # final pass: guarantee valid, ordered, non-overlapping
        c[0] = max(c[0], prev_end, 0)
        c[1] = max(c[1], c[0] + SRT_MIN_SPAN_MS)
        prev_end = c[1]
    cues = [Cue(a, b, lines) for a, b, lines in raw]
    if max_end_ms is not None:
        cues = [Cue(c.start_ms, min(c.end_ms, max_end_ms), c.lines) for c in cues if c.start_ms < max_end_ms]
    return cues


def render_srt(cues: list[Cue]) -> str:
    """SubRip text: index, 'start --> end', the lines, a blank line. CRLF line
    endings (the original SubRip convention; every player accepts them)."""
    return "".join(
        f"{i}\r\n{format_srt_timestamp(c.start_ms)} --> {format_srt_timestamp(c.end_ms)}\r\n"
        + "\r\n".join(c.lines) + "\r\n\r\n"
        for i, c in enumerate(cues, 1)
    )


def write_srt(path: Path, cues: list[Cue]) -> None:
    write_text_atomic(path, render_srt(cues), newline="")  # newline="": keep our CRLFs as written


# --------------------------------------------------------------------------
# Terminal UI
# --------------------------------------------------------------------------

# -- level / spectrum maths (pure, so they can be tested without a terminal) --

def to_dbfs(x: float, floor_db: float = -60.0) -> float:
    """Linear amplitude (0..1 of full scale) -> dBFS, never below `floor_db`."""
    import math

    return floor_db if x <= 0 else max(floor_db, 20.0 * math.log10(x))


def level_fraction(db: float, floor_db: float = -60.0) -> float:
    """dBFS -> 0..1 meter position (floor_db is empty, 0 dBFS is full)."""
    return min(1.0, max(0.0, (db - floor_db) / -floor_db))


def spectrum_bands(samples, sample_rate: int = TARGET_RATE, n_bands: int = 16, fmin: float = 100.0,
                   fmax: float = 7000.0, floor_db: float = -80.0, ceil_db: float = -20.0) -> list[float]:
    """Log-spaced spectrum of `samples` (floats in [-1, 1]) as `n_bands` values in 0..1."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    if len(x) < 256:
        return [0.0] * n_bands
    window = np.hanning(len(x)).astype(np.float32)
    amp = np.abs(np.fft.rfft(x * window)) / (window.sum() / 2.0)  # a full-scale sine reads 1.0
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    edges = np.geomspace(fmin, min(fmax, sample_rate / 2 * 0.99), n_bands + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = amp[(freqs >= lo) & (freqs < hi)]
        peak = float(band.max()) if band.size else 0.0
        db = 20.0 * np.log10(max(peak, 1e-9))
        out.append(float(min(1.0, max(0.0, (db - floor_db) / (ceil_db - floor_db)))))
    return out


_GRADIENT = [(0.0, (0, 210, 106)), (0.6, (255, 212, 0)), (1.0, (255, 59, 48))]  # green -> yellow -> red


def gradient_color(f: float) -> str:
    """0..1 -> '#rrggbb' along green -> yellow -> red."""
    f = min(1.0, max(0.0, f))
    for (f0, c0), (f1, c1) in zip(_GRADIENT, _GRADIENT[1:]):
        if f <= f1:
            t = (f - f0) / (f1 - f0)
            r, g, b = (round(a + (b2 - a) * t) for a, b2 in zip(c0, c1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#ff3b30"


_BLOCKS = " ▁▂▃▄▅▆▇█"


def render_header(elapsed: float, device: str = "", blink_on: bool = True):
    from rich.text import Text

    t = Text()
    t.append("● REC" if blink_on else "○ REC", style="bold red" if blink_on else "dim red")
    t.append("  ")
    t.append(format_clock(elapsed), style="bold white")
    if device:
        t.append(f"   {device}", style="dim")
    return t


def render_meter(fraction: float, db: float, width: int = 40):
    """A horizontal level meter whose cells run green -> yellow -> red."""
    from rich.text import Text

    t = Text("Level  ")
    for i in range(width):
        pos = (i + 0.5) / width
        if pos <= fraction:
            t.append("█", style=gradient_color(pos))
        else:
            t.append("░", style="grey30")
    t.append(f"  {db:6.1f} dB", style=gradient_color(fraction))
    return t


def render_spectrum(bands: list[float], rows: int = 4):
    """Vertical bars, `rows` characters tall, coloured green at the bottom to red at the top."""
    from rich.text import Text

    t = Text()
    for r in range(rows - 1, -1, -1):
        t.append("       ")
        for b in bands:
            cells = min(8, max(0, round(b * rows * 8) - r * 8))
            t.append(_BLOCKS[cells] * 2, style=gradient_color((r + 1) / rows) if cells else "")
            t.append(" ")
        if r:
            t.append("\n")
    return t


class RecordingView:
    """Builds one frame of the live recording display from a recorder snapshot."""

    SILENCE_WARN_S = 3.0

    def __init__(self, recorder, n_bands: int = 16, rows: int = 4, meter_width: int = 40):
        self.recorder = recorder
        self.n_bands, self.rows, self.meter_width = n_bands, rows, meter_width
        self._bands = [0.0] * n_bands
        self._level = 0.0

    def __call__(self):
        from rich.console import Group
        from rich.text import Text

        s = self.recorder.snapshot()
        peak_db = to_dbfs(s.peak)
        # Fast attack, slow decay: it reads like a real meter instead of flickering.
        self._level = max(level_fraction(peak_db), self._level * 0.9)
        raw = spectrum_bands(s.samples, TARGET_RATE, self.n_bands)
        self._bands = [max(r, old * 0.82) for r, old in zip(raw, self._bands)]

        parts = [
            render_header(s.elapsed, getattr(self.recorder, "device_name", ""), int(s.elapsed * 2) % 2 == 0),
            render_meter(self._level, peak_db, self.meter_width),
            render_spectrum(self._bands, self.rows),
        ]
        if s.silent_for >= self.SILENCE_WARN_S:
            parts.append(Text(f"⚠  No input signal for {int(s.silent_for)}s - is the microphone connected and unmuted?",
                              style="bold yellow"))
        elif s.clipped:
            parts.append(Text("⚠  Clipping - move back from the mic or lower the input gain", style="bold red"))
        if s.dropouts:
            parts.append(Text(f"{s.dropouts} audio dropout{'s' if s.dropouts != 1 else ''} reported by the device", style="dim"))
        parts.append(Text("Press Ctrl+C to stop and transcribe", style="dim"))
        return Group(*parts)


# -- UIs -------------------------------------------------------------------------

class PlainUI:
    """Line-oriented output. Used when stdout is not a terminal (pipes, cron,
    tests), when `rich` is missing, or when the user passes --no-ui."""

    def banner(self) -> None:
        print("=" * 60)
        print("Voice Recorder - Real-time Transcription")
        print("=" * 60)

    def info(self, msg: str) -> None:
        print(msg)

    def warn(self, msg: str) -> None:
        print(f"Warning: {msg}")

    def record(self, recorder: "AudioRecorder", stop_event: threading.Event) -> None:
        print("Recording... Press Ctrl+C to stop.")
        print("-" * 50)
        while not stop_event.wait(timeout=0.5):
            pass

    @contextmanager
    def progress(self, label: str, total: float):
        """Yields `update(seconds_done)`. Prints a line every 10%."""
        t0 = time.monotonic()
        state = {"next": 10}
        print(f"{label}...")

        def update(done: float) -> None:
            pct = 100 if total <= 0 else int(100 * min(done, total) / total)
            mark = pct // 10 * 10
            if mark >= state["next"]:
                print(f"  {label}: {mark}%")
                state["next"] = mark + 10

        yield update
        print(f"  {label}: done in {time.monotonic() - t0:.1f}s")


class RichUI(PlainUI):
    """Colourful live display while recording, real progress bars while processing."""

    REFRESH_HZ = 15

    def __init__(self, console=None):
        from rich.console import Console

        self.console = console or Console()

    def banner(self) -> None:
        self.console.rule("[bold cyan]Voice Recorder[/bold cyan] - real-time transcription", style="cyan")

    def info(self, msg: str) -> None:
        self.console.print(msg, markup=False, highlight=False)

    def warn(self, msg: str) -> None:
        self.console.print(f"Warning: {msg}", style="yellow", markup=False, highlight=False)

    def record(self, recorder, stop_event: threading.Event) -> None:
        from rich.live import Live

        view = RecordingView(recorder)
        with Live(get_renderable=view, console=self.console, refresh_per_second=self.REFRESH_HZ, transient=True):
            while not stop_event.wait(timeout=0.1):
                pass
        self.console.print(f"■ Recording stopped at {format_clock(recorder.elapsed())}", style="bold", markup=False)

    @contextmanager
    def progress(self, label: str, total: float):
        from rich.progress import (BarColumn, Progress, ProgressColumn, SpinnerColumn, TaskProgressColumn,
                                   TextColumn, TimeElapsedColumn, TimeRemainingColumn)
        from rich.text import Text

        class AudioClock(ProgressColumn):
            def render(self, task):
                return Text(f"{format_clock(task.completed)} / {format_clock(task.total or 0)}", style="cyan")

        class Speed(ProgressColumn):
            def render(self, task):
                elapsed = task.elapsed or 0.0
                if elapsed < 1.0 or task.completed <= 0:
                    return Text("")
                return Text(f"{task.completed / elapsed:.1f}x realtime", style="magenta")

        total = total if total > 0 else 1.0
        with Progress(
            SpinnerColumn(style="cyan"), TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None, complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(), AudioClock(), Speed(), TimeElapsedColumn(), TimeRemainingColumn(),
            console=self.console, transient=False,
        ) as bar:
            task = bar.add_task(label, total=total)
            yield lambda done: bar.update(task, completed=min(max(done, 0.0), total))
            bar.update(task, completed=total)


def make_ui(no_ui: bool = False, console=None):
    """RichUI on an interactive terminal (if `rich` is installed), else PlainUI."""
    if no_ui:
        return PlainUI()
    try:
        from rich.console import Console
    except ImportError:
        return PlainUI()
    if console is None:
        console = Console()
        if not console.is_terminal or console.is_dumb_terminal:
            return PlainUI()
    return RichUI(console)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

class VoiceRecorder:
    """Main voice recorder application."""

    def __init__(self, ui=None):
        self.ui = ui or PlainUI()
        self.recorder: AudioRecorder | None = None
        self.transcriber: Transcriber | None = None
        self.diarizer: Diarizer | None = None
        self._model_size = "small"
        self._output_dir = Path.cwd()
        self._save_audio = True
        self._srt = False
        self._stop_event = threading.Event()

    def setup(self, args):
        self._output_dir = Path(args.output_dir).expanduser() if getattr(args, "output_dir", None) else Path.cwd()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._save_audio = not getattr(args, "no_audio", False)
        self._srt = getattr(args, "srt", False)
        self.ui.info(f"Output directory: {self._output_dir}")

        self._model_size = (getattr(args, "model", None) or "small").lower()
        self.ui.info(f"Loading faster-whisper model ({self._model_size})...")
        try:
            self.transcriber = Transcriber(model_size=self._model_size)
            self.transcriber.load_model()
        except Exception as e:
            self.ui.info(f"Error loading transcription model: {e}")
            sys.exit(3)
        self.ui.info("✓ Model loaded successfully")

        if getattr(args, "diarize", False):
            self.diarizer = Diarizer(num_speakers=getattr(args, "speakers", None))
            try:
                # Load now (not after recording) so a missing model or no
                # network fails *before* the user has spent time talking.
                self.diarizer.load()
                self.ui.info("✓ Speaker diarizer ready")
            except Exception as e:
                self.ui.warn(f"diarization unavailable ({e}). Continuing without speaker labels.")
                self.diarizer = None

    # -- processing --------------------------------------------------------

    def process(self, samples) -> Result:
        """16 kHz mono int16 samples -> transcript (speaker-labelled if a diarizer is active)."""
        duration = len(samples) / TARGET_RATE
        with self.ui.progress("Transcribing", duration) as progress:
            segments = self.transcriber.transcribe(
                samples, word_timestamps=self.diarizer is not None or self._srt, progress=progress)

        turns = None
        if self.diarizer is not None:
            try:
                with self.ui.progress("Identifying speakers", duration) as progress:
                    turns = self.diarizer.diarize(samples, progress=progress)
                n = len({t.speaker for t in turns})
                self.ui.info(f"Found {n} speaker{'s' if n != 1 else ''}.")
                if not turns:
                    self.ui.warn("no speech turns found; saving unlabelled transcript.")
                    turns = None
            except Exception as e:
                # Never lose the transcript because the optional step failed.
                self.ui.warn(f"diarization failed ({e}). Saving unlabelled transcript.")
                turns = None

        words = label_words(segments, turns)
        body = format_utterances(group_utterances(words)) if turns else plain_text(segments)
        return Result(body=body, words=words, diarized=bool(turns), duration=duration)

    # -- output ------------------------------------------------------------

    def save_captions(self, result: Result, outputs: Outputs):
        """Write the .srt next to the .md. Returns its path, or None if there is nothing to caption."""
        cues = build_cues(result.words, max_end_ms=sec_to_ms(result.duration))
        if not cues:
            self.ui.warn("no speech to caption; no .srt written.")
            return None
        try:
            write_srt(outputs.srt, cues)
        except OSError as e:
            self.ui.info(f"Error saving captions: {e}")
            sys.exit(2)
        self.ui.info(f"✓ Captions saved to: {outputs.srt} ({len(cues)} cues)")
        return outputs.srt

    def save_transcript(self, result: Result, outputs: Outputs, recorded_at: datetime, audio_ref: str | None = None,
                        captions_ref: str | None = None):
        if not result.body.strip():
            self.ui.warn("No speech detected; nothing saved.")
            return None

        diarized = " + speaker diarization" if result.diarized else ""
        header = [
            "# Voice Recording Transcript\n",
            f"**Recorded:** {recorded_at.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Duration:** {format_clock(result.duration)}  ",
            f"**Model:** faster-whisper {self._model_size}{diarized}  ",
        ]
        if audio_ref:
            header.append(f"**Audio:** [{Path(audio_ref).name}]({audio_ref})  ")
        if captions_ref:
            header.append(f"**Captions:** [{captions_ref}]({captions_ref})  ")
        md_content = "\n".join(header) + "\n\n---\n\n" + result.body + "\n"
        try:
            write_text_atomic(outputs.md, md_content)
        except OSError as e:
            self.ui.info(f"Error saving transcript: {e}")
            sys.exit(2)
        self.ui.info(f"\n✓ Transcript saved to: {outputs.md}")
        return outputs.md

    # -- entry points ------------------------------------------------------

    def run(self, args):
        self.ui.banner()
        from_file = getattr(args, "from_file", None)
        if from_file:
            return self.run_from_file(Path(from_file).expanduser())
        return self.run_recording()

    def run_from_file(self, path: Path):
        try:
            samples = read_wav_as_16k_mono(path)
        except (ValueError, OSError) as e:
            self.ui.info(f"Error reading {path}: {e}")
            sys.exit(1)
        duration = len(samples) / TARGET_RATE
        self.ui.info(f"Loaded {path.name}: {format_clock(duration)} of audio")
        if duration < 0.5:
            self.ui.info("Audio too short; nothing to transcribe.")
            return
        outputs = Outputs(self._output_dir, unique_stem(self._output_dir, path.stem, (".md", ".srt")))
        audio_ref = path.name if path.resolve().parent == self._output_dir.resolve() else str(path)
        self._transcribe_and_save(samples, outputs, datetime.now(), audio_ref, None)

    def run_recording(self):
        started = datetime.now()
        outputs = Outputs(self._output_dir, unique_stem(self._output_dir, started.strftime("%Y-%m-%d_%H-%M-%S")))
        if self._save_audio:
            wav_path = outputs.wav
        else:
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="transcripter_")
            os.close(fd)
            wav_path = Path(tmp)

        self.recorder = AudioRecorder(wav_path, on_error=self._stop_event.set)
        try:
            self.recorder.start()
        except Exception as e:
            self.ui.info(f"Error starting recorder: {e}")
            wav_path.unlink(missing_ok=True)
            sys.exit(1)
        self.ui.info(f"Using microphone: {self.recorder.device_name}")

        # Ctrl+C only sets a flag; the main thread blocks on the event.
        # The previous handler is restored right after, so a second Ctrl+C
        # during transcription aborts normally.
        original = signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())
        try:
            self.ui.record(self.recorder, self._stop_event)
        finally:
            signal.signal(signal.SIGINT, original)

        seconds = self.recorder.stop()
        if self.recorder.error:
            self.ui.warn(f"recording stopped early: {self.recorder.error}")
        try:
            if seconds < 0.5:
                self.ui.info("Recording too short; nothing to transcribe.")
                wav_path.unlink(missing_ok=True)
                return
            if self._save_audio:
                self.ui.info(f"\n✓ Audio saved: {wav_path} ({format_clock(seconds)})")
            samples = read_wav_as_16k_mono(wav_path)
            self._transcribe_and_save(
                samples, outputs, started, outputs.wav.name if self._save_audio else None,
                wav_path if self._save_audio else None,
            )
        finally:
            if not self._save_audio:
                wav_path.unlink(missing_ok=True)

    def _transcribe_and_save(self, samples, outputs: Outputs, recorded_at: datetime,
                             audio_ref: str | None, keep_audio: Path | None):
        duration = len(samples) / TARGET_RATE
        if duration > 600:
            self.ui.info(
                f"Processing {format_clock(duration)} of audio; this can take a while on slow hardware. "
                "Use a smaller --model to go faster."
            )
        self.ui.info("Processing... (Ctrl+C to abort)")
        try:
            result = self.process(samples)
            self.ui.info(f"Transcription complete: {len(result.body)} characters")
            srt_path = self.save_captions(result, outputs) if self._srt else None
            self.save_transcript(result, outputs, recorded_at, audio_ref, srt_path.name if srt_path else None)
        except KeyboardInterrupt:
            self.ui.info("\nInterrupted.")
            self._audio_safe_hint(keep_audio)
            sys.exit(130)
        except RuntimeError as e:
            self.ui.info(f"Error: {e}")
            self._audio_safe_hint(keep_audio)
            sys.exit(4)

    def _audio_safe_hint(self, wav: Path | None) -> None:
        if wav is not None:
            self.ui.info(f"Your audio is safe at {wav}\nTo transcribe it later: voice-record --from-file {wav}")


def main():
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe to markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  voice-record                          # Record with default settings
  voice-record --output-dir ./notes     # Save to custom directory
  voice-record --model medium           # Use larger model for better accuracy
  voice-record --diarize --speakers 2   # Label who said what
  voice-record --from-file meeting.wav  # Transcribe an existing recording

Each recording is saved as <timestamp>.wav next to <timestamp>.md (use
--no-audio to skip the WAV). The WAV is streamed to disk while you record, so
long meetings use constant memory and survive a crash.

Model sizes (smaller = faster):
  tiny    - ~76MB, very fast but less accurate
  base    - ~149MB, balanced speed/accuracy
  small   - ~309MB, good quality (default)
  medium  - ~754MB, high quality
  large   - ~3GB, best quality

Speaker labels (--diarize) are per recording: "Speaker 1" in one file is not
necessarily the same person as "Speaker 1" in another. Models (~30 MB) are
downloaded to ~/.cache/transcripter on first use.

Exit codes: 0 ok, 1 no mic / unreadable input, 2 cannot write output,
3 model load failed, 4 transcription failed (audio is kept), 130 interrupted.
        """,
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Directory for the .md/.wav files (default: current directory)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--diarize", action="store_true",
        help="Label speakers (Speaker 1, Speaker 2, ...)",
    )
    parser.add_argument(
        "--speakers", "-n", type=int, default=None, metavar="N",
        help="Number of speakers, if known (much more reliable than auto-detect; implies --diarize)",
    )
    parser.add_argument(
        "--srt", action="store_true",
        help="Also write <timestamp>.srt (closed-caption file) next to the transcript",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Do not keep the recorded .wav (default: save it next to the transcript)",
    )
    parser.add_argument(
        "--no-ui", action="store_true",
        help="Plain text output: no colours, live meter or progress bars (automatic when output is not a terminal)",
    )
    parser.add_argument(
        "--from-file", type=str, default=None, metavar="WAV",
        help="Transcribe an existing 16-bit PCM WAV instead of recording",
    )
    args = parser.parse_args()
    if args.speakers is not None:
        if args.speakers < 1:
            parser.error("--speakers must be >= 1")
        args.diarize = True
    if args.from_file and not Path(args.from_file).expanduser().is_file():
        parser.error(f"--from-file: no such file: {args.from_file}")

    app = VoiceRecorder(make_ui(args.no_ui))
    app.setup(args)
    app.run(args)


if __name__ == "__main__":
    main()
