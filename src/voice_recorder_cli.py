"""
Voice Recorder - Real-time Audio Transcription Tool

A CLI tool that records audio from your microphone and transcribes it
to markdown using local Whisper models. Optionally labels who said what
(Speaker 1, Speaker 2, ...) using local speaker diarization.

Usage:
    voice-record [--output-dir DIR] [--model SIZE] [--diarize] [--speakers N]

Examples:
    voice-record                          # Record with default settings
    voice-record --output-dir ./notes     # Save to custom directory
    voice-record --model medium           # Use larger model for better accuracy
    voice-record --diarize --speakers 2   # Label two speakers
"""

import argparse
import os
import signal
import sys
import tarfile
import tempfile
import threading
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TARGET_RATE = 16000  # Whisper and the diarizer both expect 16 kHz mono

MODEL_DIR = Path(os.environ.get("TRANSCRIPTER_MODEL_DIR", Path.home() / ".cache" / "transcripter" / "models"))
_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_ARCHIVE_URL = f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_MODEL = MODEL_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING_URL = f"{_RELEASES}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx"  # (sic) upstream typo
EMBEDDING_MODEL = MODEL_DIR / "wespeaker_en_voxceleb_resnet34.onnx"


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
class Utterance:
    """Consecutive words by one speaker, ready for output."""
    speaker: int  # 1-based, numbered by first appearance
    start: float
    text: str


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

class AudioRecorder:
    """Records mono int16 audio from the default input device until stopped."""

    def __init__(self, sample_rate: int = TARGET_RATE, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._actual_rate = sample_rate

    def start(self) -> None:
        import sounddevice as sd

        device_info = sd.query_devices(kind="input")  # raises if no input device
        print(f"Using microphone: {device_info['name']}")

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[WARN] {status}", file=sys.stderr)
            with self._lock:
                self._buffer.extend(indata.tobytes())

        # Prefer 16 kHz directly; fall back to the device's native rate and
        # resample in stop() if the hardware refuses.
        rates = [self.sample_rate]
        native = int(device_info["default_samplerate"])
        if native != self.sample_rate:
            rates.append(native)

        last_err = None
        for rate in rates:
            try:
                self._stream = sd.InputStream(
                    samplerate=rate,
                    channels=self.channels,
                    dtype="int16",
                    callback=callback,
                )
                self._stream.start()
                self._actual_rate = rate
                return
            except sd.PortAudioError as e:
                last_err = e
                self._stream = None
        raise RuntimeError(f"Could not open input stream: {last_err}")

    def stop(self) -> bytes:
        """Stop recording; return raw 16 kHz mono int16 PCM bytes."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            pcm = bytes(self._buffer)

        if self._actual_rate != TARGET_RATE and pcm:
            import numpy as np

            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            n_out = int(len(samples) * TARGET_RATE / self._actual_rate)
            resampled = np.interp(
                np.linspace(0, len(samples) - 1, n_out),
                np.arange(len(samples)),
                samples,
            )
            pcm = resampled.astype(np.int16).tobytes()
        return pcm


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def pcm_to_float32(pcm: bytes):
    import numpy as np

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class Transcriber:
    """Local transcription using faster-whisper."""

    VALID_SIZES = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size.lower()
        self._model = None

    def load_model(self):
        if self.model_size not in self.VALID_SIZES:
            raise ValueError(f"Invalid size. Must be one of: {', '.join(self.VALID_SIZES)}")
        print(f"Loading faster-whisper model ({self.model_size})...")
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
        print("✓ Model loaded successfully")

    def transcribe(self, pcm: bytes, word_timestamps: bool = False) -> list[Segment]:
        """Transcribe raw 16 kHz mono int16 PCM. Passes a numpy array straight
        to faster-whisper, so no temp file and no PyAV decode step."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        try:
            raw_segments, _info = self._model.transcribe(
                pcm_to_float32(pcm),
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                word_timestamps=word_timestamps,
            )
            # `raw_segments` is a lazy generator; materialise *all* of it.
            return [
                Segment(
                    start=s.start,
                    end=s.end,
                    text=s.text.strip(),
                    words=[Word(w.start, w.end, w.word) for w in (s.words or [])],
                )
                for s in raw_segments
            ]
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
                tf.extractall(MODEL_DIR, filter="data")
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

    def diarize(self, pcm: bytes) -> list[Turn]:
        if self._sd is None:
            self.load()
        result = self._sd.process(pcm_to_float32(pcm)).sort_by_start_time()
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


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[Utterance]:
    """Label every word with a speaker, then merge runs into utterances.

    Done per *word*, not per Whisper segment, because a segment can straddle
    a change of speaker. Speakers are renumbered 1, 2, ... by first appearance
    so the output reads naturally regardless of the diarizer's internal ids.
    """
    labelled: list[tuple[int | None, Word]] = []
    for seg in segments:
        words = seg.words or [Word(seg.start, seg.end, seg.text)]  # no word timing: whole segment
        for w in words:
            labelled.append((_speaker_at(turns, w.start, w.end), w))

    renumber: dict[int | None, int] = {}
    utterances: list[Utterance] = []
    last_raw: object = object()  # sentinel
    for raw, w in labelled:
        if raw not in renumber:
            renumber[raw] = len(renumber) + 1
        if raw != last_raw or not utterances:
            utterances.append(Utterance(renumber[raw], w.start, w.text))
            last_raw = raw
        else:
            utterances[-1].text += w.text if w.text.startswith(" ") else " " + w.text
    for u in utterances:
        u.text = u.text.strip()
    return [u for u in utterances if u.text]


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def format_utterances(utterances: list[Utterance]) -> str:
    return "\n\n".join(f"**Speaker {u.speaker}** ({_mmss(u.start)}): {u.text}" for u in utterances)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

class VoiceRecorder:
    """Main voice recorder application."""

    def __init__(self):
        self.recorder = None
        self.transcriber = None
        self.diarizer: Diarizer | None = None
        self._model_size = "small"
        self._output_dir = Path.cwd()
        self._stop_event = threading.Event()

    def setup(self, args):
        self._output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self._output_dir}")

        self._model_size = (args.model or "small").lower()
        try:
            self.transcriber = Transcriber(model_size=self._model_size)
            self.transcriber.load_model()
        except Exception as e:
            print(f"Error loading transcription model: {e}")
            sys.exit(3)

        if args.diarize:
            self.diarizer = Diarizer(num_speakers=args.speakers)
            try:
                # Load now (not after recording) so a missing model or no
                # network fails *before* the user has spent time talking.
                self.diarizer.load()
                print("✓ Speaker diarizer ready")
            except Exception as e:
                print(f"Warning: diarization unavailable ({e}). Continuing without speaker labels.")
                self.diarizer = None

    def process(self, pcm: bytes) -> str:
        """PCM -> transcript body (speaker-labelled if a diarizer is active)."""
        segments = self.transcriber.transcribe(pcm, word_timestamps=self.diarizer is not None)
        if self.diarizer is not None:
            try:
                print("Identifying speakers...")
                turns = self.diarizer.diarize(pcm)
                n = len({t.speaker for t in turns})
                print(f"Found {n} speaker{'s' if n != 1 else ''}.")
                if turns:
                    return format_utterances(assign_speakers(segments, turns))
                print("Warning: no speech turns found; saving unlabelled transcript.")
            except Exception as e:
                # Never lose the transcript because the optional step failed.
                print(f"Warning: diarization failed ({e}). Saving unlabelled transcript.")
        return plain_text(segments)

    def save_transcript(self, transcript: str):
        if not transcript.strip():
            print("Warning: No speech detected; nothing saved.")
            return None

        now = datetime.now()
        filepath = self._output_dir / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.md"
        diarized = " + speaker diarization" if self.diarizer else ""
        md_content = (
            "# Voice Recording Transcript\n\n"
            f"**Recorded:** {now.strftime('%Y-%m-%d %H:%M:%S')}  \n"
            f"**Model:** faster-whisper {self._model_size}{diarized}  \n\n"
            "---\n\n"
            f"{transcript}\n"
        )
        try:
            filepath.write_text(md_content, encoding="utf-8")
        except OSError as e:
            print(f"Error saving transcript: {e}")
            sys.exit(2)
        print(f"\n✓ Transcript saved to: {filepath}")
        return filepath

    def run(self, args):
        print("=" * 60)
        print("Voice Recorder - Real-time Transcription")
        print("=" * 60)

        self.recorder = AudioRecorder()
        try:
            self.recorder.start()
        except Exception as e:
            print(f"Error starting recorder: {e}")
            sys.exit(1)

        print("Recording... Press Ctrl+C to stop.")
        print("-" * 50)

        # Ctrl+C only sets a flag; the main thread blocks on the event.
        # The previous handler is restored right after, so a second Ctrl+C
        # during transcription aborts normally.
        original = signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())
        try:
            while not self._stop_event.wait(timeout=0.5):
                pass
        finally:
            signal.signal(signal.SIGINT, original)

        print("\nStopping... processing transcription (Ctrl+C again to abort).")
        pcm = self.recorder.stop()
        seconds = len(pcm) / (TARGET_RATE * 2)
        print(f"Captured {seconds:.1f}s of audio.")
        if seconds < 0.5:
            print("Recording too short; nothing to transcribe.")
            return

        transcript = self.process(pcm)
        print(f"Transcription complete: {len(transcript)} characters")
        self.save_transcript(transcript)


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

Model sizes (smaller = faster):
  tiny    - ~76MB, very fast but less accurate
  base    - ~149MB, balanced speed/accuracy
  small   - ~309MB, good quality (default)
  medium  - ~754MB, high quality
  large   - ~3GB, best quality

Speaker labels (--diarize) are per recording: "Speaker 1" in one file is not
necessarily the same person as "Speaker 1" in another. Models (~30 MB) are
downloaded to ~/.cache/transcripter on first use.
        """,
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Directory to save .md files (default: current directory)",
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
    args = parser.parse_args()
    if args.speakers is not None:
        if args.speakers < 1:
            parser.error("--speakers must be >= 1")
        args.diarize = True

    app = VoiceRecorder()
    app.setup(args)
    app.run(args)


if __name__ == "__main__":
    main()
