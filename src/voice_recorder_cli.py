"""
Voice Recorder - Real-time Audio Transcription Tool

A CLI tool that records audio from your microphone and transcribes it
to markdown using local Whisper models.

Usage:
    voice-record [--output-dir DIR] [--model SIZE]

Examples:
    voice-record                          # Record with default settings
    voice-record --output-dir ./notes     # Save to custom directory
    voice-record --model medium           # Use larger model for better accuracy
"""

import argparse
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

TARGET_RATE = 16000  # Whisper expects 16 kHz mono


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

    def transcribe(self, pcm: bytes) -> str:
        """Transcribe raw 16 kHz mono int16 PCM. Passes a numpy array straight
        to faster-whisper, so no temp file and no PyAV decode step."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            segments, _info = self._model.transcribe(
                audio,
                task="transcribe",
                beam_size=5,
                vad_filter=True,
            )
            # `segments` is a lazy generator; join *all* of them.
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")


class VoiceRecorder:
    """Main voice recorder application."""

    def __init__(self):
        self.recorder = None
        self.transcriber = None
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

    def save_transcript(self, transcript: str):
        if not transcript.strip():
            print("Warning: No speech detected; nothing saved.")
            return None

        now = datetime.now()
        filepath = self._output_dir / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.md"
        md_content = (
            "# Voice Recording Transcript\n\n"
            f"**Recorded:** {now.strftime('%Y-%m-%d %H:%M:%S')}  \n"
            f"**Model:** faster-whisper {self._model_size}  \n\n"
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

        transcript = self.transcriber.transcribe(pcm)
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

Model sizes (smaller = faster):
  tiny    - ~76MB, very fast but less accurate
  base    - ~149MB, balanced speed/accuracy
  small   - ~309MB, good quality (default)
  medium  - ~754MB, high quality
  large   - ~3GB, best quality
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
    args = parser.parse_args()

    app = VoiceRecorder()
    app.setup(args)
    app.run(args)


if __name__ == "__main__":
    main()
