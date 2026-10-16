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
from pathlib import Path
from datetime import datetime


class AudioRecorder:
    """Cross-platform audio recorder that streams to terminal."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None
        self._buffer = bytearray()
        self._is_recording = False
    
    def list_devices(self) -> list[tuple[str, str, int]]:
        """List all available audio devices."""
        try:
            import sounddevice as sd
            
            devices = sd.query_devices()
            
            input_devices = []
            for i, info in enumerate(devices):
                if info['max_input_channels'] > 0 or 'input' in str(info).lower():
                    name = info.get('name', f"Device {i}")
                    is_default = info.get('default', False)
                    input_devices.append((name, str(info.get('hostapi', '?')), i))
            
            return input_devices
        
        except Exception as e:
            print(f"[WARN] Audio device detection failed: {e}")
            return []
    
    def get_default_device(self) -> tuple[str, int]:
        """Get default input device name and index."""
        try:
            import sounddevice as sd
            
            devices = sd.query_devices()
            
            # Find default input device
            for i, info in enumerate(devices):
                if info['max_input_channels'] > 0:
                    is_default = info.get('default', False) or 'default' in str(info).lower()
                    name = info.get('name', f"Device {i}")
                    if is_default and name:
                        return (name, i)
            
            # Fallback to first input device with valid channels
            for i, info in enumerate(devices):
                if info['max_input_channels'] > 0 and info.get('name'):
                    return (info['name'], i)
            
            raise RuntimeError("No input devices found")
        
        except Exception as e:
            print(f"[ERROR] {e}")
            raise
    
    def start(self) -> None:
        """Start recording stream."""
        try:
            import sounddevice as sd
            
            card_name, device_index = self.get_default_device()
            
            # Get the card object
            devices = sd.query_devices()
            if not devices or len(devices) <= device_index:
                raise RuntimeError(f"Device {device_index} not found")
            
            info = devices[device_index]
            
            print(f"Using microphone: {card_name}")
            print("Recording started... Press Ctrl+C to stop.")
            print("-" * 50)
            
            def callback(indata, frames, time, status):
                """Callback for streaming audio data."""
                if status:
                    print(f"[WARN] {status}")
                
                # Convert float32 to int16 (WAV format)
                import numpy as np
                samples = np.int16(indata * 32767.0)
                self._buffer.extend(samples.tobytes())
            
            # Create stream
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback
            )
            self._stream.start()
            self._is_recording = True
            
        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}")
            raise
    
    def stop(self) -> bytes:
        """Stop recording and return audio data."""
        try:
            import sounddevice as sd
            
            if self._stream:
                self._stream.stop()
                self._stream.close()
            
            self._is_recording = False
            
            # Convert buffer to WAV format
            import wave
            import tempfile
            
            temp_wav = None
            try:
                output_dir = Path.home() / '.hermes' / 'voice-recorder'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with tempfile.NamedTemporaryFile(
                    suffix='.wav', 
                    delete=False,
                    dir=output_dir
                ) as f:
                    temp_wav = f.name
                
                # Write WAV header and data
                with wave.open(temp_wav, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(bytes(self._buffer))
                
                with open(temp_wav, 'rb') as f:
                    return f.read()
            finally:
                if temp_wav and Path(temp_wav).exists():
                    Path(temp_wav).unlink()
                    
        except Exception as e:
            print(f"[ERROR] Failed to stop recording: {e}")
            raise
    
    def get_duration_seconds(self) -> float:
        """Get current recording duration in seconds."""
        if not self._buffer:
            return 0.0
        # Each sample is 2 bytes (int16), and we have channels samples per frame
        return len(self._buffer) / (self.sample_rate * self.channels * 2)


class Transcriber:
    """Real-time transcription using local faster-whisper model."""
    
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size.lower()
        self._model = None
    
    def load_model(self):
        """Load Whisper model from cache."""
        try:
            print(f"Loading faster-whisper model ({self.model_size})...")
            
            # Available sizes for faster-whisper
            valid_sizes = ["tiny", "base", "small", "medium", "large"]
            if self.model_size not in valid_sizes:
                raise ValueError(f"Invalid size. Must be one of: {', '.join(valid_sizes)}")
            
            # Load model (downloads if not cached)
            from faster_whisper import WhisperModel
            
            self._model = WhisperModel(
                self.model_size,
                device="cpu",  # CPU for cross-platform compatibility
                compute_type="int8"  # Optimized inference
            )
            print("✓ Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe_streaming(self, audio_data: bytes) -> str:
        """
        Transcribe audio with streaming output.
        
        Args:
            audio_data: Raw WAV audio data (bytes)
            
        Returns:
            Final complete transcript text
        """
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        # Convert bytes to numpy array
        import wave
        
        # Create temporary file for processing
        temp_path = Path.home() / '.hermes' / 'voice-recorder' / 'temp.wav'
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with wave.open(str(temp_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            
            # Transcribe the audio file
            result = self._model.transcribe(
                str(temp_path),
                language=None,
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,  # Deterministic for MVP
                word_timestamps=True
            )
            
            transcript = result[0].text.strip()
            print("-" * 50)
            print(f"\nTranscription complete: {len(transcript)} characters")
            
            return transcript
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")


class VoiceRecorder:
    """Main voice recorder application."""
    
    def __init__(self):
        self.recorder = None
        self.transcriber = None
        self._model_size = "small"
        self._output_dir = Path.cwd()
        self._is_running = False
    
    def setup(self, args):
        """Initialize components based on arguments."""
        
        # Setup output directory
        self._output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self._output_dir}")
        
        # Load transcriber
        try:
            model_size = args.model.lower() if args.model else "small"
            valid_sizes = ["tiny", "base", "small", "medium", "large"]
            
            if model_size not in valid_sizes:
                print(f"Error: Invalid model size '{model_size}'. Must be one of: {', '.join(valid_sizes)}")
                sys.exit(3)
            
            self._model_size = model_size
            self.transcriber = Transcriber(model_size=model_size)
            self.transcriber.load_model()
            
        except Exception as e:
            print(f"Error loading transcription model: {e}")
            sys.exit(3)
    
    def start_recording(self):
        """Start audio recording."""
        try:
            self.recorder = AudioRecorder(sample_rate=16000, channels=1)
            self.recorder.start()
            self._is_running = True
            
        except Exception as e:
            print(f"Error starting recorder: {e}")
            sys.exit(1)
    
    def stop_recording(self):
        """Stop audio recording."""
        if self.recorder and self._is_running:
            try:
                audio_data = self.recorder.stop()
                return audio_data
            except Exception as e:
                print(f"Error stopping recorder: {e}")
                return None
    
    def save_transcript(self, transcript: str) -> Path:
        """Save transcript to markdown file."""
        
        if not transcript.strip():
            print("Warning: No transcription text generated.")
            return None
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.md"
        filepath = self._output_dir / filename
        
        # Create markdown content
        md_content = f"""# Voice Recording Transcript

**Recorded:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** faster-whisper {self._model_size}  

---

{transcript}
"""
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            print(f"\n✓ Transcript saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving transcript: {e}")
            sys.exit(2)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nInterrupt received...")
    
    def run(self, args):
        """Main recording loop."""
        
        # Setup signal handler for graceful shutdown
        original_handler = signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            print("=" * 60)
            print("Voice Recorder - Real-time Transcription")
            print("=" * 60)
            
            # Start recording
            self.start_recording()
            
            # Main loop - in real implementation this would stream chunks
            # For MVP, we'll record the entire audio then transcribe once
            print("\nRecording complete. Processing transcription...")
            
            # Stop recording and get audio data
            audio_data = self.stop_recording()
            
            if not audio_data:
                print("Error: Failed to capture audio.")
                sys.exit(1)
            
            # Transcribe the full recording
            transcript = self.transcriber.transcribe_streaming(audio_data)
            
            # Save to file
            filepath = self.save_transcript(transcript)
            
        except KeyboardInterrupt:
            print("\n\nRecording interrupted.")
            try:
                audio_data = self.stop_recording()
                
                if audio_data and self.transcriber._model:
                    # Try to transcribe partial recording
                    transcript = self.transcriber.transcribe_streaming(audio_data)
                    filepath = self.save_transcript(transcript)
                    
            except Exception as e:
                print(f"Error during interrupt handling: {e}")
            
        finally:
            signal.signal(signal.SIGINT, original_handler)


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nInterrupt received...")


# Global signal handler reference for VoiceRecorder to use
_signal_handler_ref = _signal_handler


def main():
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe to markdown in real-time.",
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

Requirements:
  - Physical microphone connected
  - sounddevice package installed
  - faster-whisper package installed
        """
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save .md files (default: current directory)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)"
    )
    
    args = parser.parse_args()
    
    app = VoiceRecorder()
    app.setup(args)
    app.run(args)


if __name__ == "__main__":
    main()
