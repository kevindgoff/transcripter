# Voice Recorder - Real-time Audio Transcription Tool

A CLI tool that records audio from your microphone and transcribes it to markdown using local Whisper models.

## Features

- **Real-time streaming**: Text appears in terminal as you speak (post-processing for MVP)
- **Cross-platform**: Works on macOS, Windows, and Linux
- **Local processing**: No internet required - all transcription happens locally
- **Configurable output**: Choose where to save transcripts with `--output-dir`
- **Multiple model sizes**: From tiny (~76MB) to large (~3GB)

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install sounddevice faster-whisper numpy
```

## Usage

### Basic Recording

```bash
voice-record
```

This will:
1. Detect and use your default microphone
2. Record audio to WAV format
3. Transcribe using the small Whisper model (default)
4. Save transcript as `YYYY-MM-DD_HH-MM-SS.md` in current directory

### Custom Output Directory

```bash
voice-record --output-dir ~/my-voices
```

### Different Model Size

```bash
# Faster but less accurate
voice-record --model tiny

# Balanced (default)
voice-record --model small

# High quality
voice-record --model medium
```

## Requirements

- **Physical microphone** connected to your system
- **sounddevice** package for cross-platform audio capture
- **faster-whisper** package for local transcription
- **Python 3.8+**

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | No microphone found / Audio capture failed |
| 2 | Disk full / I/O error saving transcript |
| 3 | Invalid model size / Model load failed |

## Architecture

```
voice_recorder_cli.py (CLI)
├── AudioRecorder    # Cross-platform WAV recording using sounddevice
│   └── Records to WAV format (16kHz, mono, 16-bit)
└── Transcriber      # Whisper integration using faster-whisper
    └── Processes full recording after capture completes
```

## File Flow

1. User runs `voice-record` command
2. System detects default microphone
3. Audio recorded to temporary WAV file (16kHz, mono)
4. Whisper model processes audio
5. Transcript saved as `.md` file with timestamp

## Output Format

Transcripts are saved in markdown format:

```markdown
# Voice Recording Transcript

**Recorded:** 2024-01-15 14:30:25  
**Model:** faster-whisper small  

---

[transcribed text here]
```

## Troubleshooting

### "No input devices found"

This means no microphone is detected. Check:
- Microphone is connected and enabled in system settings
- Audio drivers are installed
- You're running with appropriate permissions (especially on Linux)

### Transcription fails silently

The tool continues recording even if transcription fails. Check the exit code to see what happened.

### Slow performance

Use a smaller model: `voice-record --model tiny` for faster processing.

## License

MIT
