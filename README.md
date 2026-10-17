# Transcripter - Real-time Audio Transcription Tool

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
transcripter
```

This will:
1. Detect and use your default microphone
2. Record audio to WAV format
3. Transcribe using the small Whisper model (default)
4. Save transcript as `YYYY-MM-DD_HH-MM-SS.md` in current directory

### Custom Output Directory

```bash
transcripter --output-dir ~/my-voices
```

### Different Model Size

```bash
# Faster but less accurate
transcripter --model tiny

# Balanced (default)
transcripter --model small

# High quality
transcripter --model medium
```

### Speaker Labels (Who Said What)

```bash
pip install "sherpa-onnx"        # or: pip install -e ".[diarize]"
transcripter --diarize --speakers 2
```

Output looks like:

```
**Speaker 1** (00:00): Hi Sanjay, can you help me understand what Generative AI is?

**Speaker 2** (00:14): Sure. It refers to models that can generate new content...
```

- Runs **locally** after you press Ctrl+C. Two small ONNX models (~30 MB) are downloaded to `~/.cache/transcripter/` on first use. No torch, no accounts.
- **Pass `--speakers N` when you know the count.** Auto-detect (`--diarize` alone) tends to under-count; on one test clip it found 1 speaker where there were 2.
- **It is slow on a Raspberry Pi.** Measured on a 4-core Pi: ~1.3x real-time end to end with `--model small` (a 55 s clip took 73 s). Use `--model base` or `tiny` for long recordings.
- Labels are **per recording**: "Speaker 1" in one file is not necessarily the same person as "Speaker 1" in another.
- Limits: overlapping speech is not separated, and turn boundaries can be off by a second or two, especially on very short replies ("Yeah, sure"). Treat labels as a good guide, not a legal transcript.
- If diarization can't load (e.g. no network on first use), you still get the normal unlabelled transcript.

## Requirements

- **Physical microphone** connected to your system
- **sounddevice** package for cross-platform audio capture
- **faster-whisper** package for local transcription
- **sherpa-onnx** (optional) for `--diarize`
- **Python 3.10+**

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

1. User runs `transcripter` command
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

Use a smaller model: `transcripter --model tiny` for faster processing.

## License

MIT
