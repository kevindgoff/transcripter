# Transcripter - Real-time Audio Transcription Tool

A CLI tool that records audio from your microphone and transcribes it to markdown using local Whisper models.
Optionally labels who said what, and writes closed-caption (`.srt`) files.

## Features

- **Local processing**: no internet required - transcription happens on your machine
- **Saves your audio**: by default a `.wav` is saved alongside the transcript
- **Custom output paths**: place WAV and Markdown files in separate folders, with an optional filename prefix
- **Built for long meetings**: audio streams to disk while you record, so memory stays flat and a crash does not lose the meeting
- **Speaker labels** (`--diarize`): "Speaker 1 / Speaker 2" transcripts
- **Closed captions** (`--srt`): a standards-compliant SubRip file you can load in any video player
- **Live recording display**: running clock, level meter and a colourful spectrum; real progress bars while transcribing
- **Cross-platform**: macOS, Windows, Linux
- **Multiple model sizes**: from tiny (~76MB) to large (~3GB)

## Installation

Clone the repo, then install it **into a virtual environment** (this is what creates the `transcripter` command):

```bash
git clone https://github.com/kevindgoff/transcripter.git
cd transcripter

python -m venv venv
source venv/bin/activate          # macOS / Linux
# Windows PowerShell:  venv\Scripts\Activate.ps1
# Windows cmd:         venv\Scripts\activate.bat

pip install -e .                  # installs the dependencies AND the `transcripter` command
```

Optional extras (quote the brackets in PowerShell and zsh):

```bash
pip install -e ".[diarize]"       # speaker labels (--diarize)
pip install -e ".[diarize,test]"  # ...plus pytest, to run the tests
```

Python 3.10 or newer. Check it worked: `transcripter --help`.

> **Installing only the dependencies is not enough.** `pip install sounddevice faster-whisper ...` gets the libraries but does not create the `transcripter` command, and you will see *"'transcripter' is not recognized"*. You need `pip install -e .` from inside the repo folder. Without the command you can always run the script directly (the dependencies must still be installed): `python src/voice_recorder_cli.py --help`.

### Windows notes

- **"The term 'transcripter' is not recognized"**: the package was not installed into the active venv. Run `pip install -e .` from the repo folder. If it still fails, use `python -m pip install -e .` so pip and python are guaranteed to be the same install, and check `dir venv\Scripts\transcripter*` shows `transcripter.exe`.
- **PowerShell refuses to run `Activate.ps1`** ("running scripts is disabled"): run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use cmd and `activate.bat` instead.
- **Venv not activated**: you can still run `venv\Scripts\transcripter.exe` directly.
- If you move or rename the folder after installing, recreate the venv and reinstall (the command's launcher stores absolute paths).

## Usage

```bash
transcripter                       # record until Ctrl+C, then transcribe
transcripter --output-dir ~/notes  # default destination directory
transcripter --md-path ~/notes/review.md  # WAV goes beside it: review.wav
transcripter --audio-path ~/audio/review.wav --md-path ~/notes/review.md
transcripter --prefix team- --srt  # prefix every output filename
transcripter --gpu --diarize --speakers 2  # prefer CUDA; falls back to CPU
transcripter --model tiny          # faster, less accurate (tiny, base, small, medium, large)
```

Each recording produces files that share one base name:

```
2026-10-16_22-48-48.wav   the audio (16 kHz mono)
2026-10-16_22-48-48.md    the transcript
2026-10-16_22-48-48.srt   captions (only with --srt)
```

If a name is already taken, `_1`, `_2`... is appended to the filename(s); nothing is ever overwritten.

`--audio-path` and `--md-path` accept full output filenames (a missing `.wav` or `.md` extension is added). If only one is given, the other file uses the same directory and filename stem with its own extension. If both are given, each file goes to its specified path. The `.srt`, when enabled, is placed beside the Markdown file. Explicit paths override `--output-dir`; relative paths are relative to the current working directory. The Markdown audio link is made relative to the transcript, even when the files are in different folders.

`--prefix TEXT` prepends `TEXT` to each output filename's basename, without changing its directory. It applies to generated names and explicit paths. The prefix must not contain a directory separator.

| Option | What it does |
|---|---|
| `--output-dir DIR`, `-o` | Default directory for generated output (explicit file paths override it) |
| `--audio-path PATH` | Output WAV filename; if used alone, the Markdown file shares its stem and directory |
| `--md-path PATH` | Output Markdown filename; if used alone, the WAV file shares its stem and directory |
| `--prefix TEXT` | Prefix each output filename (not its directory) |
| `--model SIZE`, `-m` | Whisper model: `tiny`, `base`, `small` (default), `medium`, `large` |
| `--gpu` | Prefer NVIDIA CUDA for transcription and enabled diarization; each stage falls back to CPU if unavailable |
| `--srt` | Also write `<name>.srt` closed captions (and show them on screen when done) |
| `--no-preview` | Do not display the caption file after transcribing (shown by default with `--srt`) |
| `--diarize` | Label speakers (needs `sherpa-onnx`) |
| `--speakers N`, `-n` | Number of speakers, if known (implies `--diarize`) |
| `--no-audio` | Do not keep the `.wav` (cannot be combined with `--audio-path`) |
| `--from-file WAV` | Transcribe an existing recording instead of using the mic; cannot be combined with `--audio-path` |
| `--no-ui` | Plain text output: no colours, no screen clear (automatic when output is piped or not a terminal) |
| `--version`, `-V` | Print the version and exit |

### GPU acceleration (`--gpu`)

CPU remains the default. With `--gpu`, faster-whisper asks CTranslate2 to use CUDA; before recording, it runs a tiny warm-up inference to catch missing CUDA/cuBLAS libraries. If that probe fails, it warns and loads the CPU INT8 model instead. If `--diarize` is also enabled, the segmentation and speaker-embedding ONNX sessions also request CUDA and fall back to CPU with a warning if the CUDA-enabled sherpa-onnx build is unavailable. If the entire diarization package is missing, the existing behavior still continues without speaker labels.

This is for compatible NVIDIA CUDA systems, currently the documented Linux/Windows x64 builds. The standard `pip install -e ".[diarize]"` installs the regular CPU sherpa-onnx wheel; GPU diarization needs an official CUDA-enabled `+cuda` wheel and matching CUDA/cuDNN runtime. The app recognizes that build and avoids sending CUDA requests to a CPU-only sherpa-onnx package (which can silently fall back inside the library). Reinstalling Transcripter does not install CUDA DLLs: an error such as `cublas64_12.dll is not found` means the CUDA/cuBLAS runtime required by CTranslate2 is missing or not on `PATH`; either install the matching runtime or omit `--gpu`. Follow the upstream [CTranslate2 GPU requirements](https://opennmt.net/CTranslate2/installation.html) and [sherpa-onnx CUDA Python installation instructions](https://k2-fsa.github.io/sherpa/onnx/python/install.html) for the versions supported by your system. The Pi’s VideoCore GPU and the Mac’s Apple GPU are not CUDA devices, so they continue on CPU; Apple CoreML is not part of this option.

### While recording

The screen is cleared when the app starts, so the recording display is the only thing on screen (your scrollback is kept). This only happens on an interactive terminal: with `--no-ui`, or when output is piped or redirected, nothing is cleared and no escape codes are written.

On a terminal you get a live display: a blinking `● REC`, the elapsed time (`MM:SS`, `HH:MM:SS` past an hour), a green-to-red level meter and 16 frequency bars. If no signal arrives for 3 seconds it warns you ("No input signal - is the microphone connected and unmuted?"), and it flags clipping. Press **Ctrl+C** to stop.

Using the existing signal threshold (peak above about **-54 dBFS**), if the microphone remains silent for **20 seconds**, an automatic **30-second stop countdown** starts. The live display shows a highlighted **[C] Cancel countdown** control—press `C` (no Enter needed) to cancel it. If audio resumes, the countdown clears; after cancellation it will only arm again once audio has returned and silence is detected anew. In non-interactive runs the countdown still stops the recording, but there is no keyboard cancel control.

### While transcribing

Real progress bars (audio processed / total, speed as a multiple of real-time, elapsed and estimated time remaining) for transcription and, with `--diarize`, for speaker identification. The bar advances as each piece of speech is finished, so it moves in steps rather than smoothly.

### Closed captions (`--srt`)

```bash
transcripter --srt
```

Writes a SubRip file: numbered cues, `HH:MM:SS,mmm --> HH:MM:SS,mmm` timing, UTF-8, CRLF line endings. Cue times start at 0 and match the saved `.wav`. Cues follow standard caption conventions: at most 2 lines of 42 characters, 1-7 seconds on screen, never overlapping, broken at pauses and sentence ends. With `--diarize`, the first cue of each turn starts with `Speaker N:`.

When the file is written, it is displayed on screen as the last thing you see: a table of cue number, start, end and caption text with the exact timestamps from the file, speaker labels in a colour per speaker. On a very long recording only the first cues that fit your terminal are shown, followed by "... N more cues in <file>". Piped or logged output (or `--no-ui`) gets a compact one-line-per-cue listing instead. Add `--no-preview` to skip the display; the file is still written.

### Speaker labels (`--diarize`)

```bash
transcripter --diarize --speakers 2
```

```
**Speaker 1 (00:00)**

Hi Sanjay, can you help me understand what Generative AI is?

**Speaker 2 (00:14)**

Sure. It refers to models that can generate new content...
```

- Runs **locally** after you press Ctrl+C. Two small ONNX models (~30 MB) download to `~/.cache/transcripter/` on first use. No torch, no accounts.
- **Pass `--speakers N` when you know the count.** Auto-detect tends to under-count.
- Labels are **per recording**: "Speaker 1" in one file is not necessarily the same person as "Speaker 1" in another.
- Limits: overlapping speech is not separated, and turn boundaries can be off by a second or two, especially on very short replies ("Yeah, sure").
- If diarization cannot load (e.g. no network on first use), you still get the normal unlabelled transcript.

### Re-processing a recording (`--from-file`)

Every recording is saved, so you can transcribe it again with a different model or speaker count without re-recording:

```bash
transcripter --from-file 2026-10-16_22-48-48.wav --model medium --speakers 3 --srt
```

Accepts 16-bit PCM WAV at any sample rate and channel count. Output is named after the input file.

## Long meetings (over an hour)

- **Recording is cheap.** Audio is written straight to disk as it arrives (about 1.9 MB per minute, ~115 MB per hour). Nothing accumulates in memory, and the WAV header is refreshed every 2 seconds, so even a power cut leaves a playable file up to the last couple of seconds.
- **The WAV is on disk before transcription starts.** If transcription fails or you abort it with a second Ctrl+C, the audio is kept and the message tells you the exact `--from-file` command to retry.
- **Transcription is chunked.** Recordings over 15 minutes are processed in ~10 minute pieces, cut at the quietest moment near each boundary, so memory stays flat however long the meeting was.
- **Processing time is the real cost.** It scales with the length of the meeting and the model size. See the measurements below; on slow hardware prefer `--model base` (or `tiny`) for meetings, or record now and run `--from-file` later.

### Measured on a Raspberry Pi (4 cores, CPU only)

A 17.5 minute recording, run through the real pipeline:

| Run | Wall time | Speed | Peak memory |
|---|---|---|---|
| `--model base` | 8 min 7 s | 0.46x real-time | 805 MB |
| `--model tiny --speakers 2` | 8 min 57 s | 0.51x real-time | 843 MB |
| ...of which speaker identification alone | 4 min 48 s | 0.27x real-time | 350 MB |
| ...of which `tiny` Whisper with word timestamps | 4 min 14 s | 0.24x real-time | |

Both stages scale roughly linearly with length, so a 2-hour meeting should take about 55 minutes with `--model base`, and about an hour with `tiny` plus speaker labels (extrapolated, not measured at that length). `--model small` is much slower: on a 55 s clip with speaker labels it ran at about 1.3x real-time, which would be well over two hours for a two-hour meeting. Word timestamps (needed for `--srt` and `--diarize`) add noticeable time on top of plain transcription.

Memory grows only with the 16-bit audio kept for processing (about 115 MB per hour), not with the number of chunks.

## Version

```bash
transcripter --version     # transcripter 1.2.0
```

The version is stored in one place: `__version__` near the top of `src/voice_recorder_cli.py`. `setup.py` reads it from there, so the installed package metadata, `--version` and the banner always agree. To release a new version, change that one line (and re-run `pip install -e .` so installed metadata updates).

## Output format

```markdown
# Voice Recording Transcript

**Recorded:** 2026-10-16 22:48:48  
**Duration:** 01:12:05  
**Model:** faster-whisper small + speaker diarization  
**Audio:** [2026-10-16_22-48-48.wav](2026-10-16_22-48-48.wav)  
**Captions:** [2026-10-16_22-48-48.srt](2026-10-16_22-48-48.srt)  

---

[transcribed text here]
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | No microphone found / audio capture failed / unreadable input file |
| 2 | Disk full / I/O error saving output |
| 3 | Invalid model size / model load failed |
| 4 | Transcription failed (the audio is kept) |
| 130 | Interrupted while processing (the audio is kept) |

## Architecture

```
src/voice_recorder_cli.py
├── WavWriter / LinearResampler / read_wav_as_16k_mono   # WAV streaming + loading
├── AudioRecorder   # sounddevice callback -> queue -> writer thread -> .wav (+ live level snapshot)
├── plan_chunks / Transcriber   # chunked faster-whisper transcription with progress
├── Diarizer        # sherpa-onnx speaker turns
├── label_words / group_utterances   # word-level speaker alignment
├── build_cues / render_srt          # caption cues -> SubRip
├── RichUI / PlainUI / RecordingView # live display, progress bars, plain fallback
└── VoiceRecorder   # the app: record -> save wav -> transcribe -> write md/srt
```

## Tests

```bash
pip install pytest
python -m pytest                       # everything (a couple of minutes on a Raspberry Pi)
python -m pytest -m "not slow and not integration"   # fast unit tests only
```

The suite uses a fake microphone and a fake Whisper model for the logic (WAV streaming and crash recovery, chunk planning, speaker alignment, SRT formatting, the UI), plus integration tests that run the real models on public two-speaker clips. Those download test clips and need the Whisper `tiny` model cached; they are skipped when it is not available. SRT output is also checked with `ffmpeg` when installed.

## Troubleshooting

### "No input devices found"

No microphone detected. Check that it is connected and enabled in system settings, the audio drivers are installed, and you have permission to use it (especially on Linux).

### The live display warns "No input signal"

Audio is arriving but silent: the mic is muted, the wrong device is selected, or the input gain is zero.

### Transcription fails or is interrupted

Your audio is already saved. Re-run with the `--from-file` command that was printed.

### Slow performance

Use a smaller model (`--model tiny` or `base`). Speaker labels add time on top of transcription.

## License

MIT
