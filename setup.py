import re
from pathlib import Path

from setuptools import setup, find_packages


def read_version() -> str:
    """The version lives in exactly one place: __version__ in src/voice_recorder_cli.py.
    Parsed as text (not imported) so building never needs the runtime dependencies."""
    text = (Path(__file__).parent / "src" / "voice_recorder_cli.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("__version__ not found in src/voice_recorder_cli.py")
    return match.group(1)


setup(
    name="transcripter",
    version=read_version(),
    description="Real-time audio transcription tool using local Whisper models",
    author="Coder",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "sounddevice>=0.5.0",
        "faster-whisper>=1.0.0",
        "numpy>=1.24.0",
        "rich>=13.0",  # live recording display + progress bars; the CLI falls back to plain text without it
    ],
    extras_require={
        # Speaker labels (--diarize). Optional: imported lazily, so base installs stay light.
        "diarize": ["sherpa-onnx>=1.10"],
        "test": ["pytest>=7"],
    },
    entry_points={
        "console_scripts": [
            "transcripter=src.voice_recorder_cli:main",
        ],
    },
)
