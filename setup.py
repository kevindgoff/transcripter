from setuptools import setup, find_packages

setup(
    name="transcripter",
    version="0.1.0",
    description="Real-time audio transcription tool using local Whisper models",
    author="Coder",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "sounddevice>=0.5.0",
        "faster-whisper>=1.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        # Speaker labels (--diarize). Optional: imported lazily, so base installs stay light.
        "diarize": ["sherpa-onnx>=1.10"],
    },
    entry_points={
        "console_scripts": [
            "transcripter=src.voice_recorder_cli:main",
        ],
    },
)
