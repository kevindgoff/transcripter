"""GPU preference, CPU fallback, and CUDA provider wiring."""
import sys
import types
from argparse import Namespace
import pytest

import voice_recorder_cli as v


def install_fake_whisper(monkeypatch, cuda_devices=1, gpu_error=None, cpu_error=None):
    calls = []

    class FakeWhisperModel:
        def __init__(self, model_size, device, compute_type):
            calls.append((model_size, device, compute_type))
            error = gpu_error if device == "cuda" else cpu_error
            if error:
                raise RuntimeError(error)
            self.model_size = model_size

    ctranslate2 = types.ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: cuda_devices
    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    return calls


def test_cpu_default_keeps_existing_cpu_int8_configuration(monkeypatch):
    calls = install_fake_whisper(monkeypatch)
    t = v.Transcriber("tiny")
    t.load_model()
    assert calls == [("tiny", "cpu", "int8")]
    assert t.device == "cpu" and t.compute_type == "int8"
    assert t.gpu_fallback_reason is None


def test_gpu_transcription_uses_cuda_when_available(monkeypatch):
    calls = install_fake_whisper(monkeypatch, cuda_devices=1)
    t = v.Transcriber("small", use_gpu=True)
    t.load_model()
    assert calls == [("small", "cuda", "auto")]
    assert t.device == "cuda" and t.compute_type == "auto"
    assert t.gpu_fallback_reason is None


def test_no_cuda_device_falls_back_to_cpu_int8(monkeypatch):
    calls = install_fake_whisper(monkeypatch, cuda_devices=0)
    t = v.Transcriber("tiny", use_gpu=True)
    t.load_model()
    assert calls == [("tiny", "cpu", "int8")]
    assert t.device == "cpu" and t.compute_type == "int8"
    assert "no cuda device" in t.gpu_fallback_reason.lower()


def test_cuda_initialization_error_retries_whisper_on_cpu(monkeypatch):
    calls = install_fake_whisper(monkeypatch, cuda_devices=1, gpu_error="CUDA libraries missing")
    t = v.Transcriber("tiny", use_gpu=True)
    t.load_model()
    assert calls == [("tiny", "cuda", "auto"), ("tiny", "cpu", "int8")]
    assert t.device == "cpu" and "CUDA libraries missing" in t.gpu_fallback_reason


def test_cpu_load_failure_after_gpu_failure_is_reported(monkeypatch):
    install_fake_whisper(monkeypatch, cuda_devices=1, gpu_error="GPU unavailable", cpu_error="model broken")
    with pytest.raises(RuntimeError, match="model broken"):
        v.Transcriber("tiny", use_gpu=True).load_model()


def make_fake_sherpa(monkeypatch, reject_cuda=False, cuda_build=True):
    attempts = []

    class Pyannote:
        def __init__(self, model, window_shift_ratio):
            self.model, self.window_shift_ratio = model, window_shift_ratio

    class Segmentation:
        def __init__(self, pyannote, num_threads, provider="cpu"):
            self.pyannote, self.num_threads, self.provider = pyannote, num_threads, provider

    class Embedding:
        def __init__(self, model, num_threads, provider="cpu"):
            self.model, self.num_threads, self.provider = model, num_threads, provider

    class Clustering:
        def __init__(self, num_clusters, threshold):
            self.num_clusters, self.threshold = num_clusters, threshold

    class Config:
        def __init__(self, segmentation, embedding, clustering, min_duration_on, min_duration_off):
            self.segmentation, self.embedding, self.clustering = segmentation, embedding, clustering
            self.min_duration_on, self.min_duration_off = min_duration_on, min_duration_off

        def validate(self):
            return True

    class Session:
        def __init__(self, config):
            attempts.append((config.segmentation.provider, config.embedding.provider))
            if reject_cuda and config.segmentation.provider == "cuda":
                raise RuntimeError("CUDA execution provider unavailable")
            self.sample_rate = v.TARGET_RATE

    sherpa = types.ModuleType("sherpa_onnx")
    sherpa.__version__ = "1.13.8+cuda12.cudnn9" if cuda_build else "1.13.8"
    sherpa.OfflineSpeakerSegmentationPyannoteModelConfig = Pyannote
    sherpa.OfflineSpeakerSegmentationModelConfig = Segmentation
    sherpa.SpeakerEmbeddingExtractorConfig = Embedding
    sherpa.FastClusteringConfig = Clustering
    sherpa.OfflineSpeakerDiarizationConfig = Config
    sherpa.OfflineSpeakerDiarization = Session
    monkeypatch.setitem(sys.modules, "sherpa_onnx", sherpa)
    monkeypatch.setattr(v, "ensure_diarization_models", lambda: None)
    return attempts


def test_diarizer_requests_cuda_for_both_onnx_stages(monkeypatch):
    attempts = make_fake_sherpa(monkeypatch)
    d = v.Diarizer(num_speakers=2, use_gpu=True)
    d.load()
    assert attempts == [("cuda", "cuda")]
    assert d.provider == "cuda" and d.gpu_fallback_reason is None


def test_diarizer_provider_failure_retries_both_stages_on_cpu(monkeypatch):
    attempts = make_fake_sherpa(monkeypatch, reject_cuda=True)
    d = v.Diarizer(num_speakers=2, use_gpu=True)
    d.load()
    assert attempts == [("cuda", "cuda"), ("cpu", "cpu")]
    assert d.provider == "cpu"
    assert "CUDA execution provider unavailable" in d.gpu_fallback_reason


def test_diarizer_cpu_default_does_not_attempt_cuda(monkeypatch):
    attempts = make_fake_sherpa(monkeypatch)
    d = v.Diarizer(num_speakers=2)
    d.load()
    assert attempts == [("cpu", "cpu")]
    assert d.provider == "cpu" and d.gpu_fallback_reason is None


def test_cpu_only_sherpa_build_skips_cuda_attempt_and_uses_cpu(monkeypatch):
    attempts = make_fake_sherpa(monkeypatch, cuda_build=False)
    d = v.Diarizer(num_speakers=2, use_gpu=True)
    d.load()
    assert attempts == [("cpu", "cpu")]
    assert d.provider == "cpu" and "not a CUDA-enabled build" in d.gpu_fallback_reason



def test_app_passes_gpu_preference_to_both_and_reports_fallback(monkeypatch, tmp_path, capsys):
    seen = {}

    class FakeTranscriber:
        def __init__(self, model_size, use_gpu=False):
            seen["transcriber_gpu"] = use_gpu
            self.gpu_fallback_reason = "no CUDA device"
            self.device = "cpu"

        def load_model(self):
            pass

    class FakeDiarizer:
        def __init__(self, num_speakers=None, use_gpu=False):
            seen["diarizer_gpu"] = use_gpu
            self.gpu_fallback_reason = "CUDA provider unavailable"
            self.provider = "cpu"

        def load(self):
            pass

    monkeypatch.setattr(v, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(v, "Diarizer", FakeDiarizer)
    app = v.VoiceRecorder(v.PlainUI())
    app.setup(Namespace(output_dir=str(tmp_path), model="tiny", gpu=True, diarize=True,
                        speakers=2, srt=False, no_preview=False, no_audio=False))
    output = capsys.readouterr().out
    assert seen == {"transcriber_gpu": True, "diarizer_gpu": True}
    assert "CUDA unavailable for transcription" in output and "using CPU" in output
    assert "CUDA unavailable for diarization" in output and "using CPU" in output


def test_cli_gpu_flag_defaults_off_and_reaches_setup(monkeypatch):
    seen = []
    monkeypatch.setattr(v.VoiceRecorder, "setup", lambda self, args: seen.append(args.gpu))
    monkeypatch.setattr(v.VoiceRecorder, "run", lambda self, args: None)
    for argv, expected in (([], False), (["--gpu"], True)):
        monkeypatch.setattr(sys, "argv", ["transcripter", "--no-ui", *argv])
        v.main()
        assert seen[-1] is expected


def test_cli_help_explains_gpu_fallback(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["transcripter", "--help"])
    with pytest.raises(SystemExit):
        v.main()
    help_text = capsys.readouterr().out
    assert "--gpu" in help_text and "CPU" in help_text
