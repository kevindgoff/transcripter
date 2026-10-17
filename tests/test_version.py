"""Version reporting (--version, banner, packaging metadata) and the startup screen clear."""
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import voice_recorder_cli as v

pytest.importorskip("rich")
from rich.console import Console  # noqa: E402

REPO = Path(v.__file__).resolve().parent.parent


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["transcripter", *argv])
    v.main()


# ---- version ----------------------------------------------------------------------

def test_version_is_a_valid_three_part_version_of_at_least_1_0():
    """Deliberately not pinned to a literal, so a release does not need a test edit."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", v.__version__)
    assert tuple(int(p) for p in v.__version__.split(".")) >= (1, 0, 0)


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_and_exits_cleanly(monkeypatch, capsys, flag):
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, flag)
    assert e.value.code == 0
    out = capsys.readouterr()
    assert out.out.strip() == f"transcripter {v.__version__}" and out.err == ""


def test_version_flag_needs_nothing_else(monkeypatch, capsys):
    """No models, no mic, no output directory touched just to print a version."""
    monkeypatch.chdir(REPO)
    before = sorted(p.name for p in REPO.iterdir())
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--version", "--from-file", "/does/not/exist.wav")
    assert sorted(p.name for p in REPO.iterdir()) == before


def test_help_lists_version_flag(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--help")
    assert "--version" in capsys.readouterr().out


def test_banner_shows_the_version_in_both_uis(capsys):
    v.PlainUI().banner()
    assert f"v{v.__version__}" in capsys.readouterr().out
    buf = io.StringIO()
    v.RichUI(Console(file=buf, force_terminal=True, width=80)).banner()
    assert f"v{v.__version__}" in re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


@pytest.mark.integration
@pytest.mark.slow
def test_installed_package_has_the_same_version_and_a_working_command(tmp_path):
    """Install into a throwaway venv (what the README tells users to do) and check
    that package metadata, `transcripter --version` and __version__ all agree."""
    env_dir = tmp_path / "venv"
    pkg = tmp_path / "pkg"  # build from a copy: pip builds in-tree and must not litter the repo
    shutil.copytree(REPO, pkg, ignore=shutil.ignore_patterns("venv", ".git", "build", "*.egg-info", "__pycache__",
                                                             ".pytest_cache", "tests"))
    subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True, timeout=120)
    bindir = env_dir / ("Scripts" if os.name == "nt" else "bin")
    py = bindir / ("python.exe" if os.name == "nt" else "python")
    r = subprocess.run([str(py), "-m", "pip", "install", "-q", "--no-deps", str(pkg)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        pytest.skip(f"could not build/install the package (offline?): {r.stderr[-300:]}")
    cmd = bindir / ("transcripter.exe" if os.name == "nt" else "transcripter")
    assert cmd.exists(), "pip install did not create the `transcripter` command"
    out = subprocess.run([str(cmd), "--version"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == f"transcripter {v.__version__}", out.stdout + out.stderr
    meta = subprocess.run([str(py), "-c", "from importlib.metadata import version; print(version('transcripter'))"],
                          capture_output=True, text=True, timeout=60)
    assert meta.stdout.strip() == v.__version__


# ---- clearing the screen ---------------------------------------------------------------

def test_plain_ui_never_writes_escape_codes_when_clearing(capsys):
    v.PlainUI().clear()
    assert capsys.readouterr().out == ""


def test_rich_ui_clears_a_real_terminal():
    buf = io.StringIO()
    v.RichUI(Console(file=buf, force_terminal=True, width=80)).clear()
    assert "\x1b[2J" in buf.getvalue() and "\x1b[H" in buf.getvalue()  # erase screen + cursor home


def test_rich_ui_does_not_clear_when_output_is_not_a_terminal():
    buf = io.StringIO()
    v.RichUI(Console(file=buf, force_terminal=False, width=80)).clear()
    assert buf.getvalue() == ""


class SpyUI(v.PlainUI):
    calls = []

    def clear(self):
        SpyUI.calls.append("clear")


def test_screen_is_cleared_before_anything_else_happens(monkeypatch):
    """clear -> setup (model loading, 'Output directory: ...') -> run."""
    SpyUI.calls = []
    monkeypatch.setattr(v, "make_ui", lambda no_ui=False, console=None: SpyUI())
    monkeypatch.setattr(v.VoiceRecorder, "setup", lambda self, args: SpyUI.calls.append("setup"))
    monkeypatch.setattr(v.VoiceRecorder, "run", lambda self, args: SpyUI.calls.append("run"))
    run_main(monkeypatch)
    assert SpyUI.calls == ["clear", "setup", "run"]


def test_bad_arguments_do_not_wipe_the_screen_first(monkeypatch, capsys):
    """An argument error must stay visible, so the clear happens only after validation."""
    SpyUI.calls = []
    monkeypatch.setattr(v, "make_ui", lambda no_ui=False, console=None: SpyUI())
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--speakers", "0")
    assert SpyUI.calls == []
    assert "--speakers must be >= 1" in capsys.readouterr().err


def test_no_ui_run_leaves_the_terminal_alone(monkeypatch, capsys):
    monkeypatch.setattr(v.VoiceRecorder, "setup", lambda self, args: None)
    monkeypatch.setattr(v.VoiceRecorder, "run", lambda self, args: None)
    run_main(monkeypatch, "--no-ui")
    assert "\x1b" not in capsys.readouterr().out
