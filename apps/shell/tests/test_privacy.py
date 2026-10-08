"""The privacy banner states only measured facts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from neuro_workbench import privacy


def _fake_run(stdout):
    return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")


def test_filevault_parses_on_off_and_unknown(monkeypatch):
    monkeypatch.setattr(privacy.sys, "platform", "darwin")
    monkeypatch.setattr(privacy.shutil, "which", lambda _: "/bin/sh")
    for text, want in (("FileVault is On.\n", "on"), ("FileVault is Off.\n", "off"), ("odd\n", "unknown")):
        monkeypatch.setattr(privacy.subprocess, "run", _fake_run(text))
        assert privacy.filevault() == want
    monkeypatch.setattr(privacy.sys, "platform", "linux")
    assert privacy.filevault() == "unknown"


def test_report_measures_the_folder_mode(tmp_path: Path):
    root = tmp_path / "archive"
    root.mkdir()
    os.chmod(root, 0o700)
    facts = privacy.report(root, "on")
    assert facts["folder_private"] is True and facts["loopback"] is True and facts["filevault"] == "on"
    assert isinstance(facts["same_volume_as_home"], bool)
    os.chmod(root, 0o755)
    assert privacy.report(root, "on")["folder_private"] is False
    assert privacy.report(None, "off") == {"loopback": True, "filevault": "off"}
