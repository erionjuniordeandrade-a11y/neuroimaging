"""The synthetic QA server must carry the same identity meta tags as the real viewer."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "serve_synthetic.py"


def test_synthetic_index_carries_build_and_protocol_meta():
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--port", "0"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert line, proc.stderr.read()
        url = json.loads(line)["url"]
        with urlopen(url, timeout=10) as response:
            page = response.read().decode("utf-8")
        assert '<meta name="tractlab-build"' in page
        assert '<meta name="tractlab-protocol"' in page
        assert "data-synthetic-guard" in page
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
