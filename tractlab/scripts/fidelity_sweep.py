#!/usr/bin/env python3
"""CLI shim: python scripts/fidelity_sweep.py --case-root <dir>"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tractlab.fidelity_sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
