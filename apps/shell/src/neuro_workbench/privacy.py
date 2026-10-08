"""What the app can measure about how the patient archive is protected.

The page states these facts and nothing more. Each value is measured, not
assumed: FileVault from `fdesetup status`, the folder mode from the file
system, and whether the archive is on the same volume as the home folder
(the volume FileVault encrypts). The server only listens on 127.0.0.1.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def filevault() -> str:
    """"on", "off" or "unknown" (not macOS, tool missing, or an answer we do not recognise)."""
    tool = shutil.which("fdesetup") or "/usr/bin/fdesetup"
    if sys.platform != "darwin" or not Path(tool).exists():
        return "unknown"
    try:
        out = subprocess.run([tool, "status"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if "FileVault is On" in out:
        return "on"
    if "FileVault is Off" in out:
        return "off"
    return "unknown"


def report(root: Path | None, vault: str) -> dict:
    """Facts for the banner. `vault` is measured once at startup and passed in."""
    facts: dict = {"loopback": True, "filevault": vault}
    if root is None:
        return facts
    try:
        mode = stat.S_IMODE(os.stat(root).st_mode)
        facts["folder_private"] = mode & 0o077 == 0
        facts["same_volume_as_home"] = os.stat(root).st_dev == os.stat(Path.home()).st_dev
    except OSError:
        facts["folder_private"] = False
        facts["same_volume_as_home"] = False
    return facts
