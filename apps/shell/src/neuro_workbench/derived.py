"""Products derived from one archive study: a capsule (imaging) and a TractLab case (tracts).

One data model: patient -> study -> derived products. A product lives in
<archive>/derived/<case-id>/ and is deleted with its study. Each product has a
JSON-visible state; the page offers the next action for that state.

Capsule states: absent, empty (no usable CT/MR series), running, ready, error.
Tracts states: nodti, unavailable (pipeline tools missing), absent, running, ready, error.
Messages never carry paths, series descriptions or patient text.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from neuro_workbench.archive import DIFFUSION

CAPSULE_NAME = "study.capsule.html"
MAX_CAPSULE_SERIES = 4
MIN_CAPSULE_IMAGES = 3
SKIP_SERIES = re.compile(r"localizer|scout|survey|loc\b|topogram|screen ?save|dose report|3-plane|tri.?plane", re.I)
T1_LIKE = re.compile(r"t1|mprage|bravo|spgr|tfe|fspgr|vibe", re.I)
PIPELINE_STAGES = 4  # ss3t, t1reg, banks, scalar_maps


def _home(*parts: str) -> Path:
    return Path.home().joinpath(*parts)


def pipeline_tools() -> dict[str, Path]:
    """Tools run_case.sh needs, at the paths it uses (overridable by the same variables)."""
    fsl = Path(os.environ.get("FSLDIR", _home("fsl")))
    return {
        "dcm2niix": Path(os.environ.get("DCM2NIIX") or shutil.which("dcm2niix") or fsl / "bin" / "dcm2niix"),
        "FSL python": Path(os.environ.get("TRACTLAB_PYTHON", fsl / "bin" / "python")),
        "MRtrix3": Path(os.environ.get("MRTRIX3", _home("mrtrix3", "bin"))) / "tckgen",
        "ANTs": Path(os.environ.get("ANTSPATH", _home("ants-2.6.5", "bin"))) / "antsRegistrationSyN.sh",
        "FreeSurfer": Path(os.environ.get("FREESURFER_HOME", _home("freesurfer"))) / "SetUpFreeSurfer.sh",
    }


def missing_tools() -> list[str]:
    return [name for name, path in pipeline_tools().items() if not path.exists()]


def capsule_series(series: list[dict]) -> list[int]:
    """Series numbers for the capsule: CT/MR volumes, no diffusion or its maps, no localizers. Reference first."""
    counts: dict[int, int] = {}
    for s in series:
        counts[s["number"]] = counts.get(s["number"], 0) + 1
    picked = [s for s in series
              if (s.get("modality") or "").upper() in ("CT", "MR") and s["number"] is not None
              and counts[s["number"]] == 1 and s["images"] >= MIN_CAPSULE_IMAGES and not s.get("dti")
              and not SKIP_SERIES.search(s.get("description") or "") and not DIFFUSION.search(s.get("description") or "")]
    picked.sort(key=lambda s: (not ((s["modality"] or "").upper() == "CT" or T1_LIKE.search(s["description"] or "")),
                               -s["images"]))
    return [s["number"] for s in picked[:MAX_CAPSULE_SERIES]]


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return (lines[-1] if lines else "no output")[:200]


class Derived:
    """Builds and reports the products of archive studies. One build of each kind at a time."""

    def __init__(self, repo: Path, archive):
        self.repo = repo
        self.archive = archive
        self._lock = threading.Lock()
        self._running: dict[str, str] = {}  # kind -> case id
        self._errors: dict[tuple[str, str], str] = {}
        self._procs: dict[str, subprocess.Popen] = {}

    # Paths ---------------------------------------------------------------
    def _dir(self, case_id: str, kind: str) -> Path:
        return self.archive.derived_dir(case_id) / kind

    def capsule_file(self, case_id: str) -> Path | None:
        f = self._dir(case_id, "capsule") / CAPSULE_NAME
        return f if f.is_file() else None

    def tract_case(self, case_id: str) -> Path | None:
        root = self._dir(case_id, "tracts")
        found = sorted(p.parent for p in root.glob("case-*/case.json")) if root.is_dir() else []
        return found[0] if found else None

    def manifest(self, case_id: str) -> Path | None:
        case = self.tract_case(case_id)
        status = self._read_status(case) if case else {}
        m = case / "manifest.json" if case else None
        return m if m and status.get("state") == "done" and m.is_file() else None

    @staticmethod
    def _read_status(case: Path) -> dict:
        try:
            return json.loads((case / "status.json").read_text())
        except (OSError, ValueError):
            return {}

    # Status --------------------------------------------------------------
    def status(self, case_id: str) -> dict:
        uid = self.archive.study_uid(case_id)
        series = self.archive.study_series(uid) if uid else []
        return {"capsule": self._capsule_status(case_id, series), "tracts": self._tracts_status(case_id, series)}

    def _capsule_status(self, case_id: str, series: list[dict]) -> dict:
        numbers = capsule_series(series)
        out = {"series": numbers}
        if self._running.get("capsule") == case_id:
            return {**out, "state": "running"}
        if self.capsule_file(case_id):
            return {**out, "state": "ready"}
        if (case_id, "capsule") in self._errors:
            return {**out, "state": "error", "error": self._errors[(case_id, "capsule")]}
        return {**out, "state": "absent" if numbers else "empty"}

    def _tracts_status(self, case_id: str, series: list[dict]) -> dict:
        dti = [{"number": s["number"], "images": s["images"]} for s in series if s.get("dti")]
        out: dict = {"dti": dti}
        case = self.tract_case(case_id)
        st = self._read_status(case) if case else {}
        stages = st.get("stages") or []
        done = sum(1 for s in stages if s.get("state") == "done")
        progress = {"stage": st.get("stage"), "done": done, "total": len(stages) or PIPELINE_STAGES}
        if self._running.get("tracts") == case_id:
            return {**out, **progress, "state": "running"}
        if st.get("state") == "done" and self.manifest(case_id):
            return {**out, "state": "ready"}
        if (case_id, "tracts") in self._errors:
            return {**out, **(progress if case else {}), "state": "error", "error": self._errors[(case_id, "tracts")]}
        if st.get("state") == "failed":
            return {**out, **progress, "state": "error", "error": f"stage {st.get('stage') or '?'} failed; see the case log"}
        if not dti:
            return {**out, "state": "nodti"}
        missing = missing_tools()
        if missing:
            return {**out, "state": "unavailable", "missing": missing}
        return {**out, **({"resume": True, **progress} if case else {}), "state": "absent"}

    # Builds --------------------------------------------------------------
    def build(self, case_id: str, kind: str) -> dict:
        """Start one build. Raises RuntimeError when it cannot start (the page shows the message)."""
        if kind not in ("capsule", "tracts"):
            raise ValueError("unknown product")
        uid = self.archive.study_uid(case_id)
        if uid is None:
            raise KeyError(case_id)
        current = self.status(case_id)[kind]
        if current["state"] in ("ready", "running"):
            return current
        if kind == "capsule" and current["state"] == "empty":
            raise RuntimeError("this study has no CT or MR series a capsule can use")
        if kind == "tracts" and current["state"] == "nodti":
            raise RuntimeError("this study has no diffusion series")
        if kind == "tracts" and current["state"] == "unavailable":
            raise RuntimeError("pipeline tools are missing: " + ", ".join(current["missing"]))
        with self._lock:
            if kind in self._running:
                raise RuntimeError(f"another {kind} build is running; wait for it to finish")
            self._running[kind] = case_id
            self._errors.pop((case_id, kind), None)
        target = self._build_capsule if kind == "capsule" else self._build_tracts
        threading.Thread(target=self._guard, args=(case_id, kind, target, uid), daemon=True).start()
        return self.status(case_id)[kind]

    def _guard(self, case_id, kind, target, uid):
        try:
            target(case_id, uid)
        except Exception as exc:  # shown on the page: one short line, no traceback
            self._errors[(case_id, kind)] = (str(exc) or exc.__class__.__name__)[:200]
        finally:
            with self._lock:
                self._running.pop(kind, None)
                self._procs.pop(kind, None)

    def _run(self, kind: str, cmd: list[str], cwd: Path, log: Path, env: dict | None = None) -> int:
        with open(log, "w") as fh:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            self._procs[kind] = proc
            return proc.wait()

    def _private_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        for p in (path, path.parent):
            os.chmod(p, 0o700)
        return path

    def _build_capsule(self, case_id: str, uid: str) -> None:
        series = capsule_series(self.archive.study_series(uid))
        out = self._private_dir(self._dir(case_id, "capsule"))
        tmp = out / f"{CAPSULE_NAME}.part"
        log = out / "build.log"
        cmd = [sys.executable, "-m", "capsule.cli", "build", str(self.archive.study_dicom_dir(uid)),
               "--series", ",".join(str(n) for n in series), "--label", "Archive study",
               "-o", str(tmp), "--anatomy", "none", "--brain-mask", "none"]
        code = self._run("capsule", cmd, self.repo / "capsule", log)
        if code != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"capsule build failed (exit {code}): {_last_line(log.read_text(errors='replace'))}")
        os.replace(tmp, out / CAPSULE_NAME)

    def _build_tracts(self, case_id: str, uid: str) -> None:
        root = self._private_dir(self._dir(case_id, "tracts"))
        env = {**os.environ, "PYTHONPATH": str(self.repo / "tractlab" / "src")}
        case = self.tract_case(case_id)
        if case is None:
            log = root / "ingest.log"
            cmd = [sys.executable, "-m", "tractlab.ingest", str(self.archive.study_dicom_dir(uid)),
                   "--cases-root", str(root), "--label", "archive-study", "--json"]
            code = self._run("tracts", cmd, self.repo / "tractlab", log, env)
            text = log.read_text(errors="replace")
            report = {}
            for line in reversed(text.splitlines()):
                if line.startswith("{"):
                    try:
                        report = json.loads(line)
                        break
                    except ValueError:
                        continue
            if code != 0 or not report.get("ok"):
                refusals = report.get("refusals") or [_last_line(text)]
                raise RuntimeError("DICOM ingest refused: " + "; ".join(str(r) for r in refusals)[:180])
            case = self.tract_case(case_id)
            if case is None:
                raise RuntimeError("ingest finished but wrote no case")
        script = self.repo / "tractlab" / "scripts" / "pipeline" / "run_case.sh"
        code = self._run("tracts", ["bash", str(script), str(case)], self.repo / "tractlab", root / "pipeline.log", env)
        if code != 0 or self._read_status(case).get("state") != "done":
            st = self._read_status(case)
            raise RuntimeError(f"tract pipeline stopped at stage {st.get('stage') or '?'} (exit {code})")

    def stop(self, case_id: str | None = None) -> None:
        for kind, proc in list(self._procs.items()):
            if case_id is None or self._running.get(kind) == case_id:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
