"""brainchop DKatlas fast-anatomy lane — PILOT, WHITE-BOX (owner reads every line).

Alternative to FreeSurfer recon-all (3.47 h on the demo subject) for the
FreeSurfer-numbered aparc+aseg the house 5TT recipe needs. `uvx brainchop -m
DKatlas` segments a T1 into 104 MeshNet classes in ~8s on Metal; this module
remaps those classes onto real FreeSurfer label ids (so 5ttgen freesurfer and
every downstream ROI-name lookup keep working unchanged) and shells the CLI
the same way track.py shells tckgen.

House rule (docs/ARCH-annex-D-verification-roadmap-slice.md, CONTEXT.md):
model-derived anatomy is an IMPORTED, signed artefact under fail-closed QC —
never lesion segmentation, never a margin. Nothing here enters the manifest;
this is a pilot lane, not a switch of the default anatomy path.

Security + honesty rules (mirrors track.py):
  * argv list only — never a shell string.
  * A hard timeout kills the process; typed outcomes — a failure is never
    silently rendered as an empty/zero result.
      OK            rc==0, fresh output, remap clean, FULL provenance -> aparc_path set
      ENGINE_ERROR  rc!=0 / missing / stale / malformed / unmapped /
                    incomplete-provenance                             -> aparc_path None
      TIMEOUT       killed by the deadline                             -> aparc_path None
      OSERROR       could not even launch brainchop (e.g. uvx missing) -> aparc_path None
  * remap_to_freesurfer fails CLOSED: any brainchop class id absent from the
    vendored LUT raises rather than silently zeroing or passing it through;
    same for a non-integer/NaN/negative/fractional label volume — refused,
    never silently truncated.
  * A brainchop_run.json sidecar is written on EVERY outcome (success or
    failure) — a failed run leaves a typed record (outcome, reason, argv,
    timestamps), never silence.
  * Stale-output guard: any pre-existing brainchop_raw/aparc_brainchop/sidecar
    in out_dir is removed before invoking brainchop, and the fresh raw output
    is additionally required to postdate the run start — a leftover file from
    a previous run can never be mistaken for this run's output.
  * Outcome.OK requires COMPLETE provenance (LUT path+sha256, model weights
    path+sha256, brainchop version, tinygrad device, model folder) — a run
    that produced a scientifically plausible remap but can't be fully
    attested is still reported as ENGINE_ERROR, not OK.

LUT provenance: src/tractlab/lut/brainchop_dkatlas_lut.json — see its
"_provenance" header for source URL, sha256, and the cross-check against the
morning Dice run's compact ids. (Not under src/tractlab/data/ — that name is
caught by the repo's blanket PHI-defense .gitignore rule `data/`.)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import nibabel as nib
import numpy as np

from neuro_core.hashing import sha256_file

LUT_DIR = os.path.join(os.path.dirname(__file__), "lut")
LUT_PATH = os.path.join(LUT_DIR, "brainchop_dkatlas_lut.json")

# Where `uvx brainchop` itself caches its model registry + weights once it has
# run at least once on this machine. The model folder for a given model name
# (e.g. "DKatlas") is read from here at run time, never hard-coded — the CLI
# is the authority on which folder a model name currently resolves to.
MODELS_JSON_PATH = os.path.expanduser("~/.cache/brainchop/models.json")
MODELS_CACHE_DIR = os.path.expanduser("~/.cache/brainchop/models")
MODEL_NAME = "DKatlas"

UVX = shutil.which("uvx") or os.path.expanduser("~/.local/bin/uvx")


class Outcome(str, Enum):
    OK = "ok"
    ENGINE_ERROR = "engine_error"
    TIMEOUT = "timeout"
    OSERROR = "oserror"


class UnknownLabelError(ValueError):
    """Raised by remap_to_freesurfer when a voxel carries an id the LUT doesn't cover."""


class InvalidLabelVolumeError(ValueError):
    """Raised by remap_to_freesurfer for a non-integer/NaN/negative/fractional volume."""


@dataclass
class BrainchopResult:
    outcome: Outcome
    aparc_path: str | None
    wall_s: float
    argv: list[str]
    warning: str | None = None
    sidecar: dict = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


_sha256_file = sha256_file


def load_lut(path: str = LUT_PATH) -> dict[int, int]:
    """brainchop class id -> FreeSurfer label id, from the vendored, provenanced LUT."""
    with open(path) as f:
        doc = json.load(f)
    entries = doc["entries"]
    lut = {int(e["brainchop_id"]): int(e["freesurfer_id"]) for e in entries}
    if len(lut) != len(entries):
        raise ValueError(f"duplicate brainchop_id in LUT: {path}")
    return lut


def _resolve_model_folder(model_name: str = MODEL_NAME, models_json_path: str = MODELS_JSON_PATH) -> str:
    """Model-name -> on-disk folder, read from brainchop's own cached models.json.

    Raises (propagated to the caller, which converts to a typed outcome):
      OSError / FileNotFoundError  if models.json hasn't been fetched yet
      json.JSONDecodeError         if it's malformed
      KeyError                     if model_name isn't a key in it
    """
    with open(models_json_path) as f:
        doc = json.load(f)
    return doc[model_name]["folder"]


def _model_weights_path(model_folder: str, models_cache_dir: str = MODELS_CACHE_DIR) -> str:
    return os.path.join(models_cache_dir, model_folder, "model.pth")


def remap_to_freesurfer(label_volume: np.ndarray, lut: dict[int, int]) -> np.ndarray:
    """Remap brainchop DKatlas class ids onto FreeSurfer label ids.

    Fails CLOSED on two independent axes:
      * malformed input — non-integer dtype, NaN, negative, or fractional
        values raise InvalidLabelVolumeError rather than being silently
        truncated/cast into a plausible-looking id;
      * unmapped content — any id present in ``label_volume`` that is not a
        key of ``lut`` raises UnknownLabelError before any output is produced.
    """
    vol = np.asarray(label_volume)

    if np.issubdtype(vol.dtype, np.floating):
        if np.isnan(vol).any():
            raise InvalidLabelVolumeError("label volume contains NaN values")
        if np.any(vol < 0):
            raise InvalidLabelVolumeError("label volume contains negative values")
        if np.any(vol != np.round(vol)):
            raise InvalidLabelVolumeError("label volume contains fractional (non-integer) values")
        raise InvalidLabelVolumeError(
            f"label volume has non-integer dtype {vol.dtype} — refused even though "
            "every value happens to be whole; a label volume must be an integer array"
        )
    if not np.issubdtype(vol.dtype, np.integer):
        raise InvalidLabelVolumeError(f"label volume has unsupported non-integer dtype {vol.dtype}")
    if np.any(vol < 0):
        raise InvalidLabelVolumeError("label volume contains negative values")

    present = np.unique(vol)
    unknown = sorted(int(v) for v in present if int(v) not in lut)
    if unknown:
        raise UnknownLabelError(f"unmapped brainchop label ids: {unknown}")
    max_id = int(present.max()) if present.size else 0
    table_size = max(max_id, max(lut.keys(), default=0)) + 1
    table = np.zeros(table_size, dtype=np.int32)
    for bc_id, fs_id in lut.items():
        table[bc_id] = fs_id
    return table[vol.astype(np.int64)]


def _brainchop_version(timeout_s: float = 60.0) -> str | None:
    """pip-metadata version of the brainchop package brainchop-cli's uvx run resolves."""
    try:
        out = subprocess.run(
            [UVX, "--from", "brainchop", "python", "-c",
             "import importlib.metadata as m; print(m.version('brainchop'))"],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    v = out.stdout.strip()
    return v or None


def _tinygrad_device(timeout_s: float = 60.0) -> str | None:
    """tinygrad's Device.DEFAULT in the same uvx-resolved environment (e.g. METAL)."""
    try:
        out = subprocess.run(
            [UVX, "--from", "brainchop", "python", "-c",
             "from tinygrad import Device; print(Device.DEFAULT)"],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    d = out.stdout.strip()
    return d or None


def run_dkatlas(
    t1_path: str,
    out_dir: str,
    timeout_s: float = 300.0,
    lut_path: str = LUT_PATH,
    models_json_path: str = MODELS_JSON_PATH,
    models_cache_dir: str = MODELS_CACHE_DIR,
    model_name: str = MODEL_NAME,
) -> BrainchopResult:
    """Run `uvx brainchop -m DKatlas` on t1_path, remap the output, write outputs.

    Caller owns out_dir. On Outcome.OK, writes:
      out_dir/brainchop_raw.nii.gz   — raw brainchop class-id volume
      out_dir/aparc_brainchop.nii.gz — FreeSurfer-numbered remap (5ttgen input)
      out_dir/brainchop_run.json     — full provenance sidecar
    On any other outcome, aparc_brainchop.nii.gz is absent and
    brainchop_run.json records a typed failure (outcome, reason, argv,
    timestamps) — a failed run is never silent.
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_out = os.path.join(out_dir, "brainchop_raw.nii.gz")
    aparc_path = os.path.join(out_dir, "aparc_brainchop.nii.gz")
    sidecar_path = os.path.join(out_dir, "brainchop_run.json")

    # Stale-output guard: a file left over from a previous run in this same
    # out_dir must never be mistaken for this run's output.
    for stale in (raw_out, aparc_path, sidecar_path):
        try:
            os.remove(stale)
        except FileNotFoundError:
            pass

    argv = [
        UVX, "brainchop", t1_path,
        "-m", model_name,
        "--no-optimize",
        "--inverse-conform",
        "-o", raw_out,
    ]

    started_at = _now_iso()
    t0 = time.time()

    def _fail(outcome: Outcome, reason: str, extra: dict | None = None) -> BrainchopResult:
        wall = time.time() - t0
        sidecar = {
            "outcome": outcome.value,
            "reason": reason,
            "argv": argv,
            "started_at": started_at,
            "finished_at": _now_iso(),
            "wall_s": round(wall, 3),
        }
        if extra:
            sidecar.update(extra)
        _write_json(sidecar_path, sidecar)
        return BrainchopResult(outcome, None, wall, argv, warning=reason, sidecar=sidecar)

    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return _fail(Outcome.TIMEOUT, f"brainchop exceeded {timeout_s}s and was killed")
    except OSError as exc:
        return _fail(Outcome.OSERROR, f"failed to launch brainchop: {exc}")

    if proc.returncode != 0:
        reason = (proc.stderr or "")[:400] or f"brainchop exited {proc.returncode}"
        return _fail(Outcome.ENGINE_ERROR, reason, {"returncode": proc.returncode})

    if not os.path.exists(raw_out):
        return _fail(
            Outcome.ENGINE_ERROR,
            "brainchop returned rc==0 but the expected output is missing",
            {"returncode": proc.returncode},
        )
    if os.path.getmtime(raw_out) < t0 - 1.0:  # 1s slack for filesystem mtime granularity
        return _fail(
            Outcome.ENGINE_ERROR,
            "brainchop returned rc==0 but the output predates this run's start "
            "(stale pre-existing file) — refused",
            {"returncode": proc.returncode},
        )

    try:
        lut = load_lut(lut_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _fail(Outcome.ENGINE_ERROR, f"failed to load LUT: {exc}", {"returncode": proc.returncode})

    try:
        img = nib.load(raw_out)
        data = np.asanyarray(img.dataobj)
    except Exception as exc:  # noqa: BLE001 — malformed NIfTI must be typed, not a crash
        return _fail(
            Outcome.ENGINE_ERROR,
            f"failed to load brainchop output as NIfTI: {exc}",
            {"returncode": proc.returncode},
        )

    try:
        remapped = remap_to_freesurfer(data, lut)
    except (UnknownLabelError, InvalidLabelVolumeError) as exc:
        return _fail(Outcome.ENGINE_ERROR, f"remap failed: {exc}", {"returncode": proc.returncode})

    try:
        out_img = nib.Nifti1Image(remapped.astype(np.int32), img.affine, img.header)
        out_img.header.set_data_dtype(np.int32)
        nib.save(out_img, aparc_path)
    except OSError as exc:
        return _fail(
            Outcome.ENGINE_ERROR, f"failed to save remapped output: {exc}", {"returncode": proc.returncode}
        )

    # Provenance — every field here is REQUIRED for Outcome.OK. A run that
    # produced a plausible-looking remap but cannot be fully attested is
    # still reported as a failure, not silently accepted as OK.
    try:
        model_folder = _resolve_model_folder(model_name, models_json_path)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return _fail(
            Outcome.ENGINE_ERROR,
            f"failed to resolve model folder from models.json: {exc}",
            {"returncode": proc.returncode},
        )

    weights_path = _model_weights_path(model_folder, models_cache_dir)
    try:
        lut_sha256 = _sha256_file(lut_path)
        input_sha256 = _sha256_file(t1_path)
        raw_output_sha256 = _sha256_file(raw_out)
        output_sha256 = _sha256_file(aparc_path)
        weights_sha256 = _sha256_file(weights_path)
    except OSError as exc:
        return _fail(
            Outcome.ENGINE_ERROR, f"failed to hash a provenance input: {exc}", {"returncode": proc.returncode}
        )

    brainchop_version = _brainchop_version()
    tinygrad_device = _tinygrad_device()

    required = {
        "model_folder": model_folder,
        "lut_sha256": lut_sha256,
        "model_weights_sha256": weights_sha256,
        "brainchop_version": brainchop_version,
        "tinygrad_device": tinygrad_device,
    }
    missing = sorted(k for k, v in required.items() if not v)
    if missing:
        try:
            os.remove(aparc_path)  # cannot attest this run: it is not OK, don't leave it looking like one
        except FileNotFoundError:
            pass
        return _fail(
            Outcome.ENGINE_ERROR,
            f"incomplete provenance, missing: {missing}",
            {"returncode": proc.returncode},
        )

    wall = time.time() - t0
    sidecar = {
        "outcome": Outcome.OK.value,
        "argv": argv,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "wall_s": round(wall, 3),
        "returncode": proc.returncode,
        "model": model_name,
        "model_folder": model_folder,
        "brainchop_version": brainchop_version,
        "tinygrad_device": tinygrad_device,
        "input_path": os.path.abspath(t1_path),
        "input_sha256": input_sha256,
        "raw_output_path": os.path.abspath(raw_out),
        "raw_output_sha256": raw_output_sha256,
        "output_path": os.path.abspath(aparc_path),
        "output_sha256": output_sha256,
        "lut_path": os.path.abspath(lut_path),
        "lut_sha256": lut_sha256,
        "model_weights_path": os.path.abspath(weights_path),
        "model_weights_sha256": weights_sha256,
    }
    _write_json(sidecar_path, sidecar)
    return BrainchopResult(Outcome.OK, aparc_path, wall, argv, sidecar=sidecar)
