"""E3 — acquisition preflight. Scorecard only; never disables tracking.

Every criterion is untestable until `acquisition_evidence` declares a file
or a signed absence. Automation never approves.

S-01/N-03: every entry point takes the caller's already-loaded manifest
object canonically (``run_preflight(case_root, manifest=...)``) instead of
re-opening ``manifest.json`` from disk — a long-running server must score
the SAME manifest object it started with, never a file that may have
changed underneath it. Receipts also carry schema-v2 evidence fingerprints
(declaration semantics + actual referenced bytes, including bvec/bval
companions and registration b0/mask dependencies) so a verdict that stayed
the same while the underlying bytes changed is still reported as drift —
"same verdict, changed bytes" is stale, not clean.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .evidence_identity import sha256_file as _sha256_file

FINGERPRINT_SCHEMA = 2

CRITERION_IDS = (
    "gradients",
    "shells",
    "isotropy",
    "pe_correction",
    "orientation",
    "registration_t1_b0",
    "eddy_qc",
)

VERDICTS = ("pass", "degraded", "fail", "untestable")
_CRITERION_KEYS = frozenset({"id", "verdict", "evidence"})
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}\Z")
_MISSING = object()


@dataclass(frozen=True)
class Criterion:
    id: str
    verdict: str
    evidence: str


@dataclass(frozen=True)
class PreflightReport:
    criteria: tuple[Criterion, ...]
    fingerprints: "dict[str, str]"

    def to_json(self) -> dict:
        return {
            "schema": FINGERPRINT_SCHEMA,
            "criteria": [asdict(c) for c in self.criteria],
            "fingerprints": dict(self.fingerprints or {}),
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(case_root: Path) -> dict:
    return json.loads((case_root / "manifest.json").read_text())


def _evidence(man: dict, cid: str) -> dict | None:
    block = man.get("acquisition_evidence")
    if not isinstance(block, dict) or cid not in block:
        return None
    raw = block[cid]
    return raw if isinstance(raw, dict) else None


def _untestable(cid: str, why: str = "no declared evidence") -> Criterion:
    return Criterion(id=cid, verdict="untestable", evidence=why)


def _resolve(case_root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"evidence path must be relative: {rel!r}")
    relative = Path(rel)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"evidence path must be relative: {rel!r}")
    return _confined_path(case_root, Path(case_root) / relative)


def _confined_path(case_root: Path, candidate: Path, *, kind: str = "evidence") -> Path:
    """Resolve a path and reject symlink traversal outside the canonical root."""
    root = Path(case_root).resolve()
    resolved = Path(candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{kind} path escapes canonical case root") from exc
    return resolved


def _check_sha(path: Path, expected: str | None) -> str | None:
    """Return an error evidence string, or None if ok/not declared."""
    if not expected:
        return None
    if not path.is_file():
        return f"missing file {path.name}"
    got = _sha256_file(path)
    if got != expected:
        return f"sha mismatch {path.name}: declared {expected[:12]}… live {got[:12]}…"
    return None


def _declared_file(case_root: Path, ev: dict) -> tuple[Path | None, Criterion | None]:
    if "absent_reason" in ev:
        return None, None
    rel = ev.get("path")
    if not isinstance(rel, str) or not rel:
        return None, None
    try:
        path = _resolve(case_root, rel)
    except ValueError as e:
        return None, Criterion(id="", verdict="fail", evidence=str(e))
    err = _check_sha(path, ev.get("sha256") if isinstance(ev.get("sha256"), str) else None)
    if err:
        return path, Criterion(id="", verdict="fail", evidence=err)
    if not path.is_file():
        return path, Criterion(id="", verdict="fail", evidence=f"missing file {rel}")
    return path, None


def _gradients(case_root: Path, ev: dict) -> Criterion:
    path, fail = _declared_file(case_root, ev)
    if fail:
        return Criterion(id="gradients", verdict="fail", evidence=fail.evidence)
    if path is None:
        return _untestable("gradients")
    bvec = path
    bval = path.with_name(path.stem + ".bval") if path.suffix == ".bvec" else path
    try:
        bval = _confined_path(case_root, bval, kind="gradient companion")
    except ValueError as e:
        return Criterion(id="gradients", verdict="fail", evidence=str(e))
    if not bvec.is_file() or not bval.is_file():
        return Criterion(
            id="gradients",
            verdict="fail",
            evidence=f"need bvec+bval beside {path.name}",
        )
    try:
        vec = np.loadtxt(bvec)
        val = np.loadtxt(bval)
    except (OSError, ValueError) as e:
        return Criterion(
            id="gradients",
            verdict="fail",
            evidence=f"cannot parse bvec/bval ({e})",
        )
    if vec.ndim == 0:
        return Criterion(
            id="gradients",
            verdict="fail",
            evidence=f"bvec shape {vec.shape} is not 3×N",
        )
    if vec.ndim == 1:
        if vec.size != 3:
            return Criterion(
                id="gradients",
                verdict="fail",
                evidence=f"bvec shape {vec.shape} is not a single 3-vector or 3×N",
            )
        n_vec = 1
    elif vec.ndim == 2 and vec.shape[0] == 3:
        n_vec = int(vec.shape[1])
    elif vec.ndim == 2 and vec.shape[1] == 3:
        n_vec = int(vec.shape[0])
    else:
        return Criterion(
            id="gradients",
            verdict="fail",
            evidence=f"bvec shape {vec.shape} is not 3×N",
        )
    n_val = int(np.atleast_1d(val).size)
    if n_vec != n_val:
        return Criterion(
            id="gradients",
            verdict="fail",
            evidence=f"count mismatch bvec={n_vec} bval={n_val} ({bvec.name})",
        )
    return Criterion(
        id="gradients",
        verdict="pass",
        evidence=f"{bvec.name} n={n_vec} agrees with {bval.name}",
    )


def _shells(case_root: Path, ev: dict) -> Criterion:
    path, fail = _declared_file(case_root, ev)
    if fail:
        return Criterion(id="shells", verdict="fail", evidence=fail.evidence)
    if path is None:
        return _untestable("shells")
    bval = np.atleast_1d(np.loadtxt(path)).astype(float)
    raw = sorted({int(round(b)) for b in bval})
    # Cluster to 100 s/mm² so 995/1000/1005 is one shell, not three.
    shells = sorted({int(round(b / 100.0) * 100) for b in bval})
    nonzero = [s for s in shells if s > 50]
    evidence = f"{path.name} shells={raw} clustered={shells}"
    if not nonzero:
        return Criterion(id="shells", verdict="fail", evidence=evidence + " no DWI shell")
    if len(nonzero) < 2:
        return Criterion(
            id="shells",
            verdict="degraded",
            evidence=evidence + " single DWI shell",
        )
    return Criterion(id="shells", verdict="pass", evidence=evidence)


def _isotropy(case_root: Path, ev: dict) -> Criterion:
    import nibabel as nib

    path, fail = _declared_file(case_root, ev)
    if fail:
        return Criterion(id="isotropy", verdict="fail", evidence=fail.evidence)
    if path is None:
        return _untestable("isotropy")
    zooms = np.asarray(nib.load(str(path)).header.get_zooms()[:3], dtype=float)
    ratio = float(zooms.max() / max(zooms.min(), 1e-6))
    evs = f"{path.name} zooms={tuple(round(z, 3) for z in zooms)} ratio={ratio:.2f}"
    if ratio > 1.15:
        return Criterion(id="isotropy", verdict="degraded", evidence=evs)
    return Criterion(id="isotropy", verdict="pass", evidence=evs)


def _pe_correction(case_root: Path, ev: dict) -> Criterion:
    if ev.get("absent_reason"):
        signed = ev.get("signed_by")
        if isinstance(signed, str) and signed.strip():
            return Criterion(
                id="pe_correction",
                verdict="degraded",
                evidence=f"absent:{ev['absent_reason']} signed_by={signed}",
            )
        return _untestable(
            "pe_correction",
            f"absent:{ev['absent_reason']} unsigned",
        )
    path, fail = _declared_file(case_root, ev)
    if fail:
        return Criterion(id="pe_correction", verdict="fail", evidence=fail.evidence)
    if path is None:
        return _untestable("pe_correction")
    return Criterion(
        id="pe_correction",
        verdict="pass",
        evidence=f"{path.name} present",
    )


def _orientation(case_root: Path, ev: dict) -> Criterion:
    import nibabel as nib

    path, fail = _declared_file(case_root, ev)
    if fail:
        return Criterion(id="orientation", verdict="fail", evidence=fail.evidence)
    if path is None:
        return _untestable("orientation")
    expected = ev.get("expected_affine")
    if not isinstance(expected, (list, tuple)) or len(expected) != 16:
        return _untestable(
            "orientation",
            "declared file but no expected_affine (16 floats)",
        )
    live = np.asarray(nib.load(str(path)).affine, dtype=float).ravel()
    exp = np.asarray(expected, dtype=float).ravel()
    if not np.allclose(live, exp, atol=1e-3):
        return Criterion(
            id="orientation",
            verdict="fail",
            evidence=f"{path.name} affine flipped/mismatch vs pinned expected",
        )
    return Criterion(
        id="orientation",
        verdict="pass",
        evidence=f"{path.name} affine matches pinned expected",
    )


def _registration(
    case_root: Path,
    ev: dict,
    manifest: dict | None = None,
) -> Criterion:
    """S-01/N-03: ``manifest`` is the caller's canonical object — never a
    fresh re-read of manifest.json, which could differ from what a
    long-running server actually started with."""
    from .check_registration import auto_sanity
    from .grid import unknown_units_assumption_from_manifest

    path, fail = _declared_file(case_root, ev)
    if fail:
        return Criterion(id="registration_t1_b0", verdict="fail", evidence=fail.evidence)
    if path is None:
        return _untestable("registration_t1_b0")
    b0 = ev.get("b0")
    mask = ev.get("mask")
    if not isinstance(b0, str) or not isinstance(mask, str):
        return _untestable(
            "registration_t1_b0",
            "need b0 and mask paths on the evidence block",
        )
    try:
        b0_p = _resolve(case_root, b0)
        mask_p = _resolve(case_root, mask)
    except ValueError as e:
        return Criterion(id="registration_t1_b0", verdict="fail", evidence=str(e))
    man = manifest if manifest is not None else _load_manifest(case_root)
    expected = ((man.get("grid") or {}).get("grid_id") if isinstance(man.get("grid"), dict) else None)
    if not expected:
        return _untestable(
            "registration_t1_b0",
            "manifest has no grid.grid_id",
        )
    s = auto_sanity(
        b0_path=str(b0_p),
        t1_path=str(path),
        mask_path=str(mask_p),
        expected_grid_id=str(expected),
        assume_unknown_spatial_units_mm=unknown_units_assumption_from_manifest(manifest),
    )
    evs = (
        f"auto_sanity ok={s.ok} overlap={s.mask_overlap_frac:.3f} "
        f"t1={path.name}"
    )
    if s.notes:
        evs += " notes=" + ";".join(s.notes)
    return Criterion(
        id="registration_t1_b0",
        verdict="pass" if s.ok else "fail",
        evidence=evs,
    )


def _eddy_qc(case_root: Path, ev: dict) -> Criterion:
    if ev.get("absent_reason"):
        who = ev.get("signed_by")
        extra = f" signed_by={who}" if isinstance(who, str) and who.strip() else " unsigned"
        return _untestable("eddy_qc", f"absent:{ev['absent_reason']}{extra}")
    path, fail = _declared_file(case_root, ev)
    if fail:
        return Criterion(id="eddy_qc", verdict="fail", evidence=fail.evidence)
    if path is None:
        return _untestable("eddy_qc")
    try:
        json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return Criterion(
            id="eddy_qc",
            verdict="fail",
            evidence=f"{path.name} not ingestible JSON: {e}",
        )
    return Criterion(
        id="eddy_qc",
        verdict="pass",
        evidence=f"{path.name} ingested (not recomputed)",
    )


_HANDLERS = {
    "gradients": _gradients,
    "shells": _shells,
    "isotropy": _isotropy,
    "pe_correction": _pe_correction,
    "orientation": _orientation,
    "eddy_qc": _eddy_qc,
}


def _companion_paths(
    cid: str, path: Path, *, case_root: Path | None = None
) -> list[Path]:
    """Files whose bytes co-determine a criterion but are not the declared path.

    Fingerprinting uses the canonical root so a companion symlink that points
    outside the case can never be opened or hashed as evidence.
    """
    raw: list[Path] = []
    if cid == "gradients" and path.suffix == ".bvec":
        raw.append(path.with_name(path.stem + ".bval"))
    if case_root is None:
        return raw
    safe: list[Path] = []
    for candidate in raw:
        try:
            safe.append(_confined_path(case_root, candidate, kind=f"{cid} companion"))
        except ValueError:
            continue
    return safe


def _registration_companion_paths(case_root: Path, ev: dict) -> list[Path]:
    out: list[Path] = []
    for key in ("b0", "mask"):
        v = ev.get(key)
        if isinstance(v, str) and v:
            try:
                out.append(_resolve(case_root, v))
            except ValueError:
                pass
    return out


def _evidence_fingerprint(case_root: Path, cid: str, ev: dict | None) -> str:
    """Schema-v2 fingerprint: declaration semantics + actual referenced bytes.

    Hashes the evidence *declaration* (so a changed threshold/path/absence
    reason moves the fingerprint even with no file present) plus the live
    bytes of every file the criterion actually reads — the declared path,
    gradient bvec/bval companions, and registration's b0/mask dependencies.
    A verdict can stay "pass" while these bytes silently change underneath
    it; the fingerprint is what lets ``compute_drift`` catch that.
    """
    h = hashlib.sha256()
    h.update(json.dumps({"id": cid, "declaration": ev}, sort_keys=True, default=str).encode())
    if isinstance(ev, dict) and "absent_reason" not in ev:
        rel = ev.get("path")
        if isinstance(rel, str) and rel:
            try:
                path = _resolve(case_root, rel)
            except ValueError:
                path = None
            if path is not None:
                candidates = [path, *_companion_paths(cid, path, case_root=case_root)]
                if cid == "registration_t1_b0":
                    candidates += _registration_companion_paths(case_root, ev)
                for p in sorted(set(candidates), key=str):
                    try:
                        rel_name = str(p.relative_to(case_root))
                    except ValueError:
                        rel_name = str(p)
                    h.update(rel_name.encode())
                    if p.is_file():
                        h.update(_sha256_file(p).encode())
                    else:
                        h.update(b"missing")
    return h.hexdigest()


def run_preflight(
    case_root: str,
    manifest: dict | None = None,
) -> PreflightReport:
    """Score every criterion against ``manifest`` (S-01/N-03).

    ``manifest`` defaults to a fresh read of ``case_root/manifest.json`` for
    CLI/offline callers. A long-running server MUST pass its own already-
    loaded manifest object instead of letting this re-open the file — that
    is the one canonical manifest every preflight helper, including
    registration, is scored against.
    """
    root = Path(case_root).resolve()
    man = manifest if manifest is not None else _load_manifest(root)
    out: list[Criterion] = []
    fingerprints: dict[str, str] = {}
    for cid in CRITERION_IDS:
        ev = _evidence(man, cid)
        fingerprints[cid] = _evidence_fingerprint(root, cid, ev)
        if ev is None:
            out.append(_untestable(cid))
            continue
        if cid == "registration_t1_b0":
            out.append(_registration(root, ev, manifest=man))
        else:
            out.append(_HANDLERS[cid](root, ev))
    return PreflightReport(criteria=tuple(out), fingerprints=fingerprints)


def _git_commit() -> str:
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = proc.stdout.strip()
        return sha or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def produced_with() -> dict:
    import platform

    import nibabel as nib

    return {
        "tractlab_commit": _git_commit(),
        "python": platform.python_version(),
        "nibabel": getattr(nib, "__version__", "unknown"),
    }


def _active_derivation_of(manifest: dict) -> str | None:
    active = manifest.get("active_derivation")
    return active if isinstance(active, str) and active else None


def _active_derivation(case_root: str) -> str | None:
    return _active_derivation_of(_load_manifest(Path(case_root)))


def build_receipt(case_root: str, manifest: dict | None = None) -> dict:
    root = Path(case_root).resolve()
    man = manifest if manifest is not None else _load_manifest(root)
    report = run_preflight(case_root, man)
    report_json = report.to_json()
    return {
        "schema": FINGERPRINT_SCHEMA,
        "produced_with": produced_with(),
        "criteria": report_json["criteria"],
        "fingerprints": report_json["fingerprints"],
        "approved_by": None,
        "derivation": _active_derivation_of(man),
    }


def _validate_receipt(receipt: object, *, allow_legacy: bool = False) -> dict:
    """Validate the persisted schema-v2 receipt contract.

    ``compute_drift`` retains one narrow schema-v1 diagnostic path so an
    already-loaded legacy object can be reported as requiring migration. Disk
    loading and writing remain schema-v2-only, so an old or future receipt can
    never reach approval.
    """
    if not isinstance(receipt, dict):
        raise ValueError("invalid preflight receipt: receipt must be an object")

    schema = receipt.get("schema", _MISSING)
    if type(schema) is not int:
        raise ValueError("invalid preflight receipt: schema must be an integer")
    legacy = allow_legacy and schema == 1
    if schema != FINGERPRINT_SCHEMA and not legacy:
        raise ValueError(
            f"invalid preflight receipt: unsupported schema {schema}; "
            f"expected {FINGERPRINT_SCHEMA}"
        )

    criteria = receipt.get("criteria", _MISSING)
    if not isinstance(criteria, list) or len(criteria) != len(CRITERION_IDS):
        raise ValueError(
            "invalid preflight receipt: criterion coverage must contain "
            f"exactly {len(CRITERION_IDS)} entries"
        )
    ids: list[str] = []
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict) or set(criterion) != _CRITERION_KEYS:
            raise ValueError(
                f"invalid preflight receipt: criterion {index} has malformed fields"
            )
        cid = criterion.get("id")
        if not isinstance(cid, str) or cid not in CRITERION_IDS:
            raise ValueError(f"invalid preflight receipt: unknown criterion id {cid!r}")
        if cid in ids:
            raise ValueError(f"invalid preflight receipt: duplicate criterion id {cid!r}")
        ids.append(cid)
        verdict = criterion.get("verdict")
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise ValueError(f"invalid preflight receipt: invalid verdict for {cid}")
        if not isinstance(criterion.get("evidence"), str):
            raise ValueError(f"invalid preflight receipt: evidence for {cid} must be text")
    if set(ids) != set(CRITERION_IDS):
        raise ValueError("invalid preflight receipt: criterion coverage is incomplete")

    fingerprints = receipt.get("fingerprints", _MISSING)
    if legacy and fingerprints is _MISSING:
        return receipt
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(CRITERION_IDS):
        raise ValueError(
            "invalid preflight receipt: fingerprint coverage must match criteria exactly"
        )
    for cid, fingerprint in fingerprints.items():
        if not isinstance(fingerprint, str) or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
            raise ValueError(f"invalid preflight receipt: malformed fingerprint for {cid}")
    return receipt


def receipt_path(case_root: str) -> Path:
    return Path(case_root).resolve() / "preflight.json"


def load_receipt(case_root: str) -> dict | None:
    root = Path(case_root).resolve()
    path = _confined_path(root, receipt_path(case_root), kind="receipt")
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid preflight receipt JSON: {e}") from e
    return _validate_receipt(parsed)


def write_receipt(case_root: str, receipt: dict) -> Path:
    _validate_receipt(receipt)
    root = Path(case_root).resolve()
    path = _confined_path(root, receipt_path(case_root), kind="receipt")
    tmp = _confined_path(root, path.with_name(path.name + ".tmp"), kind="receipt temp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    tmp.replace(path)
    return path


def compute_drift(
    stored: dict | None,
    verified: dict,
    *,
    active_derivation: str | None = None,
) -> list[str]:
    """Criterion ids (plus ``"derivation"``/``"schema"``) that drifted.

    A verdict change is drift. So, separately, is an unchanged verdict whose
    schema-v2 fingerprint moved — "same verdict, changed bytes" (S-01) is
    stale, not clean. A stored receipt built before fingerprints existed
    (schema < FINGERPRINT_SCHEMA) always reports ``"schema"`` drift: it
    requires an explicit new ``prepare --apply`` before it can be approved,
    never a silent reinterpretation under the new contract.
    """
    verified_receipt = _validate_receipt(verified)
    if stored is None:
        return []
    stored_receipt = _validate_receipt(stored, allow_legacy=True)
    prev_verdict = {
        c["id"]: c.get("verdict") for c in stored_receipt["criteria"]
    }
    stored_schema = stored_receipt["schema"]
    prev_fp = stored_receipt.get("fingerprints") or {}
    verified_fp = verified_receipt["fingerprints"]
    ids: list[str] = []
    for c in verified_receipt["criteria"]:
        cid = c["id"]
        if prev_verdict.get(cid) != c.get("verdict"):
            ids.append(cid)
            continue
        if (
            stored_schema >= FINGERPRINT_SCHEMA
            and cid in prev_fp
            and cid in verified_fp
            and prev_fp[cid] != verified_fp[cid]
        ):
            ids.append(cid)
    if stored_receipt.get("derivation") != active_derivation:
        ids.append("derivation")
    if stored_schema < FINGERPRINT_SCHEMA:
        ids.append("schema")
    return sorted(set(ids))


def approve_receipt(case_root: str, *, who: str) -> dict:
    """Owner sign: re-verify, refuse on drift, write approved_by from who only."""
    if not who or not str(who).strip():
        raise ValueError("approved_by must come from --who")
    stored = load_receipt(case_root)
    if stored is None:
        raise ValueError("no preflight.json — run prepare --apply first")
    verified = build_receipt(case_root)
    drift = compute_drift(
        stored, verified, active_derivation=_active_derivation(case_root),
    )
    if drift:
        raise ValueError(f"drift before sign: {', '.join(drift)}")
    from datetime import date

    stored["approved_by"] = str(who).strip()
    stored["approved_date"] = date.today().isoformat()
    write_receipt(case_root, stored)
    return stored


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m tractlab.preflight")
    sub = ap.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="score criteria; write only with --apply")
    prep.add_argument("case_root")
    prep.add_argument("--apply", action="store_true")
    prep.add_argument(
        "--force-unsigned",
        action="store_true",
        help="owner-gated: replace a signed receipt with a new unsigned one",
    )
    prep.add_argument(
        "--approve",
        default=None,
        help="rejected — use the approve subcommand",
    )
    appr = sub.add_parser("approve", help="owner sign after a clean re-verify")
    appr.add_argument("case_root")
    appr.add_argument("--who", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "approve":
        try:
            rec = approve_receipt(args.case_root, who=args.who)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(json.dumps(rec, indent=2))
        return 0
    if getattr(args, "approve", None):
        print(
            "automation cannot approve; use: python -m tractlab.preflight "
            "approve <case> --who NAME",
            file=sys.stderr,
        )
        return 2
    receipt = build_receipt(args.case_root)
    print(json.dumps(receipt, indent=2))
    if args.apply:
        existing = load_receipt(args.case_root)
        who = existing.get("approved_by") if existing else None
        if who and not args.force_unsigned:
            print(
                "Error: signed preflight.json present; "
                "refusing to overwrite (pass --force-unsigned to void it)",
                file=sys.stderr,
            )
            return 1
        write_receipt(args.case_root, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
