"""Derivation provenance (ADR-0004): append-only preprocessing lineages.

A derivation records WHICH preprocessing produced a case's derived artifacts.
Signatures are derivation-scoped; artifacts never mix across derivations.
Migration is explicit: the caller states the lineage kind — never guessed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path

from neuro_core.hashing import sha256_file

from .casepath import resolve_case_path

KIND_UNCORRECTED = "uncorrected"
KIND_RPE_PAIR = "rpe_pair"
KINDS = (KIND_UNCORRECTED, KIND_RPE_PAIR)
DEFAULT_ID = "d1"
_QC_BLOCKS = ("t1_qc", "atlas_prior_qc", "parcellation_qc", "delta_qc")
_ARTIFACT_KINDS = frozenset(("normative", "qc", "fidelity", "profiles"))
# First-lineage historical locations that differ from ``<case_root>/<kind>``.
# Later lineages always use ``derivations/<id>/<kind>/`` (ADR-0007 §5).
_FIRST_LINEAGE_DIRS = {"profiles": Path("work") / "profiles"}
_DERIVATION_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DerivationError(ValueError):
    pass


class ArtifactMetadataError(DerivationError):
    """An active non-first artifact differs from its manifest declaration."""


def _require_derivation_id(value: object, *, what: str = "derivation id") -> str:
    if not isinstance(value, str) or _DERIVATION_ID_RE.fullmatch(value) is None:
        raise DerivationError(
            f"{what} must be a safe path segment matching "
            r"^[a-z][a-z0-9_-]{0,31}$"
        )
    return value


def _validate_derivation_ids(derivations: dict) -> None:
    for derivation_id in derivations:
        _require_derivation_id(derivation_id)


def migrate_manifest(manifest: dict, kind: str) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown derivation kind {kind!r}; expected one of {KINDS}")
    if "derivations" in manifest:
        raise ValueError("manifest already migrated; derivations are append-only")
    _require_derivation_id(DEFAULT_ID)
    out = copy.deepcopy(manifest)
    out["derivations"] = {DEFAULT_ID: {"kind": kind}}
    out["active_derivation"] = DEFAULT_ID
    for block in _QC_BLOCKS:
        if isinstance(out.get(block), dict):
            out[block]["derivation"] = DEFAULT_ID
    return out


def active_derivation(manifest: dict) -> str | None:
    return manifest.get("active_derivation")


def _manifest_root(manifest: dict) -> Path:
    return Path(str(manifest.get("case_root") or ".")).resolve()


def _contained_case_path(root: Path, candidate: Path, *, what: str) -> Path:
    """Resolve a generated path through the shared case-root containment gate."""
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as e:
        raise DerivationError(f"{what}: path escapes case_root") from e
    try:
        return Path(resolve_case_path(str(root), relative, what=what))
    except ValueError as e:
        raise DerivationError(str(e)) from e


def _is_first_derivation(manifest: dict) -> bool:
    derivs = manifest.get("derivations")
    if not isinstance(derivs, dict) or len(derivs) <= 1:
        return True
    return active_derivation(manifest) == next(iter(derivs))


def _kind_relative(kind: str, derivation_id: str | None, *, first: bool) -> Path:
    """The single layout rule for case-derived artifact directories."""
    if first:
        return _FIRST_LINEAGE_DIRS.get(kind, Path(kind))
    return Path("derivations") / derivation_id / kind


def artifact_dir(manifest: dict, kind: str) -> Path:
    """Return the active derivation's case-derived artifact directory.

    The first (or implicit legacy) lineage deliberately retains the historical
    case-root directories. Later lineages get a private subtree so producers
    cannot silently replace or be mistaken for the first lineage's outputs.
    """
    if kind not in _ARTIFACT_KINDS:
        raise DerivationError(
            f"unknown artifact kind {kind!r}; expected one of {sorted(_ARTIFACT_KINDS)}"
        )
    derivs = manifest.get("derivations")
    root = _manifest_root(manifest)
    if derivs is None:
        relative = _kind_relative(kind, None, first=True)
    else:
        if not isinstance(derivs, dict) or not derivs:
            raise DerivationError("derivations must be a non-empty object")
        _validate_derivation_ids(derivs)
        active = active_derivation(manifest)
        if active not in derivs:
            raise DerivationError(f"active_derivation {active!r} not in derivations")
        _require_derivation_id(active, what="active_derivation")
        first = next(iter(derivs))
        relative = _kind_relative(
            kind, active, first=len(derivs) <= 1 or active == first,
        )
    return _contained_case_path(root, root / relative, what=f"{kind} artifact directory")


def artifact_path(manifest: dict, kind: str, filename: str | Path) -> Path:
    """Return one safe active-lineage output path and reject sibling aliases."""
    name = Path(filename)
    if name.is_absolute() or not name.parts or any(part in ("", ".", "..") for part in name.parts):
        raise DerivationError(f"artifact filename must be a relative safe path: {filename!r}")
    root = _manifest_root(manifest)
    target = artifact_dir(manifest, kind) / name
    target_real = _contained_case_path(root, target, what=f"{kind} artifact")
    derivs = manifest.get("derivations")
    active = active_derivation(manifest)
    if isinstance(derivs, dict):
        root = _manifest_root(manifest)
        first = next(iter(derivs), None)
        for derivation_id, entry in derivs.items():
            if derivation_id == active or not isinstance(entry, dict):
                continue
            sibling_dir = root / _kind_relative(
                kind, derivation_id, first=derivation_id == first,
            )
            sibling_dir = _contained_case_path(
                root, sibling_dir, what=f"derivation {derivation_id!r} artifact directory"
            )
            try:
                target_real.relative_to(sibling_dir)
            except ValueError:
                pass
            else:
                raise DerivationError(
                    f"artifact output would overwrite derivation {derivation_id!r}: {target_real}"
                )
            inputs = _inputs_for_derivation(manifest, derivation_id, derivs)
            for sibling in _artifact_paths(manifest, derivation_id, inputs):
                if Path(sibling) == target_real:
                    raise DerivationError(
                        f"artifact output would overwrite derivation {derivation_id!r}: {target_real}"
                    )
    return target_real


def operating_point_path(manifest: dict) -> Path:
    """Return the fidelity operating-point sheet for the active lineage."""
    if _is_first_derivation(manifest):
        root = _manifest_root(manifest)
        return _contained_case_path(
            root,
            root / "docs" / "qc" / "OPERATING-POINT-fidelity.md",
            what="fidelity operating-point sheet",
        )
    return artifact_path(manifest, "fidelity", "OPERATING-POINT-fidelity.md")


def fidelity_sweep_path(manifest: dict) -> Path:
    """Return the fidelity sweep CSV for the active lineage."""
    return artifact_path(manifest, "fidelity", "sweep.csv")


def qc_block_for(
    manifest: dict, block: str, *, derivation_id: str | None = None,
) -> dict | None:
    """Return a QC block only when it belongs to the selected lineage.

    Legacy and single-lineage manifests retain their historical meaning: an
    unscoped block belongs to that one lineage. With multiple derivations,
    unscoped blocks are invalid and therefore invisible, while a sibling block
    is never usable by the active lineage. ``derivation_id`` is an internal
    read-only override used to build the per-lineage API summary; callers that
    need the active lineage use the two-argument form.
    """
    derivs = manifest.get("derivations")
    active = active_derivation(manifest)
    selected = active if derivation_id is None else derivation_id
    if isinstance(derivs, dict) and len(derivs) > 1:
        if selected not in derivs:
            return None
        first = next(iter(derivs))
        if selected == first:
            qc = manifest.get(block)
        else:
            entry = derivs.get(selected)
            qc_store = entry.get("qc") if isinstance(entry, dict) else None
            qc = qc_store.get(block) if isinstance(qc_store, dict) else None
        if not isinstance(qc, dict):
            return None
        scope = qc.get("derivation")
        if not isinstance(scope, str) or scope not in derivs or scope != selected:
            return None
        return qc
    if derivation_id is not None and selected != active:
        return None
    qc = manifest.get(block)
    if not isinstance(qc, dict):
        return None
    scope = qc.get("derivation")
    if scope is None or active is None:
        return qc
    return qc if scope == active else None


def set_qc_block(manifest: dict, block: str, value: dict) -> dict:
    """Store one active-lineage QC block without touching a sibling lineage."""
    if block not in _QC_BLOCKS:
        raise DerivationError(f"unknown QC block {block!r}")
    active = active_derivation(manifest)
    record = copy.deepcopy(value)
    record["derivation"] = active
    derivs = manifest.get("derivations")
    if isinstance(derivs, dict) and len(derivs) > 1:
        if active not in derivs:
            raise DerivationError(f"active_derivation {active!r} not in derivations")
        first = next(iter(derivs))
        if active != first:
            entry = derivs[active]
            if not isinstance(entry, dict):
                raise DerivationError(f"derivation {active!r} is not an object")
            qc_store = entry.setdefault("qc", {})
            if not isinstance(qc_store, dict):
                raise DerivationError(f"derivation {active!r}.qc must be an object")
            qc_store[block] = record
            return record
    manifest[block] = record
    return record
def active_kind(manifest: dict) -> str | None:
    """Kind of the active derivation, or None for un-migrated / malformed manifests."""
    active = active_derivation(manifest)
    derivs = manifest.get("derivations")
    if not isinstance(derivs, dict):
        return None
    entry = derivs.get(active)
    return entry.get("kind") if isinstance(entry, dict) else None


# Exact-copy strings from the owner's grill ruling — never rephrase.
_LABEL_UNCORRECTED = "no reverse-PE — ~3 mm geometric floor"
_LABEL_RPE_UNSIGNED = "reverse-PE corrected — residual uncertainty unquantified (delta QC unsigned)"
_LABEL_RPE_SIGNED = "reverse-PE corrected — median shift {median_mm:.1f} mm (signed delta QC)"


def delta_qc_signed(manifest: dict) -> bool:
    """True only when delta_qc is human-approved AND carries a usable median.

    Single source of truth for "signed" — ``floor_label`` and the
    ``/api/derivation`` endpoint both call this so the two can never disagree
    (Task 7 fix round 1). A NaN or missing ``median_mm`` is not a usable
    measurement even if a human approved the block, so it counts as unsigned
    rather than rendering a bogus value on the disclaimer surface.

    A block scoped to a non-active derivation also counts as unsigned
    (ADR-0004) — same read-time guard as ``atlas_prior_qc_ok`` and
    ``parcellation_qc_ok``; ``validate()`` refuses such manifests but is not
    on the serve path, so this predicate must not trust the block's scope.
    """
    delta_qc = qc_block_for(manifest, "delta_qc")
    if delta_qc is None or not delta_qc.get("approved_by"):
        return False
    median_mm = delta_qc.get("auto", {}).get("median_mm")
    if isinstance(median_mm, bool) or not isinstance(median_mm, (int, float)):
        return False
    return math.isfinite(median_mm)


def t1_qc_ok(man: dict) -> bool:
    """True only when human-signed t1_qc is present, complete, and scoped to
    the active derivation — same read-time guard as ``atlas_prior_qc_ok`` /
    ``parcellation_qc_ok`` (ADR-0002, ADR-0004)."""
    qc = qc_block_for(man, "t1_qc")
    if qc is None:
        return False
    if not qc.get("approved_by"):
        return False
    if not qc.get("date") or not qc.get("sheet_sha"):
        return False
    return True


def floor_label(manifest: dict) -> str:
    """Derivation-status floor label for the viewer HUD (owner ruling, verbatim).

    Un-migrated manifests get the conservative ``uncorrected`` default. A
    migrated manifest must declare a known kind; unknown kinds are errors, not
    guesses.
    """
    active = active_derivation(manifest)
    derivs = manifest.get("derivations")
    if isinstance(derivs, dict):
        entry = derivs.get(active)
        kind = entry.get("kind") if isinstance(entry, dict) else None
        if active in derivs and kind not in KINDS:
            raise DerivationError(
                f"derivation {active!r} has missing or unknown kind {kind!r}"
            )
    else:
        kind = None

    if kind == KIND_RPE_PAIR:
        delta_qc = qc_block_for(manifest, "delta_qc")
        if delta_qc_signed(manifest) and delta_qc is not None:
            median_mm = delta_qc["auto"]["median_mm"]
            return _LABEL_RPE_SIGNED.format(median_mm=median_mm)
        return _LABEL_RPE_UNSIGNED

    return _LABEL_UNCORRECTED


def _require_inputs(raw, label: str) -> dict:
    if not isinstance(raw, dict):
        raise DerivationError(f"{label} has no resolvable inputs block")
    return raw


def _inputs_for_derivation(manifest: dict, derivation_id: str, derivs: dict) -> dict:
    entry = derivs.get(derivation_id)
    if not isinstance(entry, dict):
        raise DerivationError(f"derivation {derivation_id!r} is not an object")
    if "inputs" in entry:
        return _require_inputs(entry["inputs"], f"derivation {derivation_id!r}")

    first_id = next(iter(derivs), None)
    if derivation_id == first_id:
        return _require_inputs(manifest.get("inputs"), "legacy top-level")
    raise DerivationError(
        f"derivation {derivation_id!r} has no inputs block; top-level inputs "
        f"belong only to first derivation {first_id!r}"
    )


def active_inputs(manifest: dict) -> dict:
    """Return the input declarations for the manifest's active lineage.

    A legacy manifest has one implicit lineage and its flat ``inputs`` block is
    used. In a migrated manifest, a derivation-owned block wins; otherwise the
    flat block remains attached only to the first derivation for compatibility.
    """
    derivs = manifest.get("derivations")
    if derivs is None:
        return _require_inputs(manifest.get("inputs"), "manifest")
    if not isinstance(derivs, dict):
        raise DerivationError("derivations must be an object")
    active = manifest.get("active_derivation")
    if active not in derivs:
        raise DerivationError(f"active_derivation {active!r} not in derivations")
    return _inputs_for_derivation(manifest, active, derivs)


def add_derivation(
    manifest: dict,
    id: str,
    kind: str,
    inputs: dict,
    note: str | None = None,
) -> dict:
    """Return a copy with one new, unsigned derivation appended.

    The active derivation is deliberately unchanged. Callers must migrate a
    legacy manifest first so the original lineage kind is explicit rather than
    guessed.
    """
    _require_derivation_id(id)
    if kind not in KINDS:
        raise ValueError(f"unknown derivation kind {kind!r}; expected one of {KINDS}")
    if not isinstance(inputs, dict):
        raise ValueError("derivation inputs must be a JSON object")
    derivs = manifest.get("derivations")
    if not isinstance(derivs, dict):
        raise ValueError("manifest has no derivations; run migrate first")
    if id in derivs:
        raise ValueError(f"derivation id {id!r} already exists")

    out = copy.deepcopy(manifest)
    entry = {"kind": kind, "inputs": copy.deepcopy(inputs)}
    if note is not None:
        entry["note"] = note
    out["derivations"][id] = entry
    validate(out)
    return out


def _declared_path_fields(meta: dict, *, include_static: bool = True):
    """Yield every declared path field as ``(field-name, relative-path)``."""
    for field in ("path", "sift2", "lut_path"):
        if include_static or field != "lut_path":
            if field in meta:
                yield field, meta[field]
    for field in ("and", "not"):
        if field not in meta:
            continue
        raw = meta[field]
        if not isinstance(raw, (list, tuple)):
            raise DerivationError(
                f"input field {field!r} must be a list of paths, got {type(raw).__name__}"
            )
        for index, value in enumerate(raw):
            yield f"{field}[{index}]", value


def _artifact_paths(manifest: dict, derivation_id: str, inputs: dict):
    root = str(manifest.get("case_root") or ".")
    for key, meta in inputs.items():
        if not isinstance(meta, dict):
            continue
        # LUTs are static lookup resources and may be shared. Only case-derived
        # paths participate in lineage isolation.
        for field, value in _declared_path_fields(meta, include_static=False):
            try:
                yield resolve_case_path(
                    root, value,
                    what=f"derivation {derivation_id!r} input {key!r}.{field}"
                )
            except ValueError as e:
                raise DerivationError(str(e)) from e


def _validate_input_metadata(derivation_id: str, inputs: dict, *, first: bool) -> None:
    """Require content metadata for path-bearing non-first declarations."""
    if first:
        return
    for key, meta in inputs.items():
        if not isinstance(meta, dict):
            continue
        fields = tuple(field for field, _ in _declared_path_fields(meta))
        if not fields:
            continue
        sha_by_field = meta.get("sha256_by_field")
        bytes_by_field = meta.get("bytes_by_field")
        if not isinstance(sha_by_field, dict) or not isinstance(bytes_by_field, dict):
            raise DerivationError(
                f"derivation {derivation_id!r} input {key!r} carries path fields "
                f"{fields} but requires sha256_by_field and bytes_by_field"
            )
        for field in fields:
            digest = sha_by_field.get(field)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise DerivationError(
                    f"derivation {derivation_id!r} input {key!r}.{field} sha256 "
                    "must be 64 lowercase hex characters"
                )
            size = bytes_by_field.get(field)
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise DerivationError(
                    f"derivation {derivation_id!r} input {key!r}.{field} bytes "
                    "must be a positive integer"
                )


def validate_active_artifact_metadata(manifest: dict) -> None:
    """Verify active non-first input bytes and hashes before serving the case."""
    derivs = manifest.get("derivations")
    if not isinstance(derivs, dict) or len(derivs) <= 1:
        return
    active = active_derivation(manifest)
    first = next(iter(derivs))
    if active == first:
        return
    inputs = active_inputs(manifest)
    _validate_input_metadata(active, inputs, first=False)
    root = _manifest_root(manifest)
    for key, meta in inputs.items():
        if not isinstance(meta, dict):
            continue
        path_fields = tuple(_declared_path_fields(meta))
        if not path_fields:
            continue
        sha_by_field = meta["sha256_by_field"]
        bytes_by_field = meta["bytes_by_field"]
        for field, raw_path in path_fields:
            try:
                resolved = Path(resolve_case_path(
                    str(root), raw_path,
                    what=f"active derivation {active!r} input {key!r}.{field}"
                ))
            except ValueError as e:
                raise ArtifactMetadataError(str(e)) from e
            try:
                actual_bytes = resolved.stat().st_size
                actual_sha = _sha256_file(resolved)
            except OSError as e:
                raise ArtifactMetadataError(
                    f"active derivation {active!r} input {key!r}.{field} "
                    f"cannot be read: {resolved}"
                ) from e
            declared_bytes = bytes_by_field[field]
            declared_sha = sha_by_field[field]
            if actual_bytes != declared_bytes:
                raise ArtifactMetadataError(
                    f"active derivation {active!r} input {key!r}.{field} "
                    f"bytes mismatch: declared {declared_bytes}, actual {actual_bytes}"
                )
            if actual_sha != declared_sha:
                raise ArtifactMetadataError(
                    f"active derivation {active!r} input {key!r}.{field} "
                    f"sha256 mismatch: declared {declared_sha}, actual {actual_sha}"
                )


_sha256_file = sha256_file


def write_manifest_atomic(manifest_path: str | Path, manifest: dict) -> None:
    """Persist a manifest without exposing a partially written JSON file."""
    path = Path(manifest_path)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(str(tmp_path), str(path))


def _qc_signed_for(manifest: dict, derivation_id: str) -> dict[str, bool]:
    signed: dict[str, bool] = {}
    for block in _QC_BLOCKS:
        qc = qc_block_for(manifest, block, derivation_id=derivation_id)
        if qc is None:
            signed[block] = False
            continue
        if block == "delta_qc":
            median_mm = qc.get("auto", {}).get("median_mm")
            signed[block] = bool(qc.get("approved_by")) and (
                not isinstance(median_mm, bool)
                and isinstance(median_mm, (int, float))
                and math.isfinite(median_mm)
            )
        else:
            signed[block] = bool(
                qc.get("approved_by") and qc.get("date") and qc.get("sheet_sha")
            )
    return signed


def derivation_summary(manifest: dict) -> dict:
    """Build the public, read-only derivation payload used by the server."""
    active = active_derivation(manifest)
    derivs = manifest.get("derivations")
    kind = derivs.get(active, {}).get("kind") if isinstance(derivs, dict) else None
    rows = []
    if isinstance(derivs, dict):
        for derivation_id, entry in derivs.items():
            row = {
                "id": derivation_id,
                "kind": entry.get("kind") if isinstance(entry, dict) else None,
                "qc_signed": _qc_signed_for(manifest, derivation_id),
            }
            rows.append(row)
    return {
        "active": active,
        "kind": kind,
        "derivations": rows,
        "floor_label": floor_label(manifest),
        "delta_qc_signed": delta_qc_signed(manifest),
    }


def validate(manifest: dict) -> None:
    """Fail closed on any cross-derivation mixing or malformed shape (ADR-0004 rule 2, S-06).

    Un-migrated manifests remain valid as the implicit single lineage. Migrated
    manifests must resolve every lineage's inputs and must not reuse an
    artifact path across lineages.
    """
    if not isinstance(manifest, dict):
        raise DerivationError(f"manifest must be a dict, got {type(manifest).__name__}")
    derivs = manifest.get("derivations")
    if derivs is None:
        return
    if not isinstance(derivs, dict) or not derivs:
        raise DerivationError("derivations must be a non-empty object")
    _validate_derivation_ids(derivs)
    active = manifest.get("active_derivation")
    if not isinstance(active, str) or active not in derivs:
        raise DerivationError(f"active_derivation {active!r} not in derivations")

    first_id = next(iter(derivs))
    for derivation_id, entry in derivs.items():
        if not isinstance(entry, dict):
            raise DerivationError(f"derivation {derivation_id!r} is not an object")
        kind = entry.get("kind")
        if kind not in KINDS:
            raise DerivationError(
                f"derivation {derivation_id!r} has missing or unknown kind {kind!r}"
            )

    active_inputs(manifest)  # the active lineage must be resolvable too
    resolved = {
        derivation_id: _inputs_for_derivation(manifest, derivation_id, derivs)
        for derivation_id in derivs
    }
    for derivation_id, inputs in resolved.items():
        _validate_input_metadata(
            derivation_id, inputs, first=derivation_id == first_id,
        )
    seen: dict[str, str] = {}
    for derivation_id, inputs in resolved.items():
        for path in set(_artifact_paths(manifest, derivation_id, inputs)):
            prior = seen.get(path)
            if prior is not None and prior != derivation_id:
                raise DerivationError(
                    f"artifact path shared by derivations {prior!r} and "
                    f"{derivation_id!r}: {path}"
                )
            seen[path] = derivation_id

    for block in _QC_BLOCKS:
        qc = manifest.get(block)
        if not isinstance(qc, dict):
            continue
        scope = qc.get("derivation")
        if len(derivs) > 1:
            if not isinstance(scope, str) or scope not in derivs:
                raise DerivationError(
                    f"{block} must carry an explicit derivation id when the "
                    "manifest has multiple derivations"
                )
            if scope != first_id:
                raise DerivationError(
                    f"{block} is scoped to derivation {scope!r}, active is "
                    f"{active!r}: top-level QC belongs to first derivation "
                    f"{first_id!r}"
                )
        elif scope not in (None, active):
            raise DerivationError(
                f"derivation mismatch: {block} is scoped to derivation "
                f"{scope!r}, active is {active!r}: signatures never cross "
                "derivations"
            )

    for derivation_id, entry in derivs.items():
        qc_store = entry.get("qc")
        if qc_store is None:
            continue
        if not isinstance(qc_store, dict):
            raise DerivationError(f"derivation {derivation_id!r}.qc must be an object")
        for block in _QC_BLOCKS:
            qc = qc_store.get(block)
            if not isinstance(qc, dict):
                continue
            scope = qc.get("derivation")
            if len(derivs) > 1:
                if scope != derivation_id:
                    raise DerivationError(
                        f"{block} under derivation {derivation_id!r} must carry "
                        "that derivation id"
                    )
            elif scope not in (None, active):
                raise DerivationError(
                    f"derivation mismatch: {block} is scoped to derivation "
                    f"{scope!r}, active is {active!r}: signatures never cross "
                    "derivations"
                )


def main(argv: list[str]) -> int:
    """CLI: migrate or add derivations, writing manifests atomically.

    Reads manifest.json, migrates, validates, and writes atomically.
    Returns 0 on success, 1 on ValueError (manifest unchanged on failure).
    """
    parser = argparse.ArgumentParser(
        prog="python -m tractlab.derivation",
        description="Derivation CLI: migrate manifests to append-only lineage tracking"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate_cmd = subparsers.add_parser("migrate", help="Migrate a case manifest")
    migrate_cmd.add_argument("case_root", type=str, help="Path to case root directory")
    migrate_cmd.add_argument("--kind", choices=KINDS, required=True,
                           help="Derivation kind (uncorrected or rpe_pair)")

    add_cmd = subparsers.add_parser("add", help="Append a derivation to a case manifest")
    add_cmd.add_argument("case_root", type=str, help="Path to case root directory")
    add_cmd.add_argument("--id", required=True, help="New derivation id")
    add_cmd.add_argument("--kind", choices=KINDS, required=True,
                         help="Derivation kind (uncorrected or rpe_pair)")
    add_cmd.add_argument("--inputs-json", required=True,
                         help="JSON file containing the derivation inputs object")
    add_cmd.add_argument("--note", default=None, help="Optional derivation note")

    args = parser.parse_args(argv)

    case_root = Path(args.case_root)
    manifest_path = case_root / "manifest.json"

    try:
        # Read manifest
        manifest_text = manifest_path.read_text()
        manifest = json.loads(manifest_text)

        if args.command == "migrate":
            updated = migrate_manifest(manifest, args.kind)
            message = f"Migrated {manifest_path} to kind={args.kind}"
        else:
            inputs = json.loads(Path(args.inputs_json).read_text())
            if not isinstance(inputs, dict):
                raise ValueError("--inputs-json must contain a JSON object")
            updated = add_derivation(
                manifest, args.id, args.kind, inputs, note=args.note,
            )
            message = f"Added derivation {args.id} to {manifest_path}"

        # Validate
        validate(updated)

        write_manifest_atomic(manifest_path, updated)

        print(message)
        return 0

    except ValueError as e:
        # Fail closed: manifest unchanged on error
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
