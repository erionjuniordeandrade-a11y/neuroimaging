"""T1↔b0 registration trust checks (Slice A0).

Emits an edge-overlay QC sheet and auto-sanity metrics. Fail-closed exit codes.
Human approval is recorded separately via ``--approve`` — automation never
approves (ADR-0002 pattern for t1 underlay).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

from . import derivation
from .grid import (
    assert_grids_match,
    grid_id,
    load_grid,
    unknown_units_assumption_from_manifest,
)


@dataclass(frozen=True)
class AutoSanity:
    ok: bool
    grid_id_match: bool
    shape_match: bool
    mask_overlap_frac: float
    t1_nonzero_frac: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class CheckResult:
    outcome: str  # "ok" | "fail"
    sheet_path: str | None
    sheet_sha256: str | None
    auto: AutoSanity


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_OPP = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}


def anatomical_plane(
    arr: np.ndarray,
    axis: int,
    idx: int,
    axcodes: tuple[str, str, str],
) -> np.ndarray:
    """2-D slice for ``imshow(..., origin='upper')`` in anatomical convention.

    Superior (coronal/sagittal) or anterior (axial) is toward the top of the
    image. When L/R is in-plane, the view is radiological (patient R on the
    viewer's left). Sagittal puts anterior to the right.

    House grids differ: Leipzig D0 is L/A/S; the 3T case is L/P/S. A bare
    ``np.rot90`` + ``origin='lower'`` inverts L/A/S (vertex at the bottom).
    """
    if axis == 0:
        sl = np.asarray(arr[idx, :, :])
        inplane = (axcodes[1], axcodes[2])
    elif axis == 1:
        sl = np.asarray(arr[:, idx, :])
        inplane = (axcodes[0], axcodes[2])
    elif axis == 2:
        sl = np.asarray(arr[:, :, idx])
        inplane = (axcodes[0], axcodes[1])
    else:
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")

    want_up = "A" if axis == 2 else "S"
    if inplane[1] in (want_up, _OPP[want_up]):
        sl = sl.T
        inplane = (inplane[1], inplane[0])
    if inplane[0] == want_up:
        sl = np.flipud(sl)
        inplane = (_OPP[inplane[0]], inplane[1])

    horiz = inplane[1]
    # L already has patient R at col 0; R or P need a mirror (radiological / nose-right).
    if horiz in ("R", "P"):
        sl = np.fliplr(sl)
    return sl


def _label_laterality(ax, title: str) -> None:
    """Radiological L/R on axial/coronal; anterior-right on sagittal."""
    kw = dict(color="#e8c56a", fontsize=8, fontweight="bold", clip_on=False)
    if title == "sagittal":
        ax.text(0.04, 0.50, "P", transform=ax.transAxes, va="center", ha="left", **kw)
        ax.text(0.96, 0.50, "A", transform=ax.transAxes, va="center", ha="right", **kw)
    else:
        ax.text(0.04, 0.50, "R", transform=ax.transAxes, va="center", ha="left", **kw)
        ax.text(0.96, 0.50, "L", transform=ax.transAxes, va="center", ha="right", **kw)
    ax.text(0.50, 0.96, "S" if title != "axial" else "A", transform=ax.transAxes,
            va="top", ha="center", **kw)


def _edges(vol: np.ndarray) -> np.ndarray:
    m = vol.astype(np.float32)
    if m.max() > 0:
        m = m / float(m.max())
    gx = ndimage.sobel(m, axis=0)
    gy = ndimage.sobel(m, axis=1)
    e = np.hypot(gx, gy)
    if e.max() > 0:
        e = e / float(e.max())
    return e


def auto_sanity(
    *,
    b0_path: str,
    t1_path: str,
    mask_path: str,
    expected_grid_id: str | None,
    assume_unknown_spatial_units_mm: bool = False,
) -> AutoSanity:
    notes: list[str] = []
    try:
        g_b0 = load_grid(
            b0_path,
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
            require_authoritative_affine=True,
        )
        g_t1 = load_grid(
            t1_path,
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
            require_authoritative_affine=True,
        )
        g_mask = load_grid(
            mask_path,
            require_mm=True,
            assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
            require_authoritative_affine=True,
        )
    except ValueError as exc:
        notes.append(str(exc))
        return AutoSanity(
            ok=False,
            grid_id_match=False,
            shape_match=False,
            mask_overlap_frac=0.0,
            t1_nonzero_frac=0.0,
            notes=tuple(notes),
        )
    shape_match = g_b0.shape == g_t1.shape
    if not shape_match:
        notes.append(f"shape mismatch b0={g_b0.shape} t1={g_t1.shape}")
    # Affines and known spatial units must agree before any same-index metric.
    t1_grid_ok = True
    try:
        assert_grids_match(g_b0, g_t1, context="b0/t1 grid")
    except ValueError as exc:
        t1_grid_ok = False
        notes.append(str(exc))
        shape_match = False  # treat as fail

    gid = grid_id(g_b0)
    grid_id_match = True
    if expected_grid_id:
        grid_id_match = gid == expected_grid_id
        if not grid_id_match:
            notes.append(f"grid_id mismatch manifest={expected_grid_id[:12]}… live={gid[:12]}…")

    b0 = np.asanyarray(nib.load(b0_path).dataobj, dtype=np.float64)
    t1 = np.asanyarray(nib.load(t1_path).dataobj, dtype=np.float64)
    mask_img = nib.load(mask_path)
    mask_grid_ok = True
    try:
        assert_grids_match(g_b0, g_mask, context="b0/mask grid")
    except ValueError as exc:
        mask_grid_ok = False
        notes.append(str(exc))
    mask = np.asanyarray(mask_img.dataobj) > 0
    if b0.shape != mask.shape or t1.shape != mask.shape:
        notes.append("mask shape mismatch")
        return AutoSanity(
            ok=False,
            grid_id_match=grid_id_match,
            shape_match=False,
            mask_overlap_frac=0.0,
            t1_nonzero_frac=0.0,
            notes=tuple(notes),
        )
    if not t1_grid_ok or not mask_grid_ok:
        return AutoSanity(
            ok=False,
            grid_id_match=grid_id_match,
            shape_match=False,
            mask_overlap_frac=0.0,
            t1_nonzero_frac=0.0,
            notes=tuple(notes),
        )

    # T1 energy inside mask
    t1_in = t1[mask]
    t1_nz = float(np.mean(t1_in > (np.percentile(t1_in, 5) if t1_in.size else 0)))
    if t1_nz < 0.5:
        notes.append(f"t1 sparse in mask ({t1_nz:.2f})")

    # edge overlap: correlation of edge maps inside mask
    eb = _edges(b0)
    et = _edges(t1)
    mb = eb[mask]
    mt = et[mask]
    if mb.std() < 1e-8 or mt.std() < 1e-8:
        overlap = 0.0
        notes.append("degenerate edges")
    else:
        mb = (mb - mb.mean()) / (mb.std() + 1e-12)
        mt = (mt - mt.mean()) / (mt.std() + 1e-12)
        overlap = float(np.mean(mb * mt))
        # map correlation-ish [-1,1] roughly; require positive alignment
        if overlap < 0.05:
            notes.append(f"low edge correlation {overlap:.3f}")

    ok = shape_match and grid_id_match and t1_nz >= 0.5 and overlap >= 0.05
    return AutoSanity(
        ok=ok,
        grid_id_match=grid_id_match,
        shape_match=shape_match,
        mask_overlap_frac=overlap,
        t1_nonzero_frac=t1_nz,
        notes=tuple(notes),
    )


def write_qc_sheet(
    *,
    b0_path: str,
    t1_path: str,
    mask_path: str,
    out_png: str | Path,
    floor_label_text: str | None = None,
) -> str:
    """RGB edge overlay: R=b0 edges, G=t1 edges, B=overlap. Returns sha256.

    ``floor_label_text`` is ``derivation.floor_label(manifest)`` for the case being
    sheeted; the title never asserts a derivation on its own (audit S-4 family).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b0_img = nib.load(b0_path)
    axcodes = tuple(nib.aff2axcodes(b0_img.affine))
    b0 = np.asanyarray(b0_img.dataobj, dtype=np.float64)
    t1 = np.asanyarray(nib.load(t1_path).dataobj, dtype=np.float64)
    mask = np.asanyarray(nib.load(mask_path).dataobj) > 0
    eb = _edges(b0)
    et = _edges(t1)
    k = b0.shape[2] // 2
    j = b0.shape[1] // 2
    i = b0.shape[0] // 2

    def plane(arr, axis, idx):
        return anatomical_plane(arr, axis, idx, axcodes)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), facecolor="#0d0f12")
    titles = ["axial", "coronal", "sagittal"]
    slices = [(2, k), (1, j), (0, i)]
    for col, ((axis, idx), title) in enumerate(zip(slices, titles)):
        rb = plane(eb, axis, idx)
        gt = plane(et, axis, idx)
        m = plane(mask.astype(np.float32), axis, idx)
        rgb = np.stack([rb, gt, np.minimum(rb, gt)], axis=-1)
        rgb = np.clip(rgb * (0.35 + 0.65 * m[..., None]), 0, 1)
        axes[0, col].imshow(rgb, origin="upper")
        axes[0, col].set_title(f"{title} edges R=b0 G=t1", color="#ccc", fontsize=9)
        axes[0, col].axis("off")
        _label_laterality(axes[0, col], title)
        a = plane(b0, axis, idx)
        b = plane(t1, axis, idx)
        a = a / (np.percentile(a[a > 0], 98) + 1e-9) if (a > 0).any() else a
        b = b / (np.percentile(b[b > 0], 98) + 1e-9) if (b > 0).any() else b
        blend = np.stack([np.clip(a, 0, 1), np.clip(b, 0, 1), np.zeros_like(a)], axis=-1)
        axes[1, col].imshow(blend, origin="upper")
        axes[1, col].set_title(f"{title} R=b0 G=t1", color="#ccc", fontsize=9)
        axes[1, col].axis("off")
        _label_laterality(axes[1, col], title)
    title = "T1 vs b0 registration QC — research aid only"
    if floor_label_text:
        title += f" ({floor_label_text})"
    fig.suptitle(title, color="#e8c56a", fontsize=11)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return sha256_file(out_png)


def run_check(
    *,
    b0_path: str,
    t1_path: str,
    mask_path: str,
    out_dir: str | Path,
    expected_grid_id: str | None = None,
    floor_label_text: str | None = None,
    assume_unknown_spatial_units_mm: bool = False,
) -> CheckResult:
    out_dir = Path(out_dir)
    auto = auto_sanity(
        b0_path=b0_path,
        t1_path=t1_path,
        mask_path=mask_path,
        expected_grid_id=expected_grid_id,
        assume_unknown_spatial_units_mm=assume_unknown_spatial_units_mm,
    )
    sheet = out_dir / "t1_b0_reg_qc.png"
    try:
        sheet_sha = write_qc_sheet(
            b0_path=b0_path, t1_path=t1_path, mask_path=mask_path, out_png=sheet,
            floor_label_text=floor_label_text,
        )
    except Exception as e:  # noqa: BLE001 — surface as fail outcome
        return CheckResult(
            outcome="fail",
            sheet_path=None,
            sheet_sha256=None,
            auto=AutoSanity(
                ok=False,
                grid_id_match=auto.grid_id_match,
                shape_match=auto.shape_match,
                mask_overlap_frac=auto.mask_overlap_frac,
                t1_nonzero_frac=auto.t1_nonzero_frac,
                notes=auto.notes + (f"sheet write failed: {e}",),
            ),
        )
    return CheckResult(
        outcome="ok" if auto.ok else "fail",
        sheet_path=str(sheet),
        sheet_sha256=sheet_sha,
        auto=auto,
    )


def approve_t1_qc(
    manifest_path: str,
    *,
    approved_by: str,
    sheet_sha: str | None = None,
) -> dict:
    """Write t1_qc approval into case manifest. Does not auto-approve."""
    man_path = Path(manifest_path)
    man = json.loads(man_path.read_text())
    derivation.validate(man)
    root = Path(man["case_root"]).resolve()
    layout = {**man, "case_root": str(root)}
    sheet = derivation.artifact_path(layout, "qc", "t1_b0_reg_qc.png")
    if not sheet.is_file():
        raise FileNotFoundError(f"QC sheet missing: {sheet} — run check first")
    live_sha = sha256_file(sheet)
    if sheet_sha and sheet_sha != live_sha:
        raise ValueError(
            f"sheet_sha mismatch: provided {sheet_sha[:12]}… live {live_sha[:12]}…"
        )
    qc = derivation.qc_block_for(layout, "t1_qc")
    if qc is None:
        raise derivation.DerivationError(
            "no active-lineage t1_qc auto QC block — run check first"
        )
    auto = qc.get("auto")
    if not isinstance(auto, dict) or auto.get("ok") is not True:
        raise derivation.DerivationError(
            "auto QC is not ok — refuse human approve (fix first)"
        )
    result = derivation.set_qc_block(man, "t1_qc", {
        "auto": auto,
        "approved_by": approved_by,
        "date": date.today().isoformat(),
        "sheet_sha": live_sha,
        "sheet_path": sheet.relative_to(root).as_posix(),
    })
    derivation.write_manifest_atomic(man_path, man)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--approve", metavar="WHO", help="record human approval (name/handle)")
    ap.add_argument("--sheet-sha", default=None, help="optional echo of QC sheet sha")
    args = ap.parse_args(argv)

    man_path = Path(args.manifest)
    man = json.loads(man_path.read_text())
    derivation.validate(man)
    root = Path(man["case_root"]).resolve()
    inputs = derivation.active_inputs(man)
    b0 = root / inputs["b0"]["path"]
    t1 = root / inputs["t1"]["path"]
    mask = root / inputs["mask"]["path"]
    expected = man.get("grid", {}).get("grid_id")
    try:
        assume_unknown_units_mm = unknown_units_assumption_from_manifest(man)
    except ValueError as exc:
        print(f"UNIT POLICY FAILED: {exc}", file=sys.stderr)
        return 2

    if args.approve:
        try:
            rec = approve_t1_qc(
                str(man_path), approved_by=args.approve, sheet_sha=args.sheet_sha,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"APPROVE FAILED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(rec, indent=2))
        return 0

    sheet_path = derivation.artifact_path(man, "qc", "t1_b0_reg_qc.png")
    # Clear any prior signature before regenerating the QC artefact. The
    # manifest update is atomic and precedes the first artefact write, so an
    # interrupted regeneration cannot leave an old signature over new bytes.
    if derivation.qc_block_for(man, "t1_qc") is not None:
        derivation.set_qc_block(man, "t1_qc", {
            "approved_by": None,
            "date": None,
            "sheet_sha": None,
            "sheet_path": sheet_path.relative_to(root).as_posix(),
            "voided": True,
            "note": "voided by registration QC regeneration",
        })
        derivation.write_manifest_atomic(man_path, man)
    res = run_check(
        b0_path=str(b0),
        t1_path=str(t1),
        mask_path=str(mask),
        out_dir=sheet_path.parent,
        expected_grid_id=expected,
        floor_label_text=derivation.floor_label(man),
        assume_unknown_spatial_units_mm=assume_unknown_units_mm,
    )
    # persist auto block into manifest without approving
    derivation.set_qc_block(man, "t1_qc", {
        "auto": {
            "ok": res.auto.ok,
            "grid_id_match": res.auto.grid_id_match,
            "shape_match": res.auto.shape_match,
            "mask_overlap_frac": res.auto.mask_overlap_frac,
            "t1_nonzero_frac": res.auto.t1_nonzero_frac,
            "notes": list(res.auto.notes),
        },
        "approved_by": None,
        "date": None,
        "sheet_sha": res.sheet_sha256,
        "sheet_path": (
            Path(res.sheet_path).resolve().relative_to(root.resolve()).as_posix()
            if res.sheet_path else None
        ),
    })
    derivation.write_manifest_atomic(man_path, man)
    print(json.dumps({"outcome": res.outcome, **asdict(res.auto),
                      "sheet": res.sheet_path, "sheet_sha": res.sheet_sha256,
                      "floor_label": derivation.floor_label(man)}, indent=2))
    return 0 if res.outcome == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
