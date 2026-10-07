"""CLI contract tests for the manifest-driven CR peel runner."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np

from tractlab.cr import cli as cr_cli


REPO = Path(__file__).resolve().parents[1]
HONESTY_WALL = "Research visualization only · enhancing vessels · not navigation, not a device"


def _run_cli(
    case_root: Path, *args: str, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(REPO / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "tractlab.cr", str(case_root), *args],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )


def _write_manifest(case_root: Path, payload: dict) -> None:
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_nifti(
    data: np.ndarray, affine: np.ndarray, path: Path, spatial_unit: str | None = "mm"
) -> None:
    image = nib.Nifti1Image(data, affine)
    if spatial_unit is not None:
        image.header.set_xyzt_units(spatial_unit)
    nib.save(image, path)


def _write_case(
    case_root: Path,
    input_root: Path,
    volume: np.ndarray,
    brain: np.ndarray,
    affine: np.ndarray,
    *,
    root_literal: str | None = None,
    case_id: str = "synthetic-sphere",
    spatial_unit: str | None = "mm",
) -> tuple[Path, Path]:
    input_root.mkdir(parents=True, exist_ok=True)
    t1_path = input_root / "t1.nii.gz"
    mask_path = input_root / "mask.nii.gz"
    _save_nifti(volume, affine, t1_path, spatial_unit)
    _save_nifti(brain.astype(np.uint8), affine, mask_path, spatial_unit)
    _write_manifest(
        case_root,
        {
            "deid": True,
            "case_id": case_id,
            "inputs": {
                "t1c_unstripped": {
                    "root": root_literal if root_literal is not None else str(input_root),
                    "path": t1_path.name,
                    "brain_mask": mask_path.name,
                    "series": "synthetic post-contrast",
                    "contrast": "gadolinium",
                    "provenance": {"source": "pytest sphere phantom"},
                }
            },
        },
    )
    return t1_path, mask_path


def _sphere_phantom() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (64, 64, 64)
    centre = np.array([32.0, 32.0, 30.0])
    grid = np.indices(shape, dtype=np.float32)
    radius = np.sqrt(sum((grid[axis] - centre[axis]) ** 2 for axis in range(3)))
    volume = np.zeros(shape, np.float32)
    volume[radius <= 29.0] = 800.0
    volume[radius <= 24.0] = 50.0
    brain = radius <= 20.0
    volume[brain] = 400.0
    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 63.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return volume, brain, affine


def _string_values(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)


def _directory_snapshot(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}


def test_manifest_without_deid_true_is_refused(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    _write_manifest(case_root, {"case_id": "synthetic"})

    proc = _run_cli(case_root, "--dry-run")

    assert proc.returncode == 2
    assert "deid" in proc.stderr
    assert not (case_root / "cr").exists()


def test_input_path_escaping_root_is_refused(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    input_root = tmp_path / "input"
    input_root.mkdir()
    _write_manifest(
        case_root,
        {
            "deid": True,
            "case_id": "synthetic",
            "inputs": {
                "t1c_unstripped": {
                    "root": str(input_root),
                    "path": "../outside.nii.gz",
                    "brain_mask": "mask.nii.gz",
                    "series": "synthetic post-contrast",
                    "contrast": "gadolinium",
                    "provenance": {"source": "phantom"},
                }
            },
        },
    )

    proc = _run_cli(case_root, "--dry-run")

    assert proc.returncode == 2
    assert "path" in proc.stderr
    assert "escapes" in proc.stderr
    assert not (case_root / "cr").exists()


def test_mask_grid_mismatch_is_refused_with_both_affines(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    input_root = tmp_path / "input"
    input_root.mkdir()
    t1_affine = np.eye(4)
    mask_affine = np.eye(4)
    mask_affine[0, 3] = 1.0
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), t1_affine), input_root / "t1.nii.gz")
    nib.save(
        nib.Nifti1Image(np.ones((8, 8, 8), np.uint8), mask_affine),
        input_root / "mask.nii.gz",
    )
    _write_manifest(
        case_root,
        {
            "deid": True,
            "case_id": "synthetic",
            "inputs": {
                "t1c_unstripped": {
                    "root": str(input_root),
                    "path": "t1.nii.gz",
                    "brain_mask": "mask.nii.gz",
                    "series": "synthetic post-contrast",
                    "contrast": "gadolinium",
                    "provenance": {"source": "phantom"},
                }
            },
        },
    )

    proc = _run_cli(case_root, "--dry-run")

    assert proc.returncode == 2
    assert "grid mismatch" in proc.stderr
    assert "T1 affine" in proc.stderr
    assert "mask affine" in proc.stderr
    assert not (case_root / "cr").exists()


def test_anisotropic_zooms_are_refused_with_affine_metric(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    affine = np.diag([1.0, 1.0, 1.1, 1.0])
    _write_case(
        case_root,
        tmp_path / "input",
        np.zeros((8, 8, 8), np.float32),
        np.ones((8, 8, 8), bool),
        affine,
    )

    proc = _run_cli(case_root, "--dry-run")

    assert proc.returncode == 2
    assert "A.T @ A" in proc.stderr
    assert "within 2%" in proc.stderr
    assert not (case_root / "cr").exists()


def test_sheared_affine_is_refused_with_gram_matrix(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    volume, brain, _ = _sphere_phantom()
    affine = np.eye(4)
    affine[:3, :3] = np.array(
        [[1.0, 0.2, 0.0], [0.0, np.sqrt(0.96), 0.0], [0.0, 0.0, 1.0]]
    )
    _write_case(
        case_root,
        tmp_path / "input",
        volume,
        brain,
        affine,
    )

    proc = _run_cli(case_root, "--dry-run")

    assert proc.returncode == 2
    assert "A.T @ A" in proc.stderr
    assert "0.2" in proc.stderr
    assert not (case_root / "cr").exists()


def test_unknown_spatial_units_are_refused(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    volume, brain, affine = _sphere_phantom()
    _write_case(
        case_root,
        tmp_path / "input",
        volume,
        brain,
        affine,
        spatial_unit=None,
    )

    proc = _run_cli(case_root, "--dry-run")

    assert proc.returncode == 2
    assert "spatial units" in proc.stderr
    assert "mm" in proc.stderr
    assert not (case_root / "cr").exists()


def test_tiny_bright_blob_below_ten_percent_of_fov_is_refused(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    shape = (48, 48, 48)
    grid = np.indices(shape, dtype=np.float32)
    radius = np.sqrt(sum((grid[axis] - 24.0) ** 2 for axis in range(3)))
    volume = np.zeros(shape, np.float32)
    volume[radius <= 6.0] = 800.0
    brain = radius <= 4.0
    _write_case(case_root, tmp_path / "input", volume, brain, np.eye(4))

    proc = _run_cli(case_root, "--dry-run")

    assert proc.returncode == 2
    assert "fraction_fov" in proc.stdout
    assert "required range is 10%-95%" in proc.stderr
    assert not (case_root / "cr").exists()


def test_brain_voxel_outside_head_envelope_is_refused(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    volume, brain, affine = _sphere_phantom()
    brain[0, 0, 0] = True
    _write_case(case_root, tmp_path / "input", volume, brain, affine)

    proc = _run_cli(case_root, "--dry-run")

    assert proc.returncode == 2
    assert "1 brain-mask voxels lie outside the head envelope" in proc.stderr
    assert not (case_root / "cr").exists()


def test_cr_outputs_are_ignored_but_receipt_is_not() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", "cases/x/cr/foo.ply"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    receipt = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", "cases/x/cr/summary.json"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    receipt_quiet = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "cases/x/cr/summary.json"],
        cwd=REPO,
        check=False,
    )

    assert ignored.returncode == 0, ignored.stderr
    toplevel = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if Path(toplevel).resolve() != REPO.resolve():
        # Inside the neuroimaging monorepo the root deny on cases/ wins:
        # no case file, receipt included, can be tracked there.
        assert receipt_quiet.returncode == 0
        return
    assert "/cases/*/cr/*" in ignored.stdout
    assert "!/cases/*/cr/summary.json" in receipt.stdout
    assert receipt_quiet.returncode == 1


def test_end_to_end_synthetic_case_writes_bound_receipt(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    fake_home = tmp_path / "home"
    input_root = fake_home / "input"
    volume, brain, affine = _sphere_phantom()
    t1_path, mask_path = _write_case(
        case_root,
        input_root,
        volume,
        brain,
        affine,
        root_literal="~/input",
        case_id="Synthetic Sphere free text",
    )
    output_dir = case_root / "cr"
    output_dir.mkdir()
    (output_dir / "d99_grey.ply").write_bytes(b"stale output from an older run")

    proc = _run_cli(
        case_root,
        "--depths",
        "0:16:4",
        env_overrides={"HOME": str(fake_home)},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "head QC:" in proc.stdout
    receipt_path = case_root / "cr" / "summary.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "created_at",
        "honesty_wall",
        "inputs",
        "verified_before_publish",
        "vx",
        "code_revision",
        "code_dirty",
        "params",
        "marker_depth_mm",
        "marker_p10_p90_mm",
        "vessel_thr",
        "tophat_thr",
        "n_vertices",
        "n_faces",
        "brain_outside_envelope",
        "per_depth",
        "outputs",
    }
    assert required <= receipt.keys()
    assert receipt["schema"] == "tractlab.cr.summary/1"
    assert receipt["honesty_wall"] == HONESTY_WALL
    assert "case_id" not in receipt
    assert receipt["verified_before_publish"] is True
    assert receipt["brain_outside_envelope"] == 0
    datetime.fromisoformat(receipt["created_at"].replace("Z", "+00:00"))
    assert len(receipt["code_revision"]) == 40
    assert isinstance(receipt["code_dirty"], bool)
    if receipt["code_dirty"]:
        assert len(receipt["code_diff_sha256"]) == 64
    else:
        assert "code_diff_sha256" not in receipt
    assert receipt["params"] == {
        "head_thr": 200.0,
        "head_blur_mm": 0.7,
        "env_sigma_mm": 2.8,
        "base_cut_below_centroid_mm": 25.0,
        "band_below_marker_mm": 2.0,
        "band_above_marker_mm": 6.0,
        "convexity_above_centroid_mm": 20.0,
        "depths_mm": [0.0, 4.0, 8.0, 12.0, 16.0],
        "slab_mm": 1.0,
        "slab_samples": 5,
        "vessel_pct": 97.5,
        "tophat_radius_mm": 1.5,
        "tophat_frac": 0.5,
        "direction_sigma_mm": 8.0,
        "smooth_iters": 20,
        "newton_iters": 12,
        "newton_tol_mm": 0.05,
        "max_residual_mm": 0.5,
        "max_edge_stretch": 3.0,
    }
    assert receipt["n_vertices"] > 100
    assert receipt["n_faces"] > 100
    assert len(receipt["per_depth"]) == 5
    for depth in receipt["per_depth"]:
        assert {
            "depth_mm",
            "depth_residual_mm",
            "folded_faces",
            "folded_faces_convexity",
        "invalid_columns",
        } == set(depth)

    for key, expected_path in (("t1c_unstripped", t1_path), ("brain_mask", mask_path)):
        bound = receipt["inputs"][key]
        assert bound["sha256"] == _sha256(expected_path)
        assert bound["shape"] == [64, 64, 64]
        assert np.array_equal(np.asarray(bound["affine"]), affine)
    t1_bound = receipt["inputs"]["t1c_unstripped"]
    assert t1_bound["root_literal"] == "~/input"
    assert t1_bound["root_sha256"] == hashlib.sha256(
        str(input_root.resolve()).encode("utf-8")
    ).hexdigest()
    assert "root" not in t1_bound
    strings = list(_string_values(receipt))
    assert all(str(fake_home.resolve()) not in value for value in strings)
    assert all(str(Path.home()) not in value for value in strings)

    expected_outputs = {"cr/peel.npz"}
    for depth in (0, 4, 8, 12, 16):
        expected_outputs.add(f"cr/d{depth:02d}_grey.ply")
        expected_outputs.add(f"cr/d{depth:02d}_vmax.ply")
        expected_outputs.add(f"cr/d{depth:02d}_vessel.ply")
    assert set(receipt["outputs"]) == expected_outputs
    assert not (output_dir / "d99_grey.ply").exists()
    for relative, digest in receipt["outputs"].items():
        output_path = case_root / relative
        assert output_path.is_file()
        assert digest == _sha256(output_path)

    with np.load(case_root / "cr" / "peel.npz") as peel:
        assert set(peel.files) == {
            "faces",
            "verts_world",
            "grey",
            "vmax",
            "vessel",
            "in_brain",
            "valid",
            "depths_mm",
        }
        assert peel["verts_world"].dtype == np.float32
        assert peel["verts_world"].shape[0] == 5


def test_input_change_before_publish_refuses_without_changing_cr(
    tmp_path: Path, monkeypatch
) -> None:
    case_root = tmp_path / "case"
    volume, brain, affine = _sphere_phantom()
    t1_path, _ = _write_case(case_root, tmp_path / "input", volume, brain, affine)
    output_dir = case_root / "cr"
    output_dir.mkdir()
    (output_dir / "summary.json").write_bytes(b"old receipt")
    (output_dir / "d99_grey.ply").write_bytes(b"old mesh")
    before = _directory_snapshot(output_dir)
    original_write_ply = cr_cli._write_ply
    tampered = False

    def tamper_after_first_ply(stream, verts_world, faces, rgb):
        nonlocal tampered
        original_write_ply(stream, verts_world, faces, rgb)
        if not tampered:
            with t1_path.open("ab") as input_stream:
                input_stream.write(b"changed-after-load")
            tampered = True

    monkeypatch.setattr(cr_cli, "_write_ply", tamper_after_first_ply)

    returncode = cr_cli.main([str(case_root), "--depths", "0:16:4"])

    assert returncode == 2
    assert _directory_snapshot(output_dir) == before
    assert not list(output_dir.glob(".stage-*"))


def test_stage_is_removed_and_cr_is_unchanged_when_ply_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    case_root = tmp_path / "case"
    volume, brain, affine = _sphere_phantom()
    _write_case(case_root, tmp_path / "input", volume, brain, affine)
    output_dir = case_root / "cr"
    output_dir.mkdir()
    (output_dir / "summary.json").write_bytes(b"old receipt")
    (output_dir / "d99_grey.ply").write_bytes(b"old mesh")
    before = _directory_snapshot(output_dir)
    original_write_ply = cr_cli._write_ply
    calls = 0

    def fail_during_second_ply(stream, verts_world, faces, rgb):
        nonlocal calls
        calls += 1
        original_write_ply(stream, verts_world, faces, rgb)
        if calls == 2:
            raise OSError("synthetic PLY failure")

    monkeypatch.setattr(cr_cli, "_write_ply", fail_during_second_ply)

    returncode = cr_cli.main([str(case_root), "--depths", "0:16:4"])

    assert returncode == 2
    assert calls == 2
    assert _directory_snapshot(output_dir) == before
    assert not list(output_dir.glob(".stage-*"))
