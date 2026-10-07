"""Independent, data-gated controls for public MR and CT anatomy capsules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from capsule.pack import read_capsule
from scripts.anatomy_review import anatomy_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public"
OUTPUT = ROOT / "out"
GROUPS = {
    "Ventrículos", "Tronco e cerebelo", "Núcleos profundos", "Estruturas límbicas",
    "Osso", "Seios e cavidades", "Vasos", "Nervos e órbita",
    "Lobo frontal", "Lobo parietal", "Lobo temporal", "Lobo occipital", "Lobo límbico (cíngulo)", "Ínsula",
}


def _public_demo(prefix: str, source: Path, *, allow_missing_anatomy: bool = False) -> tuple[dict, dict]:
    if not source.is_dir():
        pytest.skip(f"public anatomy source {source} not present")
    candidates = sorted(OUTPUT.glob(f"{prefix}.v2.*.capsule.html"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        pytest.skip(f"no completed capsule for {prefix}; run scripts/demo_build.sh first")
    for candidate in candidates:
        manifest, arrays = read_capsule(candidate)
        if manifest.get("anatomy"):
            return manifest, arrays
    if allow_missing_anatomy:
        pytest.skip(f"{len(candidates)} CT capsule(s) exist but no TotalSegmentator task produced anatomy")
    raise AssertionError(f"{len(candidates)} capsule(s) exist for {prefix}, but none contains anatomy")


def _check_shared_contract(manifest: dict, arrays: dict) -> dict:
    spacing = np.asarray(manifest["grid"]["spacing_mm"], dtype=float)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    dimensions = tuple(reversed(manifest["grid"]["dims"]))
    volume_ids = {volume["id"] for volume in manifest["volumes"]}
    anatomy = manifest["anatomy"]
    ids = [item["id"] for item in anatomy]
    assert len(ids) == len(set(ids))
    for item in anatomy:
        assert item["for_volume"] in volume_ids
        assert item["source"] == "auto" and item["reviewed"] is False
        labels = arrays[item["blob"]]
        assert labels.dtype == np.uint8 and labels.shape == dimensions
        values = [int(label["value"]) for label in item["labels"]]
        assert len(values) == len(set(values)) and all(1 <= value <= 255 for value in values)
        for label in item["labels"]:
            assert label["group"] in GROUPS
            voxels = int(np.count_nonzero(labels == int(label["value"])))
            independent_volume_ml = voxels * voxel_ml
            assert abs(independent_volume_ml - float(label["volume_ml"])) <= 0.000001
    _, checks = anatomy_report(manifest, arrays)
    assert checks["label_volume_checks"]["failures"] == 0
    for pair in checks["paired_structures"]:
        left = pair["left_volume_ml"]
        right = pair["right_volume_ml"]
        if left == right == 0:
            expected = "UNMEASURED (both empty)"
        elif left == 0 or right == 0:
            expected = "FLAG (one side empty)"
        else:
            expected = "FLAG (>1.5)" if max(left, right) / min(left, right) > 1.5 else "PASS"
        assert pair["status"] == expected
    print(f"anatomy labels={checks['label_volume_checks']['count']}; pairs={checks['paired_structures']}; "
          f"licences={checks['tool_licences']}")
    return checks


def test_public_vestibular_schwannoma_mr_anatomy_controls():
    source = DATA / "Vestibular-Schwannoma-SEG" / "brain-mri"
    manifest, arrays = _public_demo("vs-seg-brain-mri", source)

    checks = _check_shared_contract(manifest, arrays)

    assert [item["id"] for item in manifest["anatomy"]] == ["anat_mr", "anat_mr_gyri"]
    item, gyri = manifest["anatomy"]
    assert len(gyri["labels"]) == 68 and "--parc" in gyri["method"]
    assert {label["group"] for label in gyri["labels"]} == {
        "Lobo frontal", "Lobo parietal", "Lobo temporal", "Lobo occipital", "Lobo límbico (cíngulo)", "Ínsula"}
    assert "synthseg 2.0" in item["method"] and "--robust" in item["method"]
    assert item["licence"].startswith("FreeSurfer Software License Agreement")
    assert checks["synthseg_grid_alignment"]["fraction"] >= 0.95
    assert checks["synthseg_grid_alignment"]["status"] == "PASS"
    assert all(control["status"] == "PASS" for control in checks["lateral_ventricle_centroids"].values())


def test_public_cq500_ct_anatomy_controls():
    source = DATA / "CQ500" / "head-ct-thin"
    manifest, arrays = _public_demo("cq500-head-ct-thin", source, allow_missing_anatomy=True)

    _check_shared_contract(manifest, arrays)

    assert all(item["id"].startswith("anat_ct_") for item in manifest["anatomy"])
    assert all(item["method"].startswith("totalsegmentator ") and " task " in item["method"]
               for item in manifest["anatomy"])
    assert all(item["licence"].startswith("Apache-2.0") for item in manifest["anatomy"])
