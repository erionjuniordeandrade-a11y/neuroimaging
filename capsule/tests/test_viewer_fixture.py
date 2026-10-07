from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from conftest import parse_capsule


def _voxel_at_ras(affine: list[list[float]], point: tuple[float, float, float]) -> tuple[int, int, int]:
    origin = [affine[row][3] for row in range(3)]
    return tuple(round(point[i] - origin[i]) for i in range(3))


def test_fixture_format_values_geometry_and_masks(capsule_file: Path) -> None:
    manifest, blobs = parse_capsule(capsule_file)
    assert manifest["schema"] == "case-capsule/1"
    assert manifest["version"] == 1
    assert manifest["locale"] == "pt-BR"
    assert manifest["case"]["anonymized"] is True

    grid = manifest["grid"]
    nx, ny, nz = grid["dims"]
    assert (nx, ny, nz) == (160, 192, 160)
    assert grid["spacing_mm"] == [1.0, 1.0, 1.0]
    affine = grid["affine_ras"]
    assert affine[3] == [0.0, 0.0, 0.0, 1.0]

    volumes = {volume["id"]: volume for volume in manifest["volumes"]}
    assert volumes["ct"]["dtype"] == "int16"
    assert volumes["mr"]["dtype"] == "uint16"
    ct = np.frombuffer(blobs["ct"], dtype=np.dtype("<i2")).reshape(nz, ny, nx)
    mr = np.frombuffer(blobs["mr"], dtype=np.dtype("<u2")).reshape(nz, ny, nx)
    for blob_id in ("ct", "mr"):
        assert len(blobs[blob_id]) == nx * ny * nz * 2

    ix, iy, iz = _voxel_at_ras(affine, (10.0, 0.0, 0.0))
    assert int(ct[iz, iy, ix]) == 60
    air_ix, air_iy, air_iz = _voxel_at_ras(affine, (70.0, 70.0, 70.0))
    assert int(ct[air_iz, air_iy, air_ix]) == -1000

    mr_meta = volumes["mr"]
    mr_center = int(mr[iz, iy, ix]) * mr_meta["slope"] + mr_meta["intercept"]
    assert mr_meta["slope"] != 1.0 and mr_meta["intercept"] != 0.0
    assert math.isclose(mr_center, 200.0, abs_tol=0.01)

    masks = {mask["id"]: mask for mask in manifest["masks"]}
    lesion = np.frombuffer(blobs[masks["lesion"]["blob"]], dtype=np.uint8).reshape(nz, ny, nx)
    unreviewed = np.frombuffer(blobs[masks["unreviewed"]["blob"]], dtype=np.uint8).reshape(nz, ny, nx)
    assert masks["lesion"]["reviewed"] is True
    assert masks["unreviewed"]["reviewed"] is False
    assert np.count_nonzero(lesion) == round(masks["lesion"]["volume_ml"] * 1000)
    expected_ml = (4.0 / 3.0) * math.pi * 10.0**3 / 1000.0
    actual_ml = np.count_nonzero(lesion) / 1000.0
    assert abs(actual_ml - expected_ml) / expected_ml < 0.10
    assert np.count_nonzero(unreviewed) > 0

    # Convert the marker voxel centroid through affine_ras; the current SPEC
    # defines patient left as negative x in RAS+.
    marker_voxels = np.flatnonzero(ct == 2000)
    marker_i = marker_voxels % nx
    marker_ras_x = marker_i + affine[0][3]
    assert float(marker_ras_x.mean()) < 0
    assert abs(float(marker_ras_x.mean()) + 35.0) < 0.1

    # Byte ordering contract: C-order [z, y, x] makes x the fastest index.
    assert ct[iz, iy, ix] == 60
    assert ct[iz, iy, ix + 1] == 60
    assert ct[iz, iy + 1, ix] == 60


def test_fixture_contains_two_pt_br_tour_steps_and_no_external_urls(capsule_file: Path) -> None:
    manifest, _ = parse_capsule(capsule_file)
    tour = manifest["tour"]
    assert len(tour) == 2
    assert all(step["view"]["window"]["volume"] == "ct" for step in tour)
    assert tour[0]["view"]["crosshair_ras"] == [10.0, 0.0, 0.0]
    assert tour[1]["view"]["crosshair_ras"] == [-35.0, 0.0, 0.0]
    assert "unreviewed" in tour[0]["view"]["visible_masks"]
    assert all(step["text"] for step in tour)

    html = capsule_file.read_text(encoding="utf-8")
    without_comments = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    without_csp = re.sub(
        r'<meta\s+http-equiv="Content-Security-Policy"[^>]*>',
        "",
        without_comments,
        flags=re.IGNORECASE,
    )
    assert re.search(r"https?://", without_csp, flags=re.IGNORECASE) is None


def test_distributable_template_matches_its_source_and_csp(capsule_file: Path, repo_root: Path) -> None:
    from viewer.build_template import build

    template = (repo_root / "viewer" / "template.html").read_text(encoding="utf-8")
    assert build() == template
    assert template.count("<!--CAPSULE_PAYLOAD-->") == 1
    assert (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src data: blob:; connect-src data: blob:; worker-src blob:; font-src data:"
    ) in template

    # The saved phantom includes the template plus schema payload, so run the
    # same offline scan over the actual file used by the browser tests.
    fixture_html = capsule_file.read_text(encoding="utf-8")
    fixture_without_comments = re.sub(r"<!--.*?-->", "", fixture_html, flags=re.DOTALL)
    fixture_without_csp = re.sub(
        r'<meta\s+http-equiv="Content-Security-Policy"[^>]*>',
        "",
        fixture_without_comments,
        flags=re.IGNORECASE,
    )
    assert re.search(r"https?://", fixture_without_csp, flags=re.IGNORECASE) is None
