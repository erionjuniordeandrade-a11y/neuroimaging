"""Boolean ROI algebra — SEED / AND / OR / NOT → MRtrix include/exclude masks.

Semantics (locked from PLAN-sol / PRIOR-grok-luna-SYNTHESIS):
  SEED → union of seed strokes → one -seed_image; empty = fail loud
  AND  → each region kept separate → one -include per region
  OR   → union of OR strokes → one additional -include (omit if empty)
  NOT  → union of NOT strokes → one -exclude (omit if empty)
"""

from __future__ import annotations

import numpy as np
import pytest

from tractlab.grid import Grid
from tractlab.roi_boolean import (
    Stroke,
    RoiLayers,
    RoiCompileError,
    compile_roi_layers,
    build_tckgen_roi_args,
)


def _tiny_grid(shape=(20, 20, 20), spacing=1.0) -> Grid:
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = spacing
    return Grid(shape=shape, affine=aff, axcodes=("R", "A", "S"))


def _stroke_at(mm, radius=1.5) -> Stroke:
    return Stroke(points_mm=np.asarray(mm, dtype=np.float64), radius_mm=radius)


def test_empty_seed_is_compile_error():
    g = _tiny_grid()
    with pytest.raises(RoiCompileError, match="seed"):
        compile_roi_layers(g, RoiLayers(seed=[], and_regions=[], or_regions=[], not_regions=[]))


def test_seed_union_rasterizes_all_points():
    g = _tiny_grid()
    layers = RoiLayers(
        seed=[_stroke_at([[5.0, 5.0, 5.0]]), _stroke_at([[15.0, 15.0, 15.0]])],
        and_regions=[],
        or_regions=[],
        not_regions=[],
    )
    c = compile_roi_layers(g, layers)
    assert c.seed_mask.sum() > 0
    # both neighbourhoods present
    assert c.seed_mask[5, 5, 5]
    assert c.seed_mask[15, 15, 15]
    assert c.and_masks == []
    assert c.or_mask is None
    assert c.not_mask is None


def test_and_regions_stay_separate():
    g = _tiny_grid()
    layers = RoiLayers(
        seed=[_stroke_at([[5.0, 5.0, 5.0]])],
        and_regions=[
            [_stroke_at([[8.0, 5.0, 5.0]])],
            [_stroke_at([[12.0, 5.0, 5.0]])],
        ],
        or_regions=[],
        not_regions=[],
    )
    c = compile_roi_layers(g, layers)
    assert len(c.and_masks) == 2
    # disjoint centres → masks not identical and both non-empty
    assert c.and_masks[0].sum() > 0 and c.and_masks[1].sum() > 0
    assert not np.array_equal(c.and_masks[0], c.and_masks[1])


def test_empty_and_region_is_skipped_not_silent_include():
    """An empty AND region must not become an all-zero -include (would kill every tract)."""
    g = _tiny_grid()
    layers = RoiLayers(
        seed=[_stroke_at([[5.0, 5.0, 5.0]])],
        and_regions=[[], [_stroke_at([[10.0, 5.0, 5.0]])]],
        or_regions=[],
        not_regions=[],
    )
    c = compile_roi_layers(g, layers)
    assert len(c.and_masks) == 1
    assert c.and_masks[0].sum() > 0


def test_or_is_union_single_mask():
    g = _tiny_grid()
    layers = RoiLayers(
        seed=[_stroke_at([[5.0, 5.0, 5.0]])],
        and_regions=[],
        or_regions=[_stroke_at([[8.0, 5.0, 5.0]]), _stroke_at([[12.0, 5.0, 5.0]])],
        not_regions=[],
    )
    c = compile_roi_layers(g, layers)
    assert c.or_mask is not None
    assert c.or_mask[8, 5, 5] and c.or_mask[12, 5, 5]


def test_not_is_union_exclude_mask():
    g = _tiny_grid()
    layers = RoiLayers(
        seed=[_stroke_at([[5.0, 5.0, 5.0]])],
        and_regions=[],
        or_regions=[],
        not_regions=[_stroke_at([[15.0, 15.0, 15.0]])],
    )
    c = compile_roi_layers(g, layers)
    assert c.not_mask is not None
    assert c.not_mask[15, 15, 15]


def test_build_tckgen_roi_args_mapping(tmp_path):
    """AND → repeated -include; OR → one -include; NOT → one -exclude."""
    g = _tiny_grid()
    layers = RoiLayers(
        seed=[_stroke_at([[5.0, 5.0, 5.0]])],
        and_regions=[[_stroke_at([[8.0, 5.0, 5.0]])], [_stroke_at([[12.0, 5.0, 5.0]])]],
        or_regions=[_stroke_at([[10.0, 10.0, 10.0]])],
        not_regions=[_stroke_at([[15.0, 15.0, 15.0]])],
    )
    c = compile_roi_layers(g, layers)
    # write dummy paths and build argv fragments
    paths = {
        "seed": str(tmp_path / "seed.nii.gz"),
        "and": [str(tmp_path / "and0.nii.gz"), str(tmp_path / "and1.nii.gz")],
        "or": str(tmp_path / "or.nii.gz"),
        "not": str(tmp_path / "not.nii.gz"),
    }
    # touch files so the builder can require existence
    for p in [paths["seed"], paths["or"], paths["not"], *paths["and"]]:
        open(p, "wb").close()
    argv = build_tckgen_roi_args(
        seed_path=paths["seed"],
        and_paths=paths["and"],
        or_path=paths["or"],
        not_path=paths["not"],
    )
    assert argv[0:2] == ["-seed_image", paths["seed"]]
    # two AND includes then one OR include
    includes = [argv[i + 1] for i, a in enumerate(argv) if a == "-include"]
    assert includes == paths["and"] + [paths["or"]]
    assert "-exclude" in argv
    assert argv[argv.index("-exclude") + 1] == paths["not"]


def test_build_tckgen_omits_empty_optional_roles(tmp_path):
    seed = str(tmp_path / "seed.nii.gz")
    open(seed, "wb").close()
    argv = build_tckgen_roi_args(seed_path=seed, and_paths=[], or_path=None, not_path=None)
    assert argv == ["-seed_image", seed]
    assert "-include" not in argv
    assert "-exclude" not in argv


def test_parse_request_layers_legacy_seed_only():
    from tractlab.roi_boolean import layers_from_request

    layers = layers_from_request(
        {
            "seed": {"points_mm": [[1.0, 2.0, 3.0]], "radius_mm": 4.0},
        }
    )
    assert len(layers.seed) == 1
    assert layers.and_regions == []
    assert layers.or_regions == []
    assert layers.not_regions == []


def test_parse_request_layers_full_boolean():
    from tractlab.roi_boolean import layers_from_request

    layers = layers_from_request(
        {
            "seed": {"points_mm": [[1.0, 2.0, 3.0]], "radius_mm": 4.0},
            "and": [
                {"points_mm": [[5.0, 5.0, 5.0]], "radius_mm": 3.0},
                {"points_mm": [[6.0, 6.0, 6.0]], "radius_mm": 3.0},
            ],
            "or": [{"points_mm": [[7.0, 7.0, 7.0]], "radius_mm": 2.0}],
            "not": [{"points_mm": [[8.0, 8.0, 8.0]], "radius_mm": 2.0}],
        }
    )
    assert len(layers.seed) == 1
    assert len(layers.and_regions) == 2
    assert len(layers.or_regions) == 1
    assert len(layers.not_regions) == 1


def test_parse_request_rejects_empty_seed_points():
    from tractlab.roi_boolean import layers_from_request

    with pytest.raises(RoiCompileError, match="seed"):
        layers_from_request({"seed": {"points_mm": [], "radius_mm": 4.0}})


def test_parse_request_rejects_nonfinite_points():
    from tractlab.roi_boolean import layers_from_request

    with pytest.raises(RoiCompileError, match="finite"):
        layers_from_request({
            "seed": {"points_mm": [[float("nan"), 0.0, 0.0]], "radius_mm": 4.0}
        })


def test_parse_request_rejects_excessive_region_count():
    from tractlab.roi_boolean import layers_from_request

    with pytest.raises(RoiCompileError, match="too many OR"):
        layers_from_request({
            "seed": {"points_mm": [[0.0, 0.0, 0.0]], "radius_mm": 4.0},
            "or": [
                {"points_mm": [[float(i), 0.0, 0.0]], "radius_mm": 2.0}
                for i in range(65)
            ],
        })
