"""MRtrix .tck reader/writer and deterministic subsampling."""

from __future__ import annotations

import numpy as np
import pytest

from capsule.tracts import decode_tck, encode_tck, read_tck, subsample, write_tck


def _streamlines(seed=0, n=5):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(int(rng.integers(2, 9)), 3)).astype(np.float32) * 50 for _ in range(n)]


def _raw_tck(streamlines, dtype="<f4", name="Float32LE", extra=b"", terminate=True):
    body = []
    for streamline in streamlines:
        body.extend([np.asarray(streamline, dtype=dtype), np.full((1, 3), np.nan, dtype=dtype)])
    if terminate:
        body.append(np.full((1, 3), np.inf, dtype=dtype))
    payload = np.concatenate(body).astype(dtype).tobytes()  # concatenate returns native byte order
    header = b"mrtrix tracks\n" + extra + f"datatype: {name}\ncount: {len(streamlines)}\n".encode()
    offset = len(header) + len(b"file: . 0000\nEND\n")
    header += f"file: . {offset:04d}\nEND\n".encode()
    assert len(header) == offset
    return header + payload


def test_round_trip_preserves_points_and_structure(tmp_path):
    source = _streamlines()
    path = tmp_path / "t.tck"
    write_tck(path, source)
    decoded = read_tck(path)
    assert len(decoded) == len(source)
    for a, b in zip(decoded, source):
        np.testing.assert_array_equal(a, b)
    body = path.read_bytes()
    header, _, _ = body.partition(b"END\n")
    offset = int(header.split(b"file: . ")[1].split(b"\n")[0])
    points = np.frombuffer(body[offset:], dtype="<f4").reshape(-1, 3)
    nan_rows = np.flatnonzero(np.isnan(points).all(axis=1))
    assert len(nan_rows) == len(source)                       # one NaN separator per streamline
    assert np.isinf(points[-1]).all() and np.isinf(points).any(axis=1).sum() == 1  # single final Inf
    assert header.decode().splitlines() == ["mrtrix tracks", "datatype: Float32LE", f"count: {len(source)}",
                                            f"file: . {offset}"]
    with pytest.raises(FileExistsError):
        write_tck(path, source)


def test_big_endian_and_extra_header_lines_are_read_but_never_written():
    source = _streamlines(seed=4)
    canary = b"CANARY-TCK-HEADER-9f2c"
    raw = _raw_tck(source, dtype=">f4", name="Float32BE",
                   extra=b"command_history: tckgen /private/" + canary + b"/x.mif\nroi: " + canary + b"\n")
    decoded = decode_tck(raw)
    assert len(decoded) == len(source)
    for a, b in zip(decoded, source):
        np.testing.assert_array_equal(a, b)
    rewritten = encode_tck(decoded)
    assert canary not in rewritten and b"command_history" not in rewritten
    for a, b in zip(decode_tck(rewritten), source):
        np.testing.assert_array_equal(a, b)


def test_data_after_inf_and_missing_terminator():
    source = _streamlines(seed=7, n=3)
    trailing = _raw_tck(source) + np.full((4, 3), 99.0, dtype="<f4").tobytes()
    assert len(decode_tck(trailing)) == 3                     # points after the Inf triplet are ignored
    unterminated = _raw_tck(source, terminate=False)
    assert [len(s) for s in decode_tck(unterminated)] == [len(s) for s in source]


@pytest.mark.parametrize("bad", [b"not a tck\nEND\n", b"mrtrix tracks\ndatatype: Float64LE\nfile: . 60\nEND\n",
                                 b"mrtrix tracks\ndatatype: Float32LE\nfile: other.dat 0\nEND\n"])
def test_rejects_malformed(bad):
    with pytest.raises(ValueError):
        decode_tck(bad)


def test_subsample_count_and_determinism():
    source = _streamlines(seed=1, n=40)
    first = subsample(source, 10, seed=3)
    again = subsample(source, 10, seed=3)
    other = subsample(source, 10, seed=4)
    assert len(first) == 10
    assert all(a is b for a, b in zip(first, again))
    assert [id(a) for a in first] != [id(a) for a in other]
    positions = [next(i for i, s in enumerate(source) if s is a) for a in first]
    assert positions == sorted(positions) and len(set(positions)) == 10
    assert len(subsample(source, 100, seed=3)) == 40
    with pytest.raises(ValueError):
        subsample(source, 0)


def _helix(step=0.25, turns=3, radius=10.0, pitch=20.0):
    length = turns * np.hypot(2 * np.pi * radius, pitch)
    t = np.linspace(0, turns * 2 * np.pi, int(length / step) + 1)
    return np.stack([radius * np.cos(t), radius * np.sin(t), pitch * t / (2 * np.pi)], axis=1).astype(np.float32)


def test_decimate_keeps_endpoints_spacing_and_source_points():
    from capsule.tracts import decimate, decimate_indices, max_deviation_mm

    source = _helix()
    kept = decimate_indices(source, 1.0)
    out = decimate(source, 1.0)
    assert kept[0] == 0 and kept[-1] == len(source) - 1
    np.testing.assert_array_equal(out[0], source[0])
    np.testing.assert_array_equal(out[-1], source[-1])
    np.testing.assert_array_equal(out, source[kept])            # selected, never interpolated
    gaps = np.linalg.norm(np.diff(out[:-1].astype(float), axis=0), axis=1)
    assert 0.7 < gaps.min() and gaps.max() < 1.3               # ~1 mm (+/- one source step)
    assert len(out) < len(source) / 3
    deviation = max_deviation_mm(source, kept)
    assert 0 < deviation < 0.05                                  # chord sag on a 10 mm-radius helix
    assert max_deviation_mm(source, np.arange(len(source))) < 1e-9
    # Control: an absurd step must show a large deviation, proving the metric can fail.
    assert max_deviation_mm(source, decimate_indices(source, 40.0)) > 2.0
    np.testing.assert_array_equal(decimate(source, 0), source)
    two = source[:2]
    np.testing.assert_array_equal(decimate(two, 1.0), two)


def test_default_tract_colors_are_distinct_and_pair_by_hemisphere() -> None:
    from capsule.cli import _tint, default_tract_colors

    labels = [f"{b}_{h}" for b in "cst af slf1 slf2 slf3 ifo ilf uf fa or".split() for h in "lr"]
    colors = default_tract_colors(labels)
    assert len(set(colors)) == len(labels) == 20  # the old palette cycled every 6, so cst_l and slf2_l matched
    for i in range(0, 20, 2):
        assert colors[i + 1] == _tint(colors[i]), labels[i]  # right = lighter tint of the left hue
    many = default_tract_colors([f"bundle{i}_l" for i in range(40)])
    assert len(set(many)) == 40
    assert len(set(default_tract_colors(["a", "a", "b"]))) == 3


def test_capsule_dwi_provenance_reads_only_allowlisted_sidecar_values(tmp_path):
    import json

    from capsule.cli import _parameter_provenance

    path = tmp_path / "fx_cst_l.tck"
    write_tck(path, _streamlines(n=1))
    (tmp_path / "capsule-dwi-provenance.json").write_text(json.dumps({
        "pipeline": "capsule dwi",
        "profile": "fast",
        "seeds": 750_000,
        "algorithm": "iFOD2",
        "cutoff": 0.06,
        "source_path": "/private/example.nii.gz",
    }))

    provenance = _parameter_provenance(path.read_bytes(), path)

    assert provenance == {
        "recorded": True,
        "pipeline": "capsule dwi",
        "profile": "fast",
        "seeds": 750_000,
        "algorithm": "iFOD2",
        "cutoff": 0.06,
    }
