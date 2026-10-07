"""build-nifti: synthetic NIfTI + synthetic tract -> capsule; tracts, de-id, viewer selection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from capsule import cli
from capsule.ingest import Series
from capsule.pack import read_capsule, read_capsule_tracts
from capsule.tracts import encode_tck, read_tck


CANARIES = {"descrip": b"CANARY-DESCRIP-7a1", "aux_file": b"CANARY-AUX-3b2", "intent_name": b"CANARY-INT-5c"}
TCK_CANARY = b"CANARY-TCKHDR-4d9e"


def _affine():
    """LAS voxel axes with a small oblique tilt and 2 mm voxels: exercises flips and rotation."""
    angle = np.radians(8)
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-2.0, 2.0, 2.0])
    affine[:3, 3] = [60.0, -70.0, -50.0]
    return affine


def _rod_line():
    """World RAS points along a straight rod inside the head (patient left, -x)."""
    t = np.linspace(-20, 20, 41)
    return np.stack([np.full_like(t, -20.0), 5.0 + 0.0 * t, t], axis=1)


def _write_inputs(root: Path):
    shape = (60, 70, 50)
    affine = _affine()
    ijk = np.indices(shape).reshape(3, -1).T.astype(float)
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    head = np.linalg.norm(world / [50.0, 60.0, 45.0], axis=1) <= 1
    data = np.where(head, 100.0, 0.0)
    # Bright rod: voxels within 3 mm of the line x=-20, y=5.
    rod = (np.hypot(world[:, 0] + 20.0, world[:, 1] - 5.0) <= 3.0) & (np.abs(world[:, 2]) <= 22)
    data[rod] = 1000.0
    image = nib.Nifti1Image(data.reshape(shape).astype(np.float32), affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    for field, value in CANARIES.items():
        image.header[field] = value
    nifti = root / "t1.nii.gz"
    nib.save(image, nifti)
    assert CANARIES["descrip"] in nib.load(nifti).header["descrip"].tobytes()

    line = _rod_line()
    streamlines = [(line + offset).astype(np.float32) for offset in ([0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [-0.5, -0.5, 0])]
    raw = encode_tck(streamlines)
    body = raw[raw.index(b"END\n") + 4:]
    extra = b"command_history: tckgen /Users/" + TCK_CANARY + b"/dwi.mif\nsource: " + TCK_CANARY + b"\n"
    header = b"mrtrix tracks\n" + extra + b"datatype: Float32LE\ncount: 4\nfile: . 0000\nEND\n"
    header = header.replace(b"0000", f"{len(header):04d}".encode())
    tck_bytes = header + body
    tck = root / "rod.tck"
    tck.write_bytes(tck_bytes)
    assert TCK_CANARY in tck.read_bytes()
    for a, b in zip(read_tck(tck), streamlines):
        np.testing.assert_array_equal(a, b)
    return nifti, tck, streamlines


def _run(*args):
    return subprocess.run([sys.executable, "-m", "capsule.cli", *map(str, args)], text=True, capture_output=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("nifti")
    nifti, tck, streamlines = _write_inputs(root)
    output = root / "nifti.capsule.html"
    run = _run("build-nifti", "--volume", f"{nifti}:MR:RM T1", "--tract", f"{tck}:Haste teste:#12ab34",
               "--max-streamlines", "3", "--label", "Caso sintético", "-o", output, "--viewer", "v1")
    assert run.returncode == 0, run.stderr
    return nifti, tck, streamlines, output, run.stdout


def test_manifest_fields(built):
    nifti, _, _, output, stdout = built
    manifest, arrays = read_capsule(output)
    assert manifest["case"] == {"label": "Caso sintético", "anonymized": False, "study_year": None,
                                "deidentification": {"identifiers_removed": True, "face_removed": False,
                                                     "method": None}}
    (volume,) = manifest["volumes"]
    assert volume["kind"] == "MR" and volume["label"] == "RM T1" and volume["units"] == "a.u."
    assert volume["registration"] == {"reference": True, "method": "none-shared-world"}
    (tract,) = manifest["tracts"]
    assert {key: tract[key] for key in ("id", "blob", "label", "color", "format", "n_streamlines",
                                        "n_streamlines_source", "source", "reviewed")} == {
        "id": "t01", "blob": "tract_t01", "label": "Haste teste", "color": "#12AB34", "format": "tck",
        "n_streamlines": 3, "n_streamlines_source": 4, "source": "tractlab", "reviewed": False}
    assert tract["fraction_points_inside_grid"] == 1.0
    # Synthetic streamlines are sampled every 1.0 mm, so the default 1.0 mm step keeps every point.
    assert tract["step_mm"] == 1.0 and tract["n_points"] == tract["n_points_before_step"] == 3 * 41
    assert set(arrays) == {"mr"}
    assert "3/4 streamlines" in stdout


def test_manifest_records_explicit_nifti_registration_method(built, tmp_path):
    nifti, _, _, _, _ = built
    output = tmp_path / "registered.capsule.html"
    run = _run("build-nifti", "--volume", f"{nifti}:MR:T1+C",
               "--volume", f"{nifti}:MR:FLAIR",
               "--volume-registration", "FLAIR:FLIRT dof6 normmi then rigid header transform",
               "--label", "Synthetic", "-o", output, "--viewer", "v1")

    assert run.returncode == 0, run.stderr
    manifest, _ = read_capsule(output)
    assert [volume["registration"] for volume in manifest["volumes"]] == [
        {"reference": True, "method": "none-shared-world"},
        {"reference": False, "method": "FLIRT dof6 normmi then rigid header transform"},
    ]


def test_tract_points_unchanged_and_consistent_with_image(built):
    nifti, _, streamlines, output, _ = built
    manifest, arrays = read_capsule(output)
    decoded = read_capsule_tracts(output)["t01"]
    assert len(decoded) == 3
    for streamline in decoded:
        assert any(np.array_equal(streamline, source) for source in streamlines)  # bit-exact, no shift
    affine = np.asarray(manifest["grid"]["affine_ras"])
    # The grid was cropped and reoriented relative to the source, so an applied shift would be visible.
    assert not np.allclose(affine, nib.load(nifti).affine)
    volume = manifest["volumes"][0]
    image = arrays["mr"].astype(float) * volume["slope"] + volume["intercept"]
    inverse = np.linalg.inv(affine)

    def mean_at(points):
        ijk = np.rint(points @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
        return float(np.mean([image[k, j, i] for i, j, k in ijk]))

    centre = decoded[0]
    on_rod, shifted = mean_at(centre), mean_at(centre + [10.0, 0, 0])
    print(f"mean intensity on tract={on_rod:.1f}; control shifted 10 mm={shifted:.1f}")
    assert on_rod > 900 and shifted < 200


def test_header_canaries_absent(built):
    _, _, _, output, stdout = built
    manifest, _ = read_capsule(output)
    raw = output.read_bytes()
    decoded_manifest = json.dumps(manifest, ensure_ascii=False).encode()
    tract_blob = __import__("capsule.pack", fromlist=["_read_payload"])._read_payload(output)[1]["tract_t01"]
    for token in [*CANARIES.values(), TCK_CANARY, b"command_history"]:
        assert token not in raw and token not in decoded_manifest and token not in tract_blob
        assert token.decode() not in stdout


def test_never_overwrite(built, tmp_path):
    nifti, _, _, output, _ = built
    before = output.read_bytes()
    run = _run("build-nifti", "--volume", f"{nifti}:MR:RM T1", "--label", "x", "-o", output)
    assert run.returncode != 0 and "never overwritten" in run.stderr
    assert output.read_bytes() == before


def test_viewer_v2_selection(built, tmp_path, monkeypatch, capsys):
    nifti, tck, _, _, _ = built
    root = tmp_path / "repo"
    monkeypatch.setattr(cli, "REPO_ROOT", root)
    missing = tmp_path / "missing.capsule.html"
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["build-nifti", "--volume", f"{nifti}:MR:RM T1", "--label", "x", "-o", str(missing), "--viewer", "v2"])
    assert exit_info.value.code == 1 and "viewer2/template.html not found" in capsys.readouterr().err
    assert not missing.exists()
    (root / "viewer2").mkdir(parents=True)
    (root / "viewer2" / "template.html").write_text("<!doctype html><title>FAKE-V2-MARKER</title><!--CAPSULE_PAYLOAD-->")
    output = tmp_path / "v2.capsule.html"
    assert cli.main(["build-nifti", "--volume", f"{nifti}:MR:RM T1", "--tract", f"{tck}:Haste",
                     "--label", "x", "-o", str(output), "--viewer", "v2"]) == 0
    text = output.read_text(encoding="utf-8")
    assert "FAKE-V2-MARKER" in text and 'id="capsule-blob-tract_t01"' in text
    manifest, _ = read_capsule(output)
    assert manifest["tracts"][0]["n_streamlines"] == 4


def test_spec_parsing():
    assert cli._parse_tract("/a:b/c.tck:FOFI E:#aabbcc") == (Path("/a:b/c.tck"), "FOFI E", "#AABBCC")
    assert cli._parse_tract("/x/c.tck:Cíngulo D") == (Path("/x/c.tck"), "Cíngulo D", None)
    assert cli._parse_volume("/x/t1.nii.gz:mr:RM T1") == (Path("/x/t1.nii.gz"), "MR", "RM T1")
    for bad in ["/x/c.tck", "/x/c.tck:#aabbcc"]:
        with pytest.raises(ValueError):
            cli._parse_tract(bad)
    with pytest.raises(ValueError):
        cli._parse_volume("/x/t1.nii.gz:PET:x")


def _series(number, modality, description, weighting):
    return Series(uid=str(number), number=number, modality=modality, description=description,
                  scanning_sequence="", sequence_name="", contrast=False, rows=1, columns=1,
                  pixel_spacing=(1, 1), slice_thickness=1, transfer_syntax="", year=None, files=[],
                  weighting=weighting)


def test_dicom_labels_are_unique_and_never_free_text():
    from capsule.deid import mr_weighting

    assert mr_weighting("t1_fl3d_tra_gk_v1") == "T1" and mr_weighting("t2_tse3d_tra_v2_384") == "T2"
    assert mr_weighting("Dr SMITH t2_flair") == "FLAIR" and mr_weighting("CANARY-NAME") is None
    labels = cli._dicom_labels([_series(2, "MR", "(omitted)", "T1"), _series(3, "MR", "(omitted)", "T2")])
    assert labels == ["RM T1", "RM T2"]
    labels = cli._dicom_labels([_series(2, "MR", "(omitted)", None), _series(3, "MR", "(omitted)", None),
                                _series(4, "CT", "(omitted)", None)])
    assert labels == ["RM (série 2)", "RM (série 3)", "TC"]


def test_dicom_build_accepts_tracts(built, tmp_path):
    from capsule.phantom import write_phantom

    _, tck, streamlines, _, _ = built
    write_phantom(tmp_path / "dicom", seed=2203)
    output = tmp_path / "dicom.capsule.html"
    run = _run("build", tmp_path / "dicom", "--series", "1", "--label", "Caso", "-o", output,
               "--tract", f"{tck}:Haste:#E4572E", "--max-streamlines", "2")
    assert run.returncode == 0, run.stderr
    manifest, _ = read_capsule(output)
    assert manifest["volumes"][0]["label"] == "Synthetic head CT"
    assert [t["n_streamlines"] for t in manifest["tracts"]] == [2]
    for streamline in read_capsule_tracts(output)["t01"]:
        assert any(np.array_equal(streamline, source) for source in streamlines)
    assert TCK_CANARY not in output.read_bytes()
