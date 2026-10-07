import numpy as np
import pytest
import SimpleITK as sitk

from capsule import cli
from capsule.pack import read_capsule


def _ras_grid(shape=(64, 64, 64)):
    grid = sitk.Image(list(shape[::-1]), sitk.sitkFloat32)
    grid.SetSpacing((1, 1, 1))
    # Array j increases anteriorly in RAS; k increases superiorly.
    grid.SetDirection((1, 0, 0, 0, -1, 0, 0, 0, 1))
    return grid


def test_deface_removes_anterior_inferior_face_and_preserves_brain():
    from capsule.deface import apply_deface, face_removal_mask

    grid = _ras_grid()
    z, y, x = np.ogrid[:64, :64, :64]
    brain = ((x - 32) / 8) ** 2 + ((y - 32) / 8) ** 2 + ((z - 42) / 9) ** 2 <= 1
    nose = np.zeros(brain.shape, dtype=bool)
    nose[10:25, 52:62, 20:44] = True
    head = ((x - 32) / 22) ** 2 + ((y - 34) / 25) ** 2 + ((z - 32) / 27) ** 2 <= 1
    head |= nose

    cut = face_removal_mask(brain, grid, buffer_mm=5.0)
    assert np.all(cut[nose]), f"nose removal={cut[nose].mean():.1%}"
    assert not np.any(cut[brain])
    assert not cut[20, 10, 32]  # posterior to brain
    assert not cut[60, 40, 32]  # superior to brain

    mr = np.full(brain.shape, 20, dtype=np.float32)
    mr[nose] = 300
    ct = np.zeros(brain.shape, dtype=np.float32)
    ct[nose] = 200
    mr_after = apply_deface(mr, cut, "MR")
    ct_after = apply_deface(ct, cut, "CT")
    np.testing.assert_array_equal(mr_after[brain], mr[brain])
    np.testing.assert_array_equal(ct_after[brain], ct[brain])
    assert np.all(mr_after[nose] == mr.min())
    assert np.all(ct_after[nose] == -1000)
    assert not np.any((head & ~cut)[nose])
    assert mr_after[20, 10, 32] == mr[20, 10, 32]
    assert mr_after[60, 40, 32] == mr[60, 40, 32]


def test_build_nifti_deface_without_brain_mask_fails_clearly(tmp_path, capsys):
    image = np.full((24, 24, 24), -1000, dtype=np.int16)
    z, y, x = np.ogrid[:24, :24, :24]
    image[((x - 12) / 8) ** 2 + ((y - 12) / 8) ** 2 + ((z - 12) / 9) ** 2 <= 1] = 40
    volume = tmp_path / "ct.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(image), str(volume))
    output = tmp_path / "capsule.html"

    with pytest.raises(SystemExit) as error:
        cli.main(["build-nifti", "--volume", f"{volume}:CT:synthetic", "--label", "synthetic",
                  "-o", str(output), "--viewer", "v1", "--deface"])

    assert error.value.code == 1
    assert "--deface requires an available brain mask" in capsys.readouterr().err


def test_build_deface_changes_volume_and_render_head(tmp_path, monkeypatch):
    grid = _ras_grid()
    z, y, x = np.ogrid[:64, :64, :64]
    brain = ((x - 32) / 8) ** 2 + ((y - 32) / 8) ** 2 + ((z - 42) / 9) ** 2 <= 1
    nose = np.zeros(brain.shape, dtype=bool)
    nose[10:25, 52:62, 20:44] = True
    values = np.full(brain.shape, 20, dtype=np.float32)
    values[brain] = 100
    values[nose] = 300
    image = sitk.GetImageFromArray(values)
    image.CopyInformation(grid)
    monkeypatch.setattr(cli.maskops, "brain_mask", lambda *_: (brain.copy(), {"method": "synthetic", "seconds": 0}))
    args = type("Args", (), {"viewer": "v2", "tract": None, "max_streamlines": 10,
                             "tract_step_mm": 1, "label": "synthetic", "brain_mask": "bet",
                             "anatomy": "none", "deface": True})()
    output = tmp_path / "defaced.html"

    manifest = cli._write(args, output, grid, 1.0,
                          [{"image": image, "transform": sitk.Transform(3, sitk.sitkIdentity), "kind": "MR",
                            "label": "MR", "registration": {"reference": True}, "series": {"number": None},
                            "lesions": []}], None)
    decoded_manifest, arrays = read_capsule(output)
    volume = decoded_manifest["volumes"][0]
    decoded = arrays[volume["blob"]].astype(np.float32) * volume["slope"] + volume["intercept"]
    head = arrays[next(m["blob"] for m in decoded_manifest["masks"] if m["id"] == "head")].astype(bool)

    assert manifest["case"]["anonymized"] is True
    assert manifest["case"]["deidentification"] == {
        "identifiers_removed": True, "face_removed": True, "method": "quickshear-hull-buffer-5mm"}
    assert np.all(decoded[nose] == values.min())
    assert np.all(decoded[brain] == values[brain])
    assert not np.any(head[nose]) and np.all(head[brain])
