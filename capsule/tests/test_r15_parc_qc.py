from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from capsule import anatomy as anatomyops


def run_fake_synthseg(tmp_path, monkeypatch, failing_count, csv_mode="valid"):
    specs = tuple(
        anatomyops.LabelSpec((2001 + index,), f"parcel_{letter}_left", letter, "synthetic", "#FFFFFF")
        for index, letter in enumerate("abcd")
    )
    monkeypatch.setattr(anatomyops, "_freesurfer_home", lambda: tmp_path)
    monkeypatch.setattr(anatomyops, "synthseg_dk_label_specs", lambda home: specs)
    monkeypatch.setattr(
        anatomyops, "_synthseg_command",
        lambda source, destination, home: (["mri_synthseg", "--i", str(source), "--o",
                                             str(destination), "--vol", str(destination.with_name("volumes.csv"))], {}))
    hard = np.array([[[2001 + index if index >= failing_count else 0 for index in range(4)]]], dtype=np.uint16)
    hard_image = sitk.GetImageFromArray(hard)
    image = sitk.Image([4, 1, 1], sitk.sitkFloat32)
    hard_image.CopyInformation(image)

    def fake_run(command, **kwargs):
        sitk.WriteImage(hard_image, command[command.index("--o") + 1])
        vol_path = command[command.index("--vol") + 1]
        if csv_mode == "valid":
            columns = [f"ctx-lh-parcel-{letter}" for letter in "abcd"]
            # Leading header whitespace exercises real-world CSV formatting.
            content = "subject, " + ", ".join(columns) + "\nsynthetic, " + ", ".join(["1"] * 4) + "\n"
            open(vol_path, "w", encoding="utf-8").write(content)
        elif csv_mode == "malformed":
            open(vol_path, "w", encoding="utf-8").write("subject, ctx-lh-parcel-a\nsynthetic, nope\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(anatomyops.subprocess, "run", fake_run)
    return anatomyops.run_synthseg(image, image, sitk.Transform(3, sitk.sitkIdentity), "synthetic-mr")[0]


def test_three_bad_parcels_fail_and_report_lut_keys(tmp_path, monkeypatch):
    _, gyri = run_fake_synthseg(tmp_path, monkeypatch, 3)

    assert gyri["qc"] == {
        "method": "synthseg hard-vs-soft parcel agreement",
        "ratio_bounds": [0.5, 2.0],
        "max_failing": 2,
        "n_failing": 3,
        "failing": ["parcel_a_left", "parcel_b_left", "parcel_c_left"],
        "verdict": "FAIL",
        "calibration": "pilot: 1 template + 2 patient scans",
    }
    assert [label["key"] for label in gyri["labels"]] == ["parcel_d_left"]
    assert gyri["labels"][0]["soft_volume_ml"] == 0.001


def test_two_bad_parcels_pass_and_whitespace_headers_parse(tmp_path, monkeypatch):
    _, gyri = run_fake_synthseg(tmp_path, monkeypatch, 2)

    assert gyri["qc"]["verdict"] == "PASS"
    assert gyri["qc"]["n_failing"] == 2
    assert gyri["qc"]["failing"] == ["parcel_a_left", "parcel_b_left"]
    assert all("soft_volume_ml" in label for label in gyri["labels"])
    assert [label["soft_volume_ml"] for label in gyri["labels"]] == [0.001, 0.001]


def test_missing_or_unparseable_volume_csv_is_unavailable(tmp_path, monkeypatch):
    for mode in ("missing", "malformed"):
        monkeypatch.undo()
        _, gyri = run_fake_synthseg(tmp_path, monkeypatch, 0, mode)
        assert gyri["qc"]["verdict"] == "UNAVAILABLE"
        assert gyri["qc"]["n_failing"] is None
        assert gyri["qc"]["failing"] == []
        assert gyri["labels"][0]["soft_volume_ml"] is None
