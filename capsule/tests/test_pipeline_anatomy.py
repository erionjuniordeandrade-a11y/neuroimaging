"""Anatomy label remapping, geometry, volume, and capsule-payload contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from capsule import anatomy as anatomyops
from capsule.anatomy import (SYNTHSEG_LABELS, _error_text, _synthseg_command, encode_label_map, resample_labels,
                             run_totalsegmentator, task_label_specs, task_specs_from_info)
from capsule import cli
from capsule.pack import read_capsule, write_capsule
from scripts.anatomy_review import anatomy_report


def test_synthseg_curated_map_merges_ventricles_and_cerebellum_and_excludes_surface():
    source = np.array([[[0, 4, 5, 7, 8, 10, 2, 3, 41, 42]]], dtype=np.uint8)

    mapped, labels = encode_label_map(source, SYNTHSEG_LABELS, spacing_mm=(1.0, 2.0, 3.0))

    assert mapped.dtype == np.uint8
    assert mapped.shape == source.shape
    assert mapped[0, 0, 1] == mapped[0, 0, 2]  # lateral + inferior lateral ventricle
    assert mapped[0, 0, 3] == mapped[0, 0, 4]  # cerebellar white matter + cortex
    assert mapped[0, 0, 6:10].tolist() == [0, 0, 0, 0]  # cortex and white matter are excluded
    ventricle = next(item for item in labels if item["key"] == "lateral_ventricle_left")
    cerebellum = next(item for item in labels if item["key"] == "cerebellum_left")
    assert ventricle["volume_ml"] == pytest.approx(0.012)
    assert cerebellum["volume_ml"] == pytest.approx(0.012)
    assert ventricle["group"] == "Ventrículos"
    assert cerebellum["group"] == "Tronco e cerebelo"
    assert len({item["value"] for item in labels}) == len(labels)


def test_multilabel_resampling_uses_nearest_neighbor_and_series_geometry():
    source = sitk.GetImageFromArray(np.array([[[0, 9, 4]]], dtype=np.uint8))
    target = sitk.Image([3, 1, 1], sitk.sitkUInt8)
    target.SetOrigin((1.0, 0.0, 0.0))
    target.SetSpacing((1.0, 1.0, 1.0))

    result = resample_labels(source, target, sitk.Transform(3, sitk.sitkIdentity))

    assert result.dtype == np.uint8
    assert result.tolist() == [[[9, 4, 0]]]
    assert set(np.unique(result)) <= {0, 4, 9}  # no interpolated label values


def test_anatomy_blob_round_trips_as_uint8_on_capsule_grid(tmp_path):
    template = tmp_path / "template.html"
    template.write_text("<!doctype html><!--CAPSULE_PAYLOAD-->", encoding="utf-8")
    destination = tmp_path / "case.capsule.html"
    item = {
        "id": "anat_mr",
        "blob": "anatomy_anat_mr",
        "for_volume": "mr",
        "source": "auto",
        "method": "synthseg 2.0 (FreeSurfer 7.4.1, --robust)",
        "licence": "FreeSurfer Software License Agreement",
        "reviewed": False,
        "labels": [{"value": 1, "key": "brainstem", "name": "Tronco encefálico",
                     "group": "Tronco e cerebelo", "color": "#CF896C", "volume_ml": 0.001}],
    }
    manifest = {"grid": {"dims": [3, 2, 1]}, "volumes": [], "masks": [], "anatomy": [item], "tracts": []}
    labels = np.array([[[0, 1, 1], [0, 0, 0]]], dtype=np.uint8)

    write_capsule(template, destination, manifest, {item["blob"]: labels})
    decoded_manifest, arrays = read_capsule(destination)

    assert decoded_manifest["anatomy"] == [item]
    np.testing.assert_array_equal(arrays[item["blob"]], labels)


def test_task_labels_are_verified_against_the_installed_class_map():
    with pytest.raises(RuntimeError, match="class map lacks expected labels"):
        task_label_specs("craniofacial_structures", {"classes": {"1": "mandible"}})


def test_totalsegmentator_numeric_class_map_matches_installed_info_contract():
    names = ["mandible", "teeth_lower", "skull", "sinus_maxillary", "sinus_frontal", "teeth_upper"]
    info = {"tasks": {"craniofacial_structures": {
        "modality": "CT", "license_required": False,
        "classes": {str(value): name for value, name in enumerate(names, start=1)},
    }}}

    selected = task_specs_from_info(info)

    assert len(selected) == 1
    assert [spec.source_values for spec in selected[0][1]] == [(1,), (2,), (3,), (4,), (5,), (6,)]


def test_totalsegmentator_shared_memory_error_keeps_root_diagnostic():
    captured = ("traceback " * 300) + (
        "RuntimeError: unable to open shared memory object </torch_example> in read-write mode: "
        "Operation not permitted (1)\nRuntimeError: Background workers died"
    )

    assert _error_text(captured) == (
        "RuntimeError: unable to open shared memory object </torch_example> in read-write mode: "
        "Operation not permitted (1)"
    )


def test_totalsegmentator_218_uses_installed_cli_flags(tmp_path, monkeypatch):
    task_names = list(anatomyops._TASK_LABELS["head_glands_cavities"])
    task_info = {"classes": {str(value): name for value, name in enumerate(task_names, start=1)}}
    info = {"totalsegmentator_version": "2.18.0", "tasks": {"head_glands_cavities": task_info}}
    definitions = task_label_specs("head_glands_cavities", task_info)
    image = sitk.Image([2, 2, 2], sitk.sitkFloat32)
    source_labels = np.zeros((2, 2, 2), dtype=np.uint8)
    source_labels[0, 0, 0] = 1
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        output = sitk.GetImageFromArray(source_labels)
        output.CopyInformation(image)
        sitk.WriteImage(output, command[command.index("-o") + 1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(anatomyops.shutil, "which", lambda name: "/external/TotalSegmentator")
    monkeypatch.setattr(anatomyops.subprocess, "run", fake_run)

    item, packed, record = run_totalsegmentator(
        image, image, sitk.Transform(3, sitk.sitkIdentity), "ct", "head_glands_cavities", definitions, info)

    command = seen["command"]
    assert command[1] == "-i" and command[3] == "-o" and command[5] == "-ml"
    assert command[6:10] == ["-ta", "head_glands_cavities", "-d", "cpu"]
    assert command[10] == "-rp"
    assert "--input" not in command and "--output" not in command
    assert item["id"] == "anat_ct_head_glands_cavities"
    assert item["licence"].startswith("Apache-2.0")
    assert packed.dtype == np.uint8 and packed[0, 0, 0] == 1
    assert record["version"] == "2.18.0" and record["task"] == "head_glands_cavities"
    # Only class 1 is present in the fake output: the other structures are 0 mL and must not be listed.
    assert [label["key"] for label in item["labels"]] == [task_names[0]]


def test_arm_fallback_runs_freesurfer_script_with_legacy_keras_and_numpy_compatibility(tmp_path, monkeypatch):
    home = tmp_path / "freesurfer"
    (home / "bin").mkdir(parents=True)
    (home / "python" / "scripts").mkdir(parents=True)
    (home / "bin" / "mri_synthseg").touch()
    script = home / "python" / "scripts" / "mri_synthseg"
    script.touch()
    (home / "build-stamp.txt").write_text("freesurfer-macOS-darwin_x86_64-7.4.1-build", encoding="utf-8")
    monkeypatch.setattr(anatomyops.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(anatomyops.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    command, environment = _synthseg_command(tmp_path / "input.nii.gz", tmp_path / "output.nii.gz", home)

    assert "-c" in command
    assert "float128" in command[command.index("-c") + 1]
    assert str(script) in command
    assert command.count("--parc") == 1
    assert command[-11:] == ["--i", str(tmp_path / "input.nii.gz"), "--o", str(tmp_path / "output.nii.gz"),
                             "--vol", str(tmp_path / "volumes.csv"), "--robust", "--cpu", "--threads", "1", "--parc"]
    assert environment["FREESURFER_HOME"] == str(home)
    assert environment["TF_USE_LEGACY_KERAS"] == "1"
    assert "tf-keras==2.17.0" in command
    probe = tmp_path / "probe.py"
    probe.write_text("import sys, numpy as np; assert sys.argv[1:] == ['probe-arg']; "
                     "assert hasattr(np, 'float128'); print('bootstrap-ok')", encoding="utf-8")
    bootstrap = command[command.index("-c") + 1]
    bootstrap = bootstrap.replace("setattr(np, 'float128', getattr(np, 'float128', np.longdouble));",
                                  "(delattr(np, 'float128') if hasattr(np, 'float128') else None); "
                                  "setattr(np, 'float128', getattr(np, 'float128', np.longdouble));")
    result = subprocess.run([sys.executable, "-c", bootstrap, str(probe), "probe-arg"],
                            capture_output=True, text=True, env=environment, check=False)
    assert result.returncode == 0, result.stderr
    assert "bootstrap-ok" in result.stdout


def test_synthseg_dk_localization_covers_installed_parcellation_values():
    home = anatomyops._freesurfer_home()
    values = np.load(home / "models" / "synthseg_parcellation_labels.npy", allow_pickle=False)
    specs = anatomyops.synthseg_dk_label_specs(home)
    requested = {int(value) for value in values if value}
    lut_rgb = {}
    for line in (home / "FreeSurferColorLUT.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[0].isdigit() and int(fields[0]) in requested:
            lut_rgb[int(fields[0])] = "#" + "".join(f"{int(channel):02X}" for channel in fields[2:5])

    assert [spec.source_values[0] for spec in specs] == [int(value) for value in values if value]
    assert {spec.key for spec in specs} == set(anatomyops.DK_PARCEL_PTBR)
    assert all(spec.name and spec.group for spec in specs)
    assert {spec.group for spec in specs} <= {
        "Lobo frontal", "Lobo parietal", "Lobo temporal", "Lobo occipital", "Lobo límbico (cíngulo)", "Ínsula"}
    assert all(spec.color == lut_rgb[spec.source_values[0]] for spec in specs)
    assert all(spec.name.endswith(("esquerdo", "esquerda")) for spec in specs if spec.key.endswith("_left"))
    assert all(spec.name.endswith(("direito", "direita")) for spec in specs if spec.key.endswith("_right"))
    assert all(1000 <= spec.source_values[0] < 2000 for spec in specs if spec.key.endswith("_left"))
    assert all(2000 <= spec.source_values[0] < 3000 for spec in specs if spec.key.endswith("_right"))
    assert len(specs) == 68


def test_synthseg_dk_parcels_preserve_side_mapping_from_freesurfer_values():
    specs = {spec.key: spec for spec in anatomyops.synthseg_dk_label_specs(anatomyops._freesurfer_home())}
    left = specs["precentral_left"]
    right = specs["precentral_right"]

    assert left.name == "Giro pré-central esquerdo"
    assert right.name == "Giro pré-central direito"
    assert 1000 <= left.source_values[0] < 2000
    assert 2000 <= right.source_values[0] < 3000


def test_synthseg_runs_parc_once_and_emits_uint8_mr_and_dk_items(tmp_path, monkeypatch):
    home = anatomyops._freesurfer_home()
    specs = {spec.key: spec for spec in anatomyops.synthseg_dk_label_specs(home)}
    left_value = specs["precentral_left"].source_values[0]
    right_value = specs["superiortemporal_right"].source_values[0]
    absent_key = "cuneus_left"
    source_values = np.array([[[0, left_value, right_value, 10, 49, 4, 43]]], dtype=np.uint16)
    source_labels = sitk.GetImageFromArray(source_values)
    image = sitk.Image([7, 1, 1], sitk.sitkFloat32)
    source_labels.CopyInformation(image)
    seen_commands = []

    def fake_run(command, **kwargs):
        seen_commands.append(command)
        sitk.WriteImage(source_labels, command[command.index("--o") + 1])
        volume_path = Path(command[command.index("--vol") + 1])
        specs = anatomyops.synthseg_dk_label_specs(home)
        volume_path.write_text(",".join(["subject", *[
            f"ctx-{'lh' if spec.key.endswith('_left') else 'rh'}-{spec.key.rsplit('_', 1)[0].replace('_', '-')}"
            for spec in specs]]) + "\nsynthetic," + ",".join(["0"] * len(specs)), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(anatomyops.subprocess, "run", fake_run)

    items, arrays, record = anatomyops.run_synthseg(
        image, image, sitk.Transform(3, sitk.sitkIdentity), "mr-reference")

    assert len(seen_commands) == 1
    assert seen_commands[0].count("--parc") == 1
    assert record["command"].find("--parc") >= 0
    assert [item["id"] for item in items] == ["anat_mr", "anat_mr_gyri"]
    mr, gyri = items
    assert mr["blob"] == "anatomy_anat_mr"
    assert gyri["blob"] == "anatomy_anat_mr_gyri"
    assert "qc" not in mr
    for item in items:
        assert item["for_volume"] == "mr-reference"
        assert item["source"] == "auto"
        assert "--parc" in item["method"]
        assert item["licence"] == anatomyops._FREESURFER_LICENSE
        assert item["reviewed"] is False
        expected = {"id", "blob", "for_volume", "source", "method", "licence", "reviewed", "labels"}
        if item["id"] == "anat_mr_gyri":
            expected.add("qc")
        assert set(item) == expected

    no_parc_labels = source_values.copy()
    no_parc_labels[no_parc_labels == left_value] = 3
    no_parc_labels[no_parc_labels == right_value] = 42
    expected_mr, expected_mr_labels = encode_label_map(
        no_parc_labels, SYNTHSEG_LABELS, image.GetSpacing())
    np.testing.assert_array_equal(arrays[mr["blob"]], expected_mr)
    assert mr["labels"] == expected_mr_labels

    assert arrays[mr["blob"]].dtype == np.uint8
    assert arrays[gyri["blob"]].dtype == np.uint8
    assert [label["value"] for label in gyri["labels"]] == [1, 2]
    assert all(label["soft_volume_ml"] == 0 for label in gyri["labels"])
    assert gyri["qc"]["verdict"] == "PASS"
    assert {label["key"] for label in gyri["labels"]} == {"precentral_left", "superiortemporal_right"}
    assert "Giro pré-central esquerdo" == next(
        label["name"] for label in gyri["labels"] if label["key"] == "precentral_left")
    assert "Giro temporal superior direito" == next(
        label["name"] for label in gyri["labels"] if label["key"] == "superiortemporal_right")
    assert absent_key not in {label["key"] for label in gyri["labels"]}
    assert set(np.unique(arrays[gyri["blob"]])) == {0, 1, 2}


def test_dicom_build_defaults_to_v2_and_automatic_anatomy(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CAPSULE_ANATOMY", raising=False)
    output = tmp_path / "case.capsule.html"
    seen = {}

    def fake_build(args):
        seen["viewer"] = args.viewer
        seen["anatomy"] = args.anatomy
        output.write_text("capsule", encoding="utf-8")
        return output, {"volumes": [], "masks": [], "tracts": []}

    monkeypatch.setattr(cli, "build", fake_build)
    assert cli.main(["build", str(tmp_path / "dicom"), "--series", "1", "--label", "demo", "-o", str(output)]) == 0
    assert seen == {"viewer": "v2", "anatomy": "auto"}
    assert "0.00 MB" in capsys.readouterr().out


def test_disabling_anatomy_keeps_the_v1_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CAPSULE_ANATOMY", "none")
    output = tmp_path / "case.capsule.html"
    seen = {}

    def fake_build(args):
        seen["viewer"] = args.viewer
        seen["anatomy"] = args.anatomy
        output.write_text("capsule", encoding="utf-8")
        return output, {"volumes": [], "masks": [], "tracts": []}

    monkeypatch.setattr(cli, "build", fake_build)
    assert cli.main(["build", str(tmp_path / "dicom"), "--series", "1", "--label", "demo", "-o", str(output)]) == 0

    assert seen == {"viewer": "v2", "anatomy": "none"}
    capsys.readouterr()


def test_auto_anatomy_rejects_the_v1_viewer(tmp_path, capsys):
    output = tmp_path / "case.capsule.html"
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["build", str(tmp_path / "dicom"), "--series", "1", "--label", "demo", "-o", str(output),
                  "--viewer", "v1", "--anatomy", "auto"])
    assert exit_info.value.code == 1
    assert "--anatomy auto requires --viewer v2" in capsys.readouterr().err
    assert not output.exists()


def test_build_passes_generated_volume_ids_to_anatomy(tmp_path, monkeypatch):
    grid = sitk.Image([2, 2, 2], sitk.sitkFloat32)
    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
    captured = {}

    def fake_produce(volumes, target_grid):
        captured["id"] = volumes[0].get("id")
        blob = "anatomy_anat_mr"
        labels = np.ones((2, 2, 2), dtype=np.uint8)
        item = {"id": "anat_mr", "blob": blob, "labels": []}
        return [item], {blob: labels}, [], {"anat_mr": "synthetic anatomy"}

    monkeypatch.setattr(cli.anatomyops, "produce_anatomy", fake_produce)
    monkeypatch.setattr(cli, "_render_masks", lambda *args: None)
    volume = {"image": image, "transform": sitk.Transform(3, sitk.sitkIdentity), "kind": "MR",
              "label": "RM T1", "registration": {"reference": True}, "series": {"modality": "MR"}}
    args = cli.argparse.Namespace(viewer="v2", anatomy="auto", label="t", tract=None,
                                  max_streamlines=1500, tract_step_mm=1.0)

    output = tmp_path / "case.capsule.html"
    cli._write(args, output, grid, 1.0, [volume], None)
    manifest, arrays = read_capsule(output)

    assert captured["id"] == "mr"
    item = manifest["anatomy"][0]
    assert item["blob_sha256"] == hashlib.sha256(arrays[item["blob"]].tobytes(order="C")).hexdigest()


def test_review_report_checks_volume_pairs_laterality_and_alignment():
    item = {"id": "anat_mr", "blob": "anatomy_anat_mr", "for_volume": "mr", "method": "synthseg test",
            "licence": "FreeSurfer Software License Agreement",
            "labels": [
                {"value": 1, "key": "lateral_ventricle_left", "name": "E", "volume_ml": 0.001},
                {"value": 2, "key": "lateral_ventricle_right", "name": "D", "volume_ml": 0.001},
            ]}
    manifest = {
        "grid": {"dims": [3, 1, 1], "spacing_mm": [1, 1, 1],
                 "affine_ras": [[-1, 0, 0, 1], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        "anatomy": [item], "masks": [{"id": "brain", "blob": "mask_brain", "for_volume": "mr"}],
    }
    arrays = {"anatomy_anat_mr": np.array([[[2, 0, 1]]], dtype=np.uint8),
              "mask_brain": np.ones((1, 1, 3), dtype=np.uint8)}

    rows, checks = anatomy_report(manifest, arrays)

    assert [row["volume_check"] for row in rows] == ["PASS", "PASS"]
    assert checks["paired_structures"][0]["left_right_ratio"] == 1.0
    assert checks["paired_structures"][0]["status"] == "PASS"
    assert checks["lateral_ventricle_centroids"]["left"]["centroid_x_ras_mm"] == -1.0
    assert checks["lateral_ventricle_centroids"]["right"]["centroid_x_ras_mm"] == 1.0
    assert checks["synthseg_grid_alignment"]["fraction"] == 1.0


def test_ct_anatomy_is_bone_and_air_only():
    # Owner ruling 2026-09-28: CT supplies bone and air; soft tissue, nerves, vessels and ventricles come from MR.
    from capsule.anatomy import _TASK_LABELS

    groups = {label.group for labels in _TASK_LABELS.values() for label in labels.values()}
    assert groups == {"Osso", "Seios e cavidades"}, groups
    assert set(_TASK_LABELS) == {"craniofacial_structures", "headneck_bones_vessels", "head_glands_cavities"}
