from __future__ import annotations

import base64
import contextlib
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from capsule import cli, masks
from capsule.ingest import scan_series
from capsule.pack import parse_capsule, read_capsule, write_capsule


ROOT = Path(__file__).resolve().parents[1]
VS = ROOT / "data" / "public" / "Vestibular-Schwannoma-SEG"
VS_DICOM = VS / "brain-mri"
RT_T1 = VS / "rtstruct" / "rtstruct_T1.dcm"


@pytest.fixture(scope="module")
def vs_capsules(tmp_path_factory):
    if not VS_DICOM.is_dir() or not RT_T1.is_file():
        pytest.skip("public Vestibular-Schwannoma-SEG data not present")
    work = tmp_path_factory.mktemp("export-vs")
    original = work / "original.capsule.html"
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main([
            "build", str(VS_DICOM), "--series", "2,3", "--label", "Export test",
            "--viewer", "v2", "--brain-mask", "none", "--anatomy", "none",
            "--rtstruct", str(RT_T1), "-o", str(original),
        ]) == 0
    manifest, arrays = read_capsule(original)
    tumour = next(mask for mask in manifest["masks"] if mask["id"] == "tumour")
    unsigned = work / "unsigned.capsule.html"
    write_capsule(ROOT / "viewer2" / "template.html", unsigned, manifest, arrays)

    signed_manifest = json.loads(json.dumps(manifest))
    signed_tumour = next(mask for mask in signed_manifest["masks"] if mask["id"] == "tumour")
    signed_tumour["reviewed"] = True
    signed_tumour["review"] = {
        "by": "Export test reviewer",
        "at_utc": "2026-09-29T12:00:00Z",
        "blob_sha256": tumour["blob_sha256"],
    }
    render_mask = next(mask for mask in signed_manifest["masks"] if mask["id"] == "head")
    render_mask["reviewed"] = True
    render_mask["review"] = {
        "by": "Export test reviewer",
        "at_utc": "2026-09-29T12:00:00Z",
        "blob_sha256": render_mask["blob_sha256"],
    }
    signed = work / "signed.capsule.html"
    write_capsule(ROOT / "viewer2" / "template.html", signed, signed_manifest, arrays)
    return {"work": work, "unsigned": unsigned, "signed": signed, "manifest": signed_manifest}


def _run_export(capsule: Path, out: Path, *extra: str) -> int:
    return cli.main([
        "export", "--capsule", str(capsule), "--dicom", str(VS_DICOM), "--out", str(out), *extra,
    ])


def _native_image(folder: Path) -> tuple[sitk.Image, list[pydicom.Dataset]]:
    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(str(folder))
    reader.SetFileNames(filenames)
    return reader.Execute(), [pydicom.dcmread(filename) for filename in filenames]


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    left, right = left.astype(bool), right.astype(bool)
    denominator = int(left.sum() + right.sum())
    return 2.0 * int(np.count_nonzero(left & right)) / denominator if denominator else 1.0


def test_shared_capsule_parser_decodes_manifest_and_blobs(tmp_path):
    template = tmp_path / "template.html"
    template.write_text("<!doctype html><!--CAPSULE_PAYLOAD-->", encoding="utf-8")
    expected_manifest = {"schema": "case-capsule/1", "grid": {"dims": [2, 2, 1]}}
    expected_blob = bytes(range(4))
    payload = base64.b64encode(gzip.compress(expected_blob)).decode("ascii")
    capsule = tmp_path / "saved.capsule.html"
    capsule.write_text(
        '<!doctype html><script id="capsule-manifest" type="application/json">'
        + json.dumps(expected_manifest)
        + '</script><script id="capsule-blob-mask" type="application/octet-stream" '
        + 'data-encoding="gzip+base64">'
        + payload
        + "</script>",
        encoding="utf-8",
    )

    manifest, blobs = parse_capsule(capsule)

    assert manifest == expected_manifest
    assert blobs == {"mask": expected_blob}


def test_export_command_is_registered(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    assert exit_info.value.code == 0
    assert "export" in capsys.readouterr().out


def test_export_requires_explicit_output_directory(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["export", "--capsule", "saved.capsule.html", "--dicom", "source"])

    assert exit_info.value.code == 2
    assert "--out" in capsys.readouterr().err


def test_unsigned_and_legacy_review_flags_are_not_signatures(vs_capsules, tmp_path, capsys):
    unsigned_out = tmp_path / "unsigned-out"
    with pytest.raises(SystemExit) as unsigned_exit:
        _run_export(vs_capsules["unsigned"], unsigned_out)
    unsigned_err = capsys.readouterr().err
    assert unsigned_exit.value.code == 1
    assert "review" in unsigned_err.lower()
    assert not unsigned_out.exists()

    manifest, arrays = read_capsule(vs_capsules["unsigned"])
    tumour = next(mask for mask in manifest["masks"] if mask["id"] == "tumour")
    tumour["reviewed"] = True
    legacy = tmp_path / "legacy-reviewed.capsule.html"
    write_capsule(ROOT / "viewer2" / "template.html", legacy, manifest, arrays)
    with pytest.raises(SystemExit) as legacy_exit:
        _run_export(legacy, tmp_path / "legacy-out")
    assert legacy_exit.value.code == 1
    assert "review" in capsys.readouterr().err.lower()


def test_include_unreviewed_marks_all_export_labels_and_series(vs_capsules, tmp_path, capsys):
    out = tmp_path / "unreviewed-out"
    assert _run_export(vs_capsules["unsigned"], out, "--include-unreviewed") == 0
    stdout = capsys.readouterr().out
    seg = pydicom.dcmread(out / "seg.dcm")
    rtstruct = pydicom.dcmread(out / "rtstruct.dcm")
    receipt = json.loads((out / "export.json").read_text(encoding="utf-8"))

    assert (out / "seg.dcm").is_file() and (out / "rtstruct.dcm").is_file()
    assert "NAO REVISADO" in str(seg.SegmentSequence[0].SegmentLabel)
    assert all("NAO REVISADO" in str(item.ROIName) for item in rtstruct.StructureSetROISequence)
    assert "NAO REVISADO" in str(seg.SeriesDescription)
    assert "NAO REVISADO" in str(rtstruct.SeriesDescription)
    assert len(receipt["masks"]) == 1 and receipt["masks"][0]["id"] == "tumour"
    assert "tumour" in stdout and "mL" in stdout


def test_render_masks_are_excluded_unless_named(vs_capsules, tmp_path, capsys):
    default_out = tmp_path / "default-selection"
    assert _run_export(vs_capsules["signed"], default_out) == 0
    capsys.readouterr()
    default_receipt = json.loads((default_out / "export.json").read_text(encoding="utf-8"))
    assert [mask["id"] for mask in default_receipt["masks"]] == ["tumour"]

    named_out = tmp_path / "named-render"
    assert _run_export(vs_capsules["signed"], named_out, "--masks", "head") == 0
    stdout = capsys.readouterr().out
    named_receipt = json.loads((named_out / "export.json").read_text(encoding="utf-8"))
    assert [mask["id"] for mask in named_receipt["masks"]] == ["head"]
    assert "head" in stdout


def test_nonempty_output_is_never_overwritten(vs_capsules, tmp_path, capsys):
    out = tmp_path / "occupied"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        _run_export(vs_capsules["signed"], out)

    assert exit_info.value.code == 1
    assert "non-empty" in capsys.readouterr().err.lower()
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in out.iterdir()) == ["keep.txt"]


def test_export_round_trip_identity_receipt_and_shift_control(vs_capsules, tmp_path, capsys):
    out = tmp_path / "round-trip"
    assert _run_export(vs_capsules["signed"], out) == 0
    stdout = capsys.readouterr().out
    receipt_path = out / "export.json"
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    seg_ds = pydicom.dcmread(out / "seg.dcm")
    rt_ds = pydicom.dcmread(out / "rtstruct.dcm")
    native_image, source_images = _native_image(VS_DICOM / "T1_fl3d")
    original = masks.rasterize(masks.read_rtstruct(RT_T1, "AN"), native_image)
    exported_rt = masks.rasterize(masks.read_rtstruct(out / "rtstruct.dcm"), native_image)
    from highdicom.seg import Segmentation

    segmentation = Segmentation.from_dataset(seg_ds)
    decoded = segmentation.get_pixels_by_source_instance(
        [str(image.SOPInstanceUID) for image in source_images],
        segment_numbers=[1], combine_segments=True, assert_missing_frames_are_empty=True,
    )
    exported_seg = np.squeeze(np.asarray(decoded)).astype(bool)
    if exported_seg.shape != original.shape and exported_seg.size == original.size:
        exported_seg = exported_seg.reshape(original.shape)
    assert exported_seg.shape == original.shape

    positive_control = _dice(original, original)
    rt_dice = _dice(original, exported_rt)
    seg_dice = _dice(original, exported_seg)
    shift_voxels = max(1, int(round(5.0 / native_image.GetSpacing()[0])))
    shifted = np.roll(exported_seg, shift_voxels, axis=2)
    negative_control = _dice(original, shifted)
    print(
        f"VS T1 round trip: known-positive Dice {positive_control:.4f}; "
        f"RTSTRUCT Dice {rt_dice:.4f}; SEG Dice {seg_dice:.4f}; "
        f"voxels original={int(original.sum())}, RTSTRUCT={int(exported_rt.sum())}, "
        f"SEG={int(exported_seg.sum())}; 5 mm-shifted SEG Dice {negative_control:.4f}"
    )
    assert positive_control == 1.0
    assert rt_dice >= 0.90 and seg_dice >= 0.90
    assert negative_control < seg_dice - 0.15

    source_uid_set = {str(image.SOPInstanceUID) for image in source_images}
    source_study = str(source_images[0].StudyInstanceUID)
    source_frame = str(source_images[0].FrameOfReferenceUID)
    assert str(seg_ds.StudyInstanceUID) == source_study == str(rt_ds.StudyInstanceUID)
    assert str(seg_ds.FrameOfReferenceUID) == source_frame == str(rt_ds.FrameOfReferenceUID)
    assert str(seg_ds.SOPInstanceUID) not in source_uid_set
    assert str(rt_ds.SOPInstanceUID) not in source_uid_set
    source_class = str(source_images[0].SOPClassUID)
    for dataset in (seg_ds, rt_ds):
        # Image references carry the source image SOP class; RTSTRUCT's RT Referenced Study item names the
        # study (Study Component Management class) and is not an image reference.
        pairs = []

        def walk(items):
            for item in items:
                if "ReferencedSOPInstanceUID" in item:
                    pairs.append((str(item.get("ReferencedSOPClassUID", "")), str(item.ReferencedSOPInstanceUID)))
                for element in item:
                    if element.VR == "SQ":
                        walk(element.value)

        walk([dataset])
        images = {uid for sop_class, uid in pairs if sop_class == source_class}
        others = [(sop_class, uid == source_study) for sop_class, uid in pairs if sop_class != source_class]
        assert images and images <= source_uid_set, len(images - source_uid_set)
        assert all(is_study for _, is_study in others), others

    assert receipt["target_series_number"] == 2
    assert receipt["n_referenced_instances"] == len(source_images)
    assert receipt["masks"][0]["reviewed_by"] == "Export test reviewer"
    assert receipt["masks"][0]["voxels_native"] == int(exported_seg.sum())
    assert receipt["masks"][0]["volume_ml_native"] > 0
    blocked_text = receipt_text + stdout
    for private_value in (
        str(VS_DICOM),
        source_study,
        source_frame,
        str(source_images[0].PatientName),
        str(source_images[0].PatientID),
        *source_uid_set,
    ):
        if private_value and private_value in blocked_text:
            raise AssertionError("export receipt or stdout contains source identity or path data")
    assert "SeriesInstanceUID" not in receipt_text and "SOPInstanceUID" not in receipt_text


def test_cross_series_export_tracks_registered_t2_geometry(vs_capsules, tmp_path, capsys):
    out = tmp_path / "cross-series"
    assert _run_export(vs_capsules["signed"], out, "--series", "3") == 0
    capsys.readouterr()
    t1, _ = _native_image(VS_DICOM / "T1_fl3d")
    t2, t2_sources = _native_image(VS_DICOM / "T2_tse3d")
    original_t1 = masks.rasterize(masks.read_rtstruct(RT_T1, "AN"), t1)
    grid = masks.grid_from_manifest(vs_capsules["manifest"])
    original_grid = masks.to_grid(original_t1, t1, grid, sitk.Transform(3, sitk.sitkIdentity))
    volume = next(item for item in vs_capsules["manifest"]["volumes"] if item["series"]["number"] == 3)
    moving_to_reference = np.asarray(volume["registration"]["moving_to_reference_ras"], dtype=float)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    target_to_grid = ras_to_lps @ moving_to_reference @ ras_to_lps
    target_to_grid_transform = sitk.AffineTransform(3)
    target_to_grid_transform.SetMatrix(target_to_grid[:3, :3].ravel().tolist())
    target_to_grid_transform.SetTranslation(target_to_grid[:3, 3].tolist())
    grid_mask = sitk.GetImageFromArray(original_grid.astype(np.uint8))
    grid_mask.CopyInformation(grid)
    expected_t2 = sitk.GetArrayFromImage(sitk.Resample(
        grid_mask, t2, target_to_grid_transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
    )).astype(bool)
    exported_rt = masks.rasterize(masks.read_rtstruct(out / "rtstruct.dcm"), t2)
    from highdicom.seg import Segmentation

    segmentation = Segmentation.from_dataset(pydicom.dcmread(out / "seg.dcm"))
    decoded = segmentation.get_pixels_by_source_instance(
        [str(image.SOPInstanceUID) for image in t2_sources],
        segment_numbers=[1], combine_segments=True, assert_missing_frames_are_empty=True,
    )
    exported_seg = np.squeeze(np.asarray(decoded)).astype(bool)
    if exported_seg.shape != expected_t2.shape and exported_seg.size == expected_t2.size:
        exported_seg = exported_seg.reshape(expected_t2.shape)
    rt_dice, seg_dice = _dice(expected_t2, exported_rt), _dice(expected_t2, exported_seg)
    print(
        f"VS T2 cross-series: RTSTRUCT Dice {rt_dice:.4f}; SEG Dice {seg_dice:.4f}; "
        f"voxels expected={int(expected_t2.sum())}, RTSTRUCT={int(exported_rt.sum())}, "
        f"SEG={int(exported_seg.sum())}"
    )
    assert rt_dice >= 0.85 and seg_dice >= 0.85
    seg_ds = pydicom.dcmread(out / "seg.dcm")
    assert {str(item.SeriesInstanceUID) for item in seg_ds.ReferencedSeriesSequence} == {
        str(t2_sources[0].SeriesInstanceUID)
    }
    assert str(seg_ds.StudyInstanceUID) == str(t2_sources[0].StudyInstanceUID)
    assert str(seg_ds.FrameOfReferenceUID) == str(t2_sources[0].FrameOfReferenceUID)


def test_wrong_spacing_source_series_is_rejected(vs_capsules, tmp_path, capsys):
    source = next(item for item in scan_series(VS_DICOM)[0] if item.number == 2)
    wrong_root = tmp_path / "wrong-spacing"
    wrong_root.mkdir()
    # A different acquisition of the same series number: every slice has another in-plane spacing.
    for index, original in enumerate(sorted(source.files)):
        dataset = pydicom.dcmread(original)
        dataset.PixelSpacing = [float(value) * 1.5 for value in dataset.PixelSpacing]
        dataset.save_as(wrong_root / f"{index:05d}.dcm")
    output = tmp_path / "wrong-spacing-out"
    with pytest.raises(SystemExit) as exit_info:
        cli.main([
            "export", "--capsule", str(vs_capsules["signed"]), "--dicom", str(wrong_root),
            "--out", str(output),
        ])
    assert exit_info.value.code == 1
    assert "spacing" in capsys.readouterr().err.lower()
    assert not output.exists()


def test_export_stages_source_slices_beside_out_not_in_system_temp(vs_capsules, tmp_path, monkeypatch, capsys):
    import tempfile as tempfile_module

    from capsule import export as export_module

    seen = []
    real = tempfile_module.TemporaryDirectory

    def recording(*args, **kwargs):
        seen.append(kwargs.get("dir"))
        return real(*args, **kwargs)

    monkeypatch.setattr(export_module.tempfile, "TemporaryDirectory", recording)
    out = tmp_path / "phi-root" / "export"
    assert _run_export(vs_capsules["signed"], out) == 0
    capsys.readouterr()
    assert seen and all(value is not None and Path(value) == out.parent for value in seen), seen
    assert sorted(item.name for item in out.parent.iterdir()) == ["export"]


def test_segment_codes_follow_mask_role():
    from capsule.export import _segment_codes

    lesion_category, lesion_type = _segment_codes({"id": "tumour"})
    assert (lesion_category.value, lesion_type.value) == ("49755003", "4147007")
    anatomy_category, anatomy_type = _segment_codes({"id": "nerve", "role": "structure"})
    assert (anatomy_category.value, anatomy_type.value) == ("91723000", "91723000")
