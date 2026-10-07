import hashlib
import json
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from capsule import cli
from capsule.pack import read_capsule, read_capsule_tract_blobs
from capsule.tracts import decode_tck, encode_tck


def _bundle():
    line = np.stack([np.linspace(0, 30, 31), np.zeros(31), np.zeros(31)], axis=1).astype(np.float32)
    kept = [(line + [0, y, 0]).astype(np.float32) for y in np.linspace(-1, 1, 10)]
    short_strays = [np.array([[100, 0, 0], [101, 0, 0]], dtype=np.float32),
                    np.array([[-100, 0, 0], [-101, 0, 0]], dtype=np.float32)]
    return kept + short_strays, short_strays


def test_load_tracts_packs_dropped_streamlines_and_preserves_counts(tmp_path):
    source, short_strays = _bundle()
    path = tmp_path / "source.tck"
    path.write_bytes(encode_tck(source))

    (entry,), blobs = cli._load_tracts([f"{path}:synthetic"], max_streamlines=100, step_mm=2.0)

    assert entry["n_streamlines_outliers"] == len(short_strays)
    assert len(decode_tck(blobs[entry["blob"]].tobytes())) + len(
        decode_tck(blobs[entry["outlier_blob"]].tobytes())) == len(source)
    assert len(decode_tck(blobs[entry["outlier_blob"]].tobytes())) == len(short_strays)


def test_load_tracts_subsamples_outliers_at_the_kept_ratio(tmp_path):
    source, _ = _bundle()
    path = tmp_path / "source.tck"
    path.write_bytes(encode_tck(source))

    (entry,), blobs = cli._load_tracts([f"{path}:synthetic"], max_streamlines=5, step_mm=0)

    assert entry["n_streamlines"] == 5
    assert entry["n_streamlines_outliers"] == 2
    assert len(decode_tck(blobs[entry["outlier_blob"]].tobytes())) == 1


def test_build_manifest_hashes_decoded_mask_and_tract_and_sanitizes_provenance(tmp_path):
    source, _ = _bundle()
    private_stem = "private-subject-token"
    tck = tmp_path / f"{private_stem}.tck"
    tck.write_bytes(encode_tck(source[:1]))
    # A source header can contain a path-bearing tracking command. Only safe scalar settings may survive.
    clean = encode_tck(source[:1])
    body = clean[clean.index(b"END\n") + 4:]
    header = (b"mrtrix tracks\ncommand_history: tckgen -algorithm iFOD2 -select 1 "
              b"/private/subject/input.mif -seed_image /private/subject/mask.nii.gz\n"
              b"datatype: Float32LE\ncount: 1\nfile: . 0000\nEND\n")
    header = header.replace(b"0000", f"{len(header):04d}".encode())
    tck.write_bytes(header + body)

    grid = sitk.Image([40, 40, 40], sitk.sitkFloat32)
    values = np.zeros((40, 40, 40), dtype=np.float32)
    z, y, x = np.ogrid[:40, :40, :40]
    values[((x - 20) / 13) ** 2 + ((y - 20) / 14) ** 2 + ((z - 20) / 15) ** 2 <= 1] = 100
    image = sitk.GetImageFromArray(values)
    image.CopyInformation(grid)
    args = SimpleNamespace(viewer="v2", tract=[f"{tck}:synthetic"], max_streamlines=100,
                           tract_step_mm=0, label="synthetic", brain_mask="none", anatomy="none", deface=False)
    output = tmp_path / "capsule.html"

    manifest = cli._write(args, output, grid, 1.0,
                          [{"image": image, "transform": sitk.Transform(3, sitk.sitkIdentity), "kind": "MR",
                            "label": "MR", "registration": {"reference": True}, "series": {"number": None},
                            "lesions": []}], None)
    decoded_manifest, arrays = read_capsule(output)
    decoded_tracts = read_capsule_tract_blobs(output)
    mask = next(item for item in decoded_manifest["masks"] if item["id"] == "head")
    tract = decoded_manifest["tracts"][0]

    assert manifest == decoded_manifest
    assert mask["blob_sha256"] == hashlib.sha256(arrays[mask["blob"]].tobytes()).hexdigest()
    assert tract["blob_sha256"] == hashlib.sha256(decoded_tracts[tract["blob"]].tobytes()).hexdigest()
    assert tract["source_sha256"] == hashlib.sha256(tck.read_bytes()).hexdigest()
    provenance = json.dumps(tract["provenance"], sort_keys=True)
    assert "/" not in provenance and "\\" not in provenance and private_stem not in provenance
    assert tract["provenance"]["recorded"] is True
    assert tract["provenance"]["algorithm"] == "iFOD2"


def test_tck_header_count_alone_is_not_recorded_as_tracking_parameters(tmp_path):
    """The .tck header `count` is the file's streamline count, not a tracking parameter (Leipzig demo had only it)."""
    bank = tmp_path / "bank"
    bank.mkdir()
    tract = bank / "t.tck"
    source = b"mrtrix tracks\ndatatype: Float32LE\ncount: 0000006971\nfile: . 64\nEND\n"
    tract.write_bytes(source)
    assert cli._parameter_provenance(source, tract) == {"recorded": False}
