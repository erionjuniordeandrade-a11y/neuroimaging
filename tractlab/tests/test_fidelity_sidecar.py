"""E1/S3 — fidelity sidecars keyed by (bank sha, ordinal); exact join or refuse."""
import hashlib

import nibabel as nib
import numpy as np
import pytest

from tractlab.fidelity import (
    FidelityRefusal,
    R_GRID,
    FLAG_PROVENANCE_INCOMPLETE,
    build_sidecar,
    load_sidecar,
    rows_for,
    save_sidecar,
)

Y00 = 0.28209479177387814
AFF = np.diag([2.0, 2.0, 2.0, 1.0])


@pytest.fixture()
def case(tmp_path):
    sh = np.zeros((8, 8, 8, 45))
    sh[..., 0] = 2.0
    sh_image = nib.Nifti1Image(sh.astype(np.float32), AFF)
    peak_image = nib.Nifti1Image((sh[..., 0] * Y00).astype(np.float32), AFF)
    sh_image.header.set_xyzt_units("mm", "sec")
    peak_image.header.set_xyzt_units("mm", "sec")
    nib.save(sh_image, str(tmp_path / "fod.nii.gz"))
    nib.save(peak_image, str(tmp_path / "peak.nii.gz"))
    lines = [
        np.array([[3.0, 7.0, 7.0], [11.0 + i, 7.0, 7.0]], dtype=np.float32)
        for i in range(5)
    ]
    t = nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(t, str(tmp_path / "bank.tck"))
    bank_sha = hashlib.sha256((tmp_path / "bank.tck").read_bytes()).hexdigest()
    fod_sha = hashlib.sha256((tmp_path / "fod.nii.gz").read_bytes()).hexdigest()
    prov = {"bank_sha256": bank_sha, "fod_sha256": fod_sha}
    return tmp_path, prov


def test_sidecar_roundtrip_and_exact_join(case):
    root, prov = case
    data = build_sidecar(root / "bank.tck", root / "fod.nii.gz",
                         root / "peak.nii.gz", provenance=prov)
    p = root / "bank_test.fidelity.npz"
    save_sidecar(p, data)
    sc = load_sidecar(p, expected_bank_sha=prov["bank_sha256"])
    assert sc.n == 5
    assert sc.frac_ge.shape == (5, len(R_GRID))
    assert sc.p5_ratio.shape == (5,)
    # isotropic support: every streamline fully >= all thresholds up to 0.6
    assert np.all(sc.frac_ge[:, 0] > 0.99)
    sub = rows_for(sc, np.array([4, 0, 2]))
    assert sub.n == 3
    assert np.allclose(sub.p5_ratio, sc.p5_ratio[[4, 0, 2]])
    assert np.array_equal(sub.flags, sc.flags[[4, 0, 2]])


def test_wrong_bank_sha_refuses(case):
    root, prov = case
    data = build_sidecar(root / "bank.tck", root / "fod.nii.gz",
                         root / "peak.nii.gz", provenance=prov)
    p = root / "s.npz"
    save_sidecar(p, data)
    with pytest.raises(FidelityRefusal, match="sha"):
        load_sidecar(p, expected_bank_sha="00" * 32)


def test_truncated_sidecar_refuses(case):
    root, prov = case
    data = build_sidecar(root / "bank.tck", root / "fod.nii.gz",
                         root / "peak.nii.gz", provenance=prov)
    data["p5_ratio"] = data["p5_ratio"][:3]  # truncate one array
    p = root / "bad.npz"
    save_sidecar(p, data)
    with pytest.raises(FidelityRefusal, match="length"):
        load_sidecar(p, expected_bank_sha=prov["bank_sha256"])


def test_rebuild_is_byte_identical(case):
    root, prov = case
    a, b = root / "a.npz", root / "b.npz"
    save_sidecar(a, build_sidecar(root / "bank.tck", root / "fod.nii.gz",
                                  root / "peak.nii.gz", provenance=prov))
    save_sidecar(b, build_sidecar(root / "bank.tck", root / "fod.nii.gz",
                                  root / "peak.nii.gz", provenance=prov))
    assert a.read_bytes() == b.read_bytes()


def test_out_of_range_ordinal_refuses(case):
    root, prov = case
    data = build_sidecar(root / "bank.tck", root / "fod.nii.gz",
                         root / "peak.nii.gz", provenance=prov)
    p = root / "s.npz"
    save_sidecar(p, data)
    sc = load_sidecar(p, expected_bank_sha=prov["bank_sha256"])
    with pytest.raises(FidelityRefusal, match="ordinal"):
        rows_for(sc, np.array([0, 7]))


def test_incomplete_provenance_records_refusal_flags(case):
    root, _prov = case
    data = build_sidecar(root / "bank.tck", root / "fod.nii.gz",
                         root / "peak.nii.gz",
                         provenance={"bank_sha256": "ab" * 32})  # no fod sha
    assert data["ratios_present"] is False
    assert data["p5_ratio"].shape == (0,)
    assert np.all(data["flags"] & FLAG_PROVENANCE_INCOMPLETE)
    assert data["flags"].shape == (5,)


def test_arbitrary_mif_bytes_cannot_be_legitimized_by_a_load_copy(case):
    root, prov = case
    # Stand-in for a .mif FOD: bytes nibabel cannot read, pinned in the
    # manifest; the nibabel-readable conversion is passed as sh_load_path.
    mif = root / "fod.mif"
    mif.write_bytes(b"mif-container-bytes, not nifti")
    prov_mif = dict(prov,
                    fod_sha256=hashlib.sha256(mif.read_bytes()).hexdigest())
    with pytest.raises(FidelityRefusal, match="verify native MIF"):
        build_sidecar(root / "bank.tck", mif, root / "peak.nii.gz",
                      provenance=prov_mif, sh_load_path=root / "fod.nii.gz")


def test_sh_load_path_never_satisfies_the_sha_gate(case):
    root, prov = case
    mif = root / "fod.mif"
    mif.write_bytes(b"mif-container-bytes, not nifti")
    # prov carries the sha of fod.nii.gz — the LOAD copy. The pin must be
    # computed on sh_path (the .mif), so this has to refuse.
    with pytest.raises(FidelityRefusal, match="FOD sha mismatch"):
        build_sidecar(root / "bank.tck", mif, root / "peak.nii.gz",
                      provenance=prov,
                      sh_load_path=root / "fod.nii.gz")
