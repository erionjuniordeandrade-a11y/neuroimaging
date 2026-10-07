"""Additional integration checks using generated geometry and loopback only."""
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import shutil
import subprocess
import threading

import nibabel as nib
import numpy as np
import pytest

from tractlab.bank import load_tracks_cached
from tractlab.evidence_identity import sha256_file
from tractlab.fidelity import FidelityRefusal, _verified_sh_image
from tractlab.grid import Grid, grid_id
from tractlab.http_policy import PolicyRejected, check_content_length, check_origin, enforce_browser_origin_policy
from tractlab.results import Result, ResultRegistry
from tractlab.roi_boolean import RoiCompileError, layers_from_request
from test_audit_server_contract import _tiny_case, _serve, _load_bank, _post_json, _hdr


def test_grid_owns_its_affine():
    source = np.eye(4)
    grid = Grid((4, 4, 4), source, ('R', 'A', 'S'), spatial_unit='mm')
    identity = grid_id(grid)
    source[0, 3] = 40
    assert grid_id(grid) == identity
    with pytest.raises(ValueError):
        grid.affine[0, 3] = 40


@pytest.mark.parametrize('key', ['and', 'or', 'not'])
@pytest.mark.parametrize('value', [1, False, None, {}, 'bad'])
def test_malformed_roi_collections_are_typed(key, value):
    with pytest.raises(RoiCompileError, match='list'):
        layers_from_request({key: value}, require_seed=False)


@pytest.mark.parametrize('value', ['+1', '1_0', ' 2', '2.0', '-1'])
def test_content_length_requires_wire_decimal(value):
    with pytest.raises(PolicyRejected):
        check_content_length(value, max_body=100, allow_zero=False)


@pytest.mark.parametrize('origin', ['http://127.0.0.1:abc', 'http://[',
    'http://user@127.0.0.1:8765', 'http://127.0.0.1:8765/path'])
def test_malformed_origin_is_typed(origin):
    with pytest.raises(PolicyRejected):
        check_origin(origin, expected_port=8765)


def test_origin_must_match_the_request_host():
    with pytest.raises(PolicyRejected):
        enforce_browser_origin_policy(host_header='127.0.0.1:8765',
            origin_header='http://localhost:8765', sec_fetch_site='same-site', expected_port=8765)


def test_same_size_mtime_preserving_bank_replacement_invalidates_caches(tmp_path):
    path = tmp_path / 'bank.tck'
    def write(x):
        nib.streamlines.save(nib.streamlines.Tractogram(
            [np.array([[x, 1, 1], [x+1, 1, 1]], dtype=np.float32)], affine_to_rasmm=np.eye(4)), str(path))
    write(1)
    first = sha256_file(path)
    original = path.stat()
    assert load_tracks_cached(str(path))[0][0][0, 0] == 1
    write(9)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert path.stat().st_size == original.st_size
    assert sha256_file(path) != first
    assert load_tracks_cached(str(path))[0][0][0, 0] == 9


def _image(path, coefficient):
    image = nib.Nifti1Image(np.full((4, 5, 6, 6), coefficient, np.float32), np.diag([1., 2., 3., 1.]))
    image.header.set_xyzt_units('mm')
    nib.save(image, str(path))


def test_wrong_same_grid_sh_conversion_refuses(tmp_path):
    source, wrong = tmp_path/'fod.nii.gz', tmp_path/'wrong.nii.gz'
    _image(source, 1)
    _image(wrong, 2)
    with pytest.raises(FidelityRefusal, match='coefficients'):
        _verified_sh_image(source, wrong)
    assert np.all(_verified_sh_image(source, source).get_fdata() == 1)


@pytest.mark.skipif(not (shutil.which('mrconvert') or (Path.home()/'mrtrix3/bin/mrconvert').is_file()), reason='optional MRtrix format oracle')
def test_native_mif_conversion_is_compared_to_fresh_source(tmp_path):
    exe = shutil.which('mrconvert') or str(Path.home()/'mrtrix3/bin/mrconvert')
    original, native, conversion = tmp_path/'source.nii.gz', tmp_path/'source.mif', tmp_path/'converted.nii.gz'
    _image(original, 1)
    subprocess.run([exe, str(original), str(native), '-quiet'], check=True)
    subprocess.run([exe, str(native), str(conversion), '-quiet'], check=True)
    assert np.all(_verified_sh_image(native, conversion).get_fdata() == 1)
    _image(conversion, 2)
    with pytest.raises(FidelityRefusal, match='coefficients'):
        _verified_sh_image(native, conversion)


def test_result_retention_metadata_is_bounded_and_owned():
    registry = ResultRegistry(max_count=2)
    metadata = {'fidelityStatus': 'absent'}
    def build():
        return Result.build(source_population='generated', derivation_id=None,
            grid_id='grid', volume_id='volume', lines=[np.zeros((2,3))], evidence=metadata)
    for _ in range(30):
        current = registry.publish(build())
    assert len(registry._last_access) == 2
    metadata['fidelityStatus'] = 'ok'
    assert current.evidence['fidelityStatus'] == 'absent'
    assert list(current.source_ordinals) == [0]
    with pytest.raises(TypeError):
        current.evidence['fidelityStatus'] = 'ok'


def test_changed_peak_refuses_without_publishing(tmp_path):
    manifest, _ = _tiny_case(tmp_path, with_sidecar=True)
    httpd, service, base = _serve(manifest)
    try:
        assert _load_bank(base, service)[0] == 200
        peak = tmp_path/'fidelity/fod_peak.nii.gz'
        image = nib.load(str(peak))
        nib.save(nib.Nifti1Image(image.get_fdata()+1, image.affine, image.header), str(peak))
        before = service.analytic_sources._latest
        assert _load_bank(base, service)[0] == 422
        assert service.analytic_sources._latest is before
    finally:
        httpd.shutdown()


@pytest.mark.skip(reason="ResultRegistry contract from archive fix/audit-20260906; main uses AnalyticSourceRegistry. Follow-up in docs/INVENTORY.md.")
def test_simultaneous_exports_keep_request_population_and_private_paths(tmp_path, monkeypatch):
    import tractlab.serve as serve_module
    manifest, _ = _tiny_case(tmp_path)
    httpd, service, base = _serve(manifest)
    paths, barrier = [], threading.Barrier(2)
    original = serve_module.write_tck
    def delayed_write(lines, path):
        paths.append(path)
        n = original(lines, path)
        barrier.wait(timeout=10)
        return n
    monkeypatch.setattr(serve_module, 'write_tck', delayed_write)
    try:
        a = service.results.publish(Result.build(source_population='A', derivation_id=None,
            grid_id=service.grid_id, volume_id=service.volume_id,
            lines=[np.array([[1,2,3],[4,5,6]], np.float32)]))
        b = service.results.publish(Result.build(source_population='B', derivation_id=None,
            grid_id=service.grid_id, volume_id=service.volume_id,
            lines=[np.array([[9,2,3],[14,5,6]], np.float32)]*2))
        with ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(lambda r: _post_json(base, '/api/export/tck', {'resultId':r.id}), [a,b]))
        for result, (status, data, headers) in zip([a,b], responses):
            assert status == 200
            exported = nib.streamlines.tck.TckFile.load(io.BytesIO(data))
            assert len(exported.streamlines) == result.n_analytic_full
            assert np.array_equal(exported.streamlines[0], result.lines[0])
            assert _hdr(headers, 'X-resultId') == result.id
        assert len(set(paths)) == 2
        assert all(not Path(p).parent.exists() for p in paths)
    finally:
        httpd.shutdown()
