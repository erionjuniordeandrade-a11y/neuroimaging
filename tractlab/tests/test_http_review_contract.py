"""Additional generated regressions from the independent integration review."""
import http.client
import json
from urllib.parse import urlsplit
import numpy as np
import pytest

from tractlab.fidelity import FidelityRefusal, load_sidecar, save_sidecar
from tractlab.serve import header_ascii
from test_audit_server_contract import _tiny_case, _serve, _load_bank, _post_json, _hdr


def test_header_metadata_cannot_inject_response_fields(tmp_path):
    assert header_ascii('name\r\nX-Injected: yes\x00') == 'name  X-Injected: yes '
    manifest, _ = _tiny_case(tmp_path)
    m=json.loads(manifest.read_text());m['inputs']['bank_test']['label']='name\r\nX-Injected: yes'
    manifest.write_text(json.dumps(m))
    server, service, base = _serve(manifest)
    try:
        status, _, headers = _load_bank(base,service)
        assert status==200
        assert _hdr(headers,'X-Injected') is None
    finally:
        server.shutdown()


def test_unknown_post_closes_instead_of_parsing_its_body_as_a_request(tmp_path):
    manifest,_=_tiny_case(tmp_path);server,service,base=_serve(manifest)
    try:
        url=urlsplit(base);conn=http.client.HTTPConnection(url.hostname,url.port,timeout=5)
        conn.request('POST','/api/not-installed',body=b'{"unread":1}',headers={'Content-Type':'application/json'})
        response=conn.getresponse();response.read()
        assert response.status==404 and response.getheader('Connection')=='close'
        # A client may reconnect, but the stale body cannot become a method.
        conn.request('GET','/api/health');response=conn.getresponse()
        assert response.status==200;response.read();conn.close()
    finally:
        server.shutdown()


@pytest.mark.parametrize('kind',['negative','above_one','infinity','partial_nan'])
def test_corrupt_support_rows_refuse_before_publication(tmp_path,kind):
    manifest,_=_tiny_case(tmp_path,with_sidecar=True)
    path=next((tmp_path/'fidelity').glob('*.npz'))
    with np.load(path) as archive:data={k:archive[k] for k in archive.files}
    data['frac_ge']=data['frac_ge'].copy()
    data['frac_ge'][0,0]={'negative':-.1,'above_one':1.1,'infinity':np.inf,'partial_nan':np.nan}[kind]
    np.savez(path,**data)
    server,service,base=_serve(manifest)
    try:
        assert _load_bank(base,service)[0]==422
        assert service.analytic_sources._latest is None
    finally:server.shutdown()
