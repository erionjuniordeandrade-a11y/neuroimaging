"""Interactive QA fixture: mathematical curves and ellipsoid, no anatomy.

Runs the current integration viewer with generated data. Ctrl-C closes the listener and
removes temporary volumes/exports. Does not invoke a tracking engine.
"""
import hashlib
import argparse
import json
import shutil
import signal
import sys
import tempfile
import threading
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tractlab.serve import serve
from tractlab.fidelity import R_GRID, SIDECAR_SCHEMA, save_sidecar
from tractlab.http_policy import PolicyRejected

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='tractlab-synthetic-viewer-') as tmp:
        case=Path(tmp)
        shape=(40,40,40)
        aff=np.diag([2.,2.,2.,1.]);aff[:3,3]=-40
        xyz=np.indices(shape).astype(float)
        ell=((xyz[0]-20)/16)**2+((xyz[1]-20)/18)**2+((xyz[2]-20)/18)**2
        mask=(ell<1).astype(np.uint8)
        b0=(np.maximum(0,1-ell)*100).astype(np.float32)
        fod=np.ones(shape+(6,),dtype=np.float32)
        lesion=(((xyz[0]-25)**2+(xyz[1]-20)**2+(xyz[2]-20)**2)<9).astype(np.uint8)
        for name,data in [('mask',mask),('b0',b0),('fod',fod),('lesion',lesion)]:
            image = nib.Nifti1Image(data,aff)
            image.header.set_xyzt_units('mm')
            nib.save(image,str(case/f'{name}.nii.gz'))
        fod_sha=hashlib.sha256((case/'fod.nii.gz').read_bytes()).hexdigest()
        inputs={name:{'path':f'{name}.nii.gz'} for name in ['b0','mask','fod','lesion']}
        (case/'fidelity').mkdir()
        peak_image = nib.Nifti1Image(np.ones(shape, np.float32), aff)
        peak_image.header.set_xyzt_units('mm')
        peak_path = case/'fidelity/fod_peak.nii.gz'
        nib.save(peak_image, str(peak_path))
        peak_sha = hashlib.sha256(peak_path.read_bytes()).hexdigest()
        rng=np.random.default_rng(20260906)
        t=np.linspace(-1,1,96)
        for bid,n,offset,marked in [('bank_cst_r',900,7,True),('bank_cst_l',100,-9,False),('bank_or_l',3000,-2,True)]:
            curves=[]
            for _ in range(n):
                dx,dy=rng.normal(0,1.5,2)
                curves.append(np.column_stack([offset+dx+5*t*t,dy+7*np.sin(t*1.7),28*t]).astype(np.float32))
            filename=f'{bid}.tck'
            nib.streamlines.save(nib.streamlines.Tractogram(curves,affine_to_rasmm=np.eye(4)),str(case/filename))
            sha=hashlib.sha256((case/filename).read_bytes()).hexdigest()
            inputs[bid]={'path':filename,'label':'SYNTHETIC curves — no anatomy','role':'true_cst' if bid=='bank_cst_r' else 'bank',
                'n_streamlines':n,'engine':'synthetic mathematical QA fixture',
                'provenance':{'bank_sha256':sha,'fod_sha256':fod_sha,'sources':{'bank_sha256':'recorded','fod_sha256':'recorded'}}}
            if marked:
                frac=np.full((n,len(R_GRID)),.9,dtype=np.float32);frac[::4]=.1
                save_sidecar(case/'fidelity'/f'{bid}.fidelity.npz',{'schema':SIDECAR_SCHEMA,'bank_sha256':sha,'fod_sha256':fod_sha,
                    'sh_load_sha256':fod_sha,'peak_sha256':peak_sha,
                    'R_grid':R_GRID,'frac_ge':frac,'p5_ratio':frac[:,0].copy(),
                    'n_segments':np.full(n,95,dtype=np.uint16),'flags':np.zeros(n,dtype=np.uint8),'ratios_present':True})
        manifest={'case_id':'SYNTHETIC-QA-NO-ANATOMY','case_root':str(case),'inputs':inputs,
            'acquisition':{'geom_floor_mm':3},'recipe_hash':'synthetic-fixture'}
        man=case/'manifest.json';man.write_text(json.dumps(manifest))
        httpd,service=serve(str(man),str(ROOT/'viewer'),port=args.port)
        handler=httpd.RequestHandlerClass
        original_get=handler.do_GET
        original_post=handler.do_POST
        def do_GET(self):
            try:
                self._enforce_browser_origin_policy()
            except PolicyRejected as e:
                return self._reject_policy(e)
            if self.path.split('?',1)[0] in ('','/','/index.html'):
                # Only a QA identification badge is injected. Renderer and
                # product controls are exactly the frozen source.
                html=(ROOT/'viewer/index.html').read_text().replace('<title>TractLab — DWI seed tracking</title>',
                    '<title>TractLab synthetic QA — no anatomy</title>')
                # Full-canvas guard: a screenshot of this page must never pass for the product.
                guard=('<div id="scene"><div data-synthetic-guard style="position:absolute;inset:0;z-index:9;pointer-events:none;'
                    'display:flex;align-items:center;justify-content:center;overflow:hidden">'
                    '<div style="transform:rotate(-24deg);font:700 clamp(20px,3vw,40px)/1.15 monospace;letter-spacing:.06em;'
                    'color:rgba(255,96,64,.28);text-align:center;white-space:nowrap">SYNTHETIC QA<br>NO ANATOMY · NOT THE CASE</div></div>'
                    '<div style="position:absolute;top:8px;left:8px;z-index:10;display:flex;gap:16px;align-items:center;max-width:60%;'
                    'font:12px monospace;color:#fff;background:#b3261e;padding:6px 10px;border-radius:6px">'
                    '<span>SYNTHETIC QA FIXTURE · generated curves, no anatomy or measured support</span>'
                    '<a href="./atlas.html?profile=teaching" style="color:#fff;font-weight:700;pointer-events:auto;white-space:nowrap">Open the reference atlas →</a></div>')
                html=html.replace('<div id="scene">',guard)
                return self._send(200,service.runtime_identity.inject_document_meta(html.encode()),'text/html')
            return original_get(self)
        def do_POST(self):
            if self.path=='/api/track':
                return self._json(400,{'error':'tracking disabled in synthetic QA fixture'})
            return original_post(self)
        handler.do_GET=do_GET;handler.do_POST=do_POST
        stop=threading.Event()
        signal.signal(signal.SIGINT,lambda *_:stop.set())
        signal.signal(signal.SIGTERM,lambda *_:stop.set())
        thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
        print(json.dumps({'url':f'http://127.0.0.1:{httpd.server_address[1]}/?teaching=1',
            'scope':'generated geometry only; no anatomy; real integration viewer; tracking disabled'}),flush=True)
        try:
            stop.wait()
        finally:
            httpd.shutdown();httpd.server_close();thread.join(timeout=3)

if __name__=='__main__':main()
