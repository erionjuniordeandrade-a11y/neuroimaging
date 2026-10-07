"""Import a bounded, public atlas package from the pinned reference checkout.

No downloads or case paths. Source terms were checked against the upstream
HCP, Labsolver, Melbourne and MNI pages on 2026-09-07. Rerunning requires the
same source commit; geometry and label order are never changed independently.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path.home() / 'reference' / 'human-brain'
PIN = 'e028766'
OUT = ROOT / 'viewer' / 'atlas'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if not commit.startswith(PIN):
        raise SystemExit('Source revision changed: re-review provenance before importing')
    assets = []
    def record(src, dst, license_name):
        assets.append({'path': str(dst.relative_to(OUT)), 'sha256': sha(dst),
                       'bytes': dst.stat().st_size, 'source_path': src,
                       'source_sha256': sha(SOURCE / src), 'terms': license_name})
    def copy(src, dest, terms):
        dst = OUT / dest
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / src, dst)
        record(src, dst, terms)
    surface = json.loads((SOURCE / 'data/surface.json').read_text())
    labelset = surface['sets']['glasser']
    raw = (SOURCE / 'data/surface-labels.bin').read_bytes()
    offset, count = labelset['offset'], labelset['count']
    if count != 64984 or labelset['counts'] != [32492, 32492] or not surface['built_from_compressed_mesh']:
        raise SystemExit('Unexpected cortex / label correspondence')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'surface-labels.bin').write_bytes(raw[offset:offset + count * 2])
    record('data/surface-labels.bin', OUT / 'surface-labels.bin', 'HCP Open Access Data Use Terms')
    labelset['offset'] = 0
    surface['sets'] = {'glasser': labelset}
    surface['bin'] = 'surface-labels.bin'
    surface['surface']['mesh'] = {'L': 'cortex-L.glb', 'R': 'cortex-R.glb'}
    (OUT / 'surface.json').write_text(json.dumps(surface, indent=2) + '\n')
    record('data/surface.json', OUT / 'surface.json', 'HCP Open Access Data Use Terms')
    for h in ['L', 'R']:
        copy(f'meshes/cortex/{h}.glb', f'cortex-{h}.glb', 'HCP Open Access Data Use Terms')
    copy('data/tracts.bin', 'tracts.bin', 'CC BY-SA 4.0 + HCP acknowledgement')
    copy('data/tracts.json', 'tracts.json', 'CC BY-SA 4.0 + HCP acknowledgement')
    copy('meshes/cerebellum/rest.glb', 'inferior-context.glb', 'MNI template permission notice')
    copy('licenses/mni-template-license.txt', 'licenses/mni-template-license.txt', 'MNI template permission notice')
    sub = json.loads((SOURCE / 'data/subcortex.json').read_text())
    for entry in sub['structures']:
        copy(entry['mesh'], f"subcortex/{entry['id']}.glb", 'Melbourne Subcortex Atlas License')
        entry['mesh'] = f"subcortex/{entry['id']}.glb"
    (OUT / 'subcortex.json').write_text(json.dumps(sub, indent=2) + '\n')
    record('data/subcortex.json', OUT / 'subcortex.json', 'Melbourne Subcortex Atlas License')
    manifest = {
        'id': 'tractlab-reference-2026-09-07', 'version': 1, 'evidenceClass': 'atlas',
        'source_repository': 'https://github.com/amyleesterling/human-brain',
        'source_commit': commit, 'license_review_date': '2026-09-07',
        'coordinate_contract': 'Group reference surfaces and pathways in MNI conventions. Different template constructions; illustrative juxtaposition only. No cross-atlas distance or endpoint connectivity inference, no case registration.',
        'modifications': 'GLB and tract binary copied unchanged. HCP-MMP1 block extracted with its paired hemisphere tables; vertex order unchanged. Paths shortened. No Yeo or H01 assets included.',
        'assets': assets,
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Imported {len(assets)} public atlas files, {sum(a["bytes"] for a in assets):,} bytes, source {commit}')

if __name__ == '__main__':
    main()
