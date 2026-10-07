// Offline after npm ci: copy the locked, licensed browser runtime into the
// static viewer root. Keep patient assets and the network out of this step.
import { readFile, mkdir, copyFile, cp } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const src=path.join(root,'node_modules','three');
const pkg=JSON.parse(await readFile(path.join(src,'package.json'),'utf8'));
if(pkg.version!=='0.185.0') throw new Error(`Expected locked Three.js 0.185.0, got ${pkg.version}`);
const dest=path.join(root,'viewer','vendor');
await mkdir(dest,{recursive:true});
for(const [from,to] of [['build/three.module.js','three.module.js'],
  ['build/three.core.js','three.core.js'],['examples/jsm/controls/OrbitControls.js','OrbitControls.js'],
  ['LICENSE','LICENSE.md']]) await copyFile(path.join(src,from),path.join(dest,to));
// Keep loaders, their shared utilities and Draco decoder on the same locked
// Three release. Local imports and workers require no CDN at runtime.
for (const dir of ['loaders','utils','libs/draco']) {
  await cp(path.join(src,'examples/jsm',dir),path.join(dest,'addons',dir),{recursive:true});
}
console.log(`Restored Three.js ${pkg.version} and its MIT license to viewer/vendor`);
