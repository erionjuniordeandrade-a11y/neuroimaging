#!/usr/bin/env node
// Vendors pinned ESM builds from npm tarballs into the viewer's vendor/ directory.
//
//   node scripts/vendor-three.mjs           refresh vendor/ from scripts/vendor-manifest.json (needs network)
//   node scripts/vendor-three.mjs --local-assets   refresh local assets without re-fetching npm packages
//   node scripts/vendor-three.mjs --check   verify vendor/ bytes against vendor/VENDOR.json (offline)
//
// The manifest pins every package version and lists each copied file explicitly, so a
// new dependency is an intentional manifest change, never an automatic whole-tree copy.
// VENDOR.json records package versions and the sha256 of every vendored byte.
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const manifestPath=path.join(root,'scripts','vendor-manifest.json');
const manifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
const target=path.join(root,manifest.target);
const receiptPath=path.join(target,'VENDOR.json');
const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
const check=process.argv.includes('--check');
const localAssetsOnly=process.argv.includes('--local-assets');

if(check){
  if(!fs.existsSync(receiptPath)){console.error(`vendor: no receipt at ${path.relative(root,receiptPath)}; run without --check first`);process.exit(2);}
  const receipt=JSON.parse(fs.readFileSync(receiptPath,'utf8'));
  const problems=[];
  for(const [name,version] of Object.entries(manifest.packages))if(receipt.packages[name]!==version)problems.push(`package ${name}: manifest ${version}, vendored ${receipt.packages[name]??'missing'}`);
  for(const file of receipt.files){
    const full=path.join(target,file.path);
    if(!fs.existsSync(full)){problems.push(`missing ${file.path}`);continue;}
    const actual=sha256(fs.readFileSync(full));
    if(actual!==file.sha256)problems.push(`modified ${file.path}`);
  }
  const listed=new Set(receipt.files.map(f=>f.path));
  for(const [pkg,from,to] of manifest.files)if(!listed.has(to))problems.push(`not vendored yet: ${to} (${pkg}:${from})`);
  for(const [from,to] of manifest.localFiles||[]){
    if(!listed.has(to))problems.push(`not vendored yet: ${to} (${from})`);
    if(!fs.existsSync(path.join(root,from)))problems.push(`missing local source: ${from}`);
    else if(fs.existsSync(path.join(target,to))
      &&sha256(fs.readFileSync(path.join(root,from)))!==sha256(fs.readFileSync(path.join(target,to))))problems.push(`local source differs: ${to}`);
  }
  if(problems.length){console.error('vendor check FAIL\n  '+problems.join('\n  '));process.exit(1);}
  console.log(`vendor check PASS: ${receipt.files.length} files, ${Object.keys(receipt.packages).length} packages`);
  process.exit(0);
}

const tmp=localAssetsOnly?null:fs.mkdtempSync(path.join(os.tmpdir(),'vendor-three-'));
const extracted=new Map();
function packageDir(name){
  if(extracted.has(name))return extracted.get(name);
  const version=manifest.packages[name];
  if(!version)throw Error(`manifest.packages lacks a pinned version for ${name}`);
  const dest=path.join(tmp,name.replace('/','__'));
  fs.mkdirSync(dest,{recursive:true});
  const out=execFileSync('npm',['pack',`${name}@${version}`,'--pack-destination',dest,'--silent'],{encoding:'utf8',stdio:['ignore','pipe','inherit']}).trim();
  const tarball=path.join(dest,out.split('\n').pop());
  execFileSync('tar',['xzf',tarball,'-C',dest]);
  const pkg=JSON.parse(fs.readFileSync(path.join(dest,'package','package.json'),'utf8'));
  if(pkg.version!==version)throw Error(`${name}: npm delivered ${pkg.version}, manifest pins ${version}`);
  extracted.set(name,path.join(dest,'package'));
  return extracted.get(name);
}

const files=localAssetsOnly?JSON.parse(fs.readFileSync(receiptPath,'utf8')).files.filter(f=>f.package):[];
if(localAssetsOnly){
  const prior=JSON.parse(fs.readFileSync(receiptPath,'utf8'));
  for(const [name,version] of Object.entries(manifest.packages))if(prior.packages[name]!==version)throw Error(`stale ${name} package`);
  for(const file of files)if(!fs.existsSync(path.join(target,file.path))||sha256(fs.readFileSync(path.join(target,file.path)))!==file.sha256)throw Error(`stale ${file.path}`);
}
for(const [pkg,from,to] of localAssetsOnly?[]:manifest.files){
  const src=path.join(packageDir(pkg),from);
  if(!fs.existsSync(src))throw Error(`${pkg}@${manifest.packages[pkg]} has no ${from}`);
  const bytes=fs.readFileSync(src);
  const dest=path.join(target,to);
  fs.mkdirSync(path.dirname(dest),{recursive:true});
  fs.writeFileSync(dest,bytes);
  files.push({path:to,package:pkg,from,bytes:bytes.length,sha256:sha256(bytes)});
}
for(const [from,to] of manifest.localFiles||[]){
  const src=path.join(root,from);
  if(!fs.existsSync(src))throw Error(`missing local vendor source ${from}`);
  const bytes=fs.readFileSync(src),dest=path.join(target,to);
  fs.mkdirSync(path.dirname(dest),{recursive:true});fs.writeFileSync(dest,bytes);
  files.push({path:to,local:from,bytes:bytes.length,sha256:sha256(bytes)});
}
// Vendored files that the manifest no longer lists are stale copies; remove them so the
// import map cannot silently resolve to an old build.
if(fs.existsSync(receiptPath)){
  const previous=JSON.parse(fs.readFileSync(receiptPath,'utf8'));
  const keep=new Set(files.map(f=>f.path));
  for(const file of previous.files)if(!keep.has(file.path)&&fs.existsSync(path.join(target,file.path))){fs.rmSync(path.join(target,file.path));console.log(`removed stale ${file.path}`);}
}
const receipt={generated:new Date().toISOString().slice(0,10),packages:manifest.packages,files:files.sort((a,b)=>a.path.localeCompare(b.path))};
fs.writeFileSync(receiptPath,JSON.stringify(receipt,null,1)+'\n');
if(tmp)fs.rmSync(tmp,{recursive:true,force:true});
console.log(`vendored ${files.length} files from ${Object.keys(manifest.packages).length} packages into ${path.relative(root,target)}`);
for(const [name,version] of Object.entries(manifest.packages))console.log(`  ${name}@${version}`);
