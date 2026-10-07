"""Every three.js addon the viewer imports must be fetched by the vendor manifest.

viewer/vendor/ is git-ignored, so a clean clone gets only what
scripts/vendor-three.mjs copies from the manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
import posixpath
import re

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
MANIFEST = json.loads((ROOT / "scripts" / "vendor-manifest.json").read_text(encoding="utf-8"))
VENDORED = {to for _pkg, _src, to in MANIFEST["files"]}
ADDON_IMPORT = re.compile(r"""['"](three/addons/[^'"]+)['"]""")
IMPORTMAP = re.compile(r'<script type="importmap">(.*?)</script>', re.S)
RELATIVE_IMPORT = re.compile(r"""(?:from|import)\s*['"](\.{1,2}/[^'"]+)['"]""")


def _resolve(specifier: str, imports: dict[str, str]) -> str:
    if specifier in imports:
        target = imports[specifier]
    else:
        prefix = max((k for k in imports if k.endswith("/") and specifier.startswith(k)), key=len)
        target = imports[prefix] + specifier[len(prefix):]
    assert target.startswith("./vendor/"), target
    return target.removeprefix("./vendor/")


def test_every_addon_import_resolves_to_a_vendored_file():
    pages = sorted(VIEWER.glob("*.html"))
    sources = sorted(VIEWER.glob("*.js")) + pages
    specifiers = {m for path in sources for m in ADDON_IMPORT.findall(path.read_text(encoding="utf-8"))}
    assert specifiers, "expected the atlas viewer to import three.js addons"
    maps = [json.loads(m)["imports"] for page in pages for m in IMPORTMAP.findall(page.read_text(encoding="utf-8"))]
    for specifier in specifiers:
        exact = [imports for imports in maps if specifier in imports]
        candidates = exact or [
            imports for imports in maps if any(k.endswith("/") and specifier.startswith(k) for k in imports)
        ]
        assert candidates, f"no importmap resolves {specifier}"
        for imports in candidates:
            assert _resolve(specifier, imports) in VENDORED, specifier


def test_draco_decoder_files_are_vendored():
    for name in ("draco_decoder.js", "draco_decoder.wasm", "draco_wasm_wrapper.js"):
        assert f"addons/libs/draco/gltf/{name}" in VENDORED


def test_vendored_addons_only_import_vendored_siblings():
    vendor = ROOT / MANIFEST["target"]
    for to in sorted(VENDORED):
        if not (to.startswith("addons/") and to.endswith(".js")):
            continue
        path = vendor / to
        if not path.is_file():
            continue
        for rel in RELATIVE_IMPORT.findall(path.read_text(encoding="utf-8")):
            sibling = posixpath.normpath(posixpath.join(posixpath.dirname(to), rel))
            assert sibling in VENDORED, f"{to} imports {rel}"
