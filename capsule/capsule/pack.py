"""Single-file payload writer and a small independent decoder."""

from __future__ import annotations

import base64
import gzip
import json
from html.parser import HTMLParser
from pathlib import Path
import re

import numpy as np


PLACEHOLDER = "<!--CAPSULE_PAYLOAD-->"
_MANIFEST = re.compile(r'<script id="capsule-manifest" type="application/json">(.*?)</script>', re.S)
_BLOB = re.compile(r'<script id="capsule-blob-([^"<>]+)" type="application/octet-stream" data-encoding="gzip\+base64">(.*?)</script>', re.S)


class _CapsuleHTMLParser(HTMLParser):
    """Decode only the inline manifest and blob scripts from a saved capsule."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.scripts: dict[str, dict[str, str]] = {}
        self._active_id: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        attributes = {key: value or "" for key, value in attrs}
        script_id = attributes.get("id")
        if script_id and (script_id == "capsule-manifest" or script_id.startswith("capsule-blob-")):
            if script_id in self.scripts:
                raise ValueError("capsule contains duplicate manifest or blob scripts")
            self._active_id = script_id
            self.scripts[script_id] = {"text": "", **attributes}

    def handle_data(self, data: str) -> None:
        if self._active_id is not None:
            self.scripts[self._active_id]["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            self._active_id = None


def parse_capsule(path: str | Path, wanted=None) -> tuple[dict, dict[str, bytes]]:
    """Return the saved manifest and raw decoded blobs using the viewer payload format.

    ``wanted(manifest) -> set[str]`` limits decompression to those blob ids (large volume
    blobs are then never inflated)."""
    parser = _CapsuleHTMLParser()
    parser.feed(Path(path).read_text(encoding="utf-8"))
    manifest_script = parser.scripts.get("capsule-manifest")
    if manifest_script is None:
        raise ValueError("capsule manifest not found")
    manifests = [script_id for script_id in parser.scripts if script_id == "capsule-manifest"]
    if len(manifests) != 1:
        raise ValueError("capsule must contain exactly one manifest")
    try:
        manifest = json.loads(manifest_script["text"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("capsule manifest is invalid JSON") from exc
    keep = None if wanted is None else set(wanted(manifest))
    blobs: dict[str, bytes] = {}
    for script_id, item in parser.scripts.items():
        if not script_id.startswith("capsule-blob-"):
            continue
        if keep is not None and script_id.removeprefix("capsule-blob-") not in keep:
            continue
        if item.get("type") != "application/octet-stream" or item.get("data-encoding") != "gzip+base64":
            raise ValueError("capsule blob has an unsupported encoding")
        blob_id = script_id.removeprefix("capsule-blob-")
        try:
            blobs[blob_id] = gzip.decompress(base64.b64decode(item["text"].strip(), validate=True))
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError("capsule blob is invalid") from exc
    return manifest, blobs


def write_capsule(template: str | Path, destination: str | Path, manifest: dict, arrays: dict[str, np.ndarray]) -> None:
    template_text = Path(template).read_text(encoding="utf-8")
    if template_text.count(PLACEHOLDER) != 1:
        raise ValueError("viewer template must contain exactly one <!--CAPSULE_PAYLOAD--> placeholder")
    encoded_manifest = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    parts = [f'<script id="capsule-manifest" type="application/json">{encoded_manifest}</script>']
    for blob_id, array in arrays.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", blob_id):
            raise ValueError("unsafe blob id")
        payload = base64.b64encode(gzip.compress(np.ascontiguousarray(array).tobytes(order="C"), mtime=0)).decode("ascii")
        parts.append(f'<script id="capsule-blob-{blob_id}" type="application/octet-stream" data-encoding="gzip+base64">{payload}</script>')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(template_text.replace(PLACEHOLDER, "\n".join(parts)))


def _read_payload(path: str | Path) -> tuple[dict, dict[str, bytes]]:
    return parse_capsule(path)


def read_capsule(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Decode volume and mask grids; tract blobs are validated but not returned."""
    manifest, blobs = _read_payload(path)
    nx, ny, nz = manifest["grid"]["dims"]
    descriptors = {item["blob"]: item for item in manifest.get("volumes", []) + manifest.get("masks", [])
                   + manifest.get("anatomy", [])}
    tract_blobs = {blob for item in manifest.get("tracts", [])
                   for blob in (item.get("blob"), item.get("outlier_blob")) if blob}
    if set(blobs) != set(descriptors) | tract_blobs or set(descriptors) & tract_blobs:
        raise ValueError("capsule blobs do not match manifest")
    arrays = {}
    for blob_id, descriptor in descriptors.items():
        dtype = np.dtype(descriptor.get("dtype", "uint8"))
        arrays[blob_id] = np.frombuffer(blobs[blob_id], dtype=dtype).reshape((nz, ny, nx)).copy()
    return manifest, arrays


def read_capsule_tract_blobs(path: str | Path) -> dict[str, np.ndarray]:
    """Read raw tract payloads as uint8 arrays without holding decoded volume blobs in memory."""
    def tract_blob_ids(manifest: dict) -> set[str]:
        return {blob for item in manifest.get("tracts", [])
                for blob in (item.get("blob"), item.get("outlier_blob")) if blob}

    manifest, blobs = parse_capsule(path, tract_blob_ids)
    expected = tract_blob_ids(manifest)
    if not expected <= set(blobs):
        raise ValueError("capsule blobs do not match manifest")
    return {blob_id: np.frombuffer(blobs[blob_id], dtype=np.uint8).copy() for blob_id in expected}


def read_capsule_tracts(path: str | Path) -> dict[str, list[np.ndarray]]:
    """Decode each manifest tract (keyed by tract id) into world-mm streamlines."""
    from .tracts import decode_tck

    manifest, blobs = _read_payload(path)
    return {item["id"]: decode_tck(blobs[item["blob"]]) for item in manifest.get("tracts", [])}
