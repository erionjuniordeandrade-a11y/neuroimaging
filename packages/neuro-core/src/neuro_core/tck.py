"""Minimal MRtrix .tck streamline reader/writer.

Coordinates are world RAS+ millimetres (MRtrix scanner space) and are never
transformed here. The writer emits a minimal header only: source header lines
can carry file paths or identifiers and are never copied.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np


_MAGIC = b"mrtrix tracks"
_DTYPES = {"Float32LE": np.dtype("<f4"), "Float32BE": np.dtype(">f4")}


def decode_tck(data: bytes) -> list[np.ndarray]:
    """Parse .tck bytes into a list of (n, 3) float32 arrays in world mm."""
    end = re.search(rb"(?m)^END\r?\n", data)
    if not data.startswith(_MAGIC) or end is None:
        raise ValueError("not an MRtrix .tck file (missing 'mrtrix tracks' magic or END line)")
    fields: dict[str, str] = {}
    for line in data[:end.start()].decode("latin-1").splitlines()[1:]:
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    dtype_name = fields.get("datatype")
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported .tck datatype {dtype_name!r}; expected Float32LE or Float32BE")
    match = re.fullmatch(r"\.\s+(\d+)", fields.get("file", ""))
    if match is None:
        raise ValueError("unsupported .tck 'file' field; expected '. <offset>'")
    offset = int(match.group(1))
    if offset < end.end() or offset > len(data):
        raise ValueError(".tck data offset lies inside the header or beyond the file")
    body = data[offset:]
    usable = len(body) - len(body) % 12
    points = np.frombuffer(body[:usable], dtype=_DTYPES[dtype_name]).reshape(-1, 3).astype(np.float32)
    inf_rows = np.flatnonzero(np.isinf(points).any(axis=1))
    if len(inf_rows):
        points = points[:inf_rows[0]]
    separators = np.flatnonzero(np.isnan(points).any(axis=1))
    streamlines = []
    start = 0
    for stop in [*separators.tolist(), len(points)]:
        if stop > start:
            streamlines.append(points[start:stop].copy())
        start = stop + 1
    return streamlines


def read_tck(path: str | Path) -> list[np.ndarray]:
    return decode_tck(Path(path).read_bytes())


def encode_tck(streamlines: list[np.ndarray]) -> bytes:
    """Serialise streamlines as Float32LE .tck with a minimal header."""
    chunks = []
    separator = np.full((1, 3), np.nan, dtype="<f4")
    for streamline in streamlines:
        array = np.asarray(streamline, dtype="<f4")
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError("each streamline must be a finite (n, 3) array")
        chunks.extend([array, separator])
    chunks.append(np.full((1, 3), np.inf, dtype="<f4"))
    body = np.concatenate(chunks).astype("<f4").tobytes()
    offset = 0
    while True:
        header = (f"mrtrix tracks\ndatatype: Float32LE\ncount: {len(streamlines)}\n"
                  f"file: . {offset}\nEND\n").encode("ascii")
        if len(header) == offset:
            return header + body
        offset = len(header)


def write_tck(path: str | Path, streamlines: list[np.ndarray]) -> None:
    with Path(path).open("xb") as output:
        output.write(encode_tck(streamlines))
