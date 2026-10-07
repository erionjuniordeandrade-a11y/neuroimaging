"""Fetch one thin-slice head CT series from the public CQ500 dataset and store it uncompressed.

CQ500 (qure.ai; Chilamkurthy et al., arXiv:1803.05854) is served login-free from S3 under
CC BY-NC-SA 4.0. Its DICOM uses JPEG 2000 lossless (1.2.840.10008.1.2.4.90), which the
bundled dcm2niix cannot decode, so the selected series is rewritten as Explicit VR Little
Endian with identical pixel values. Nothing but the pixel encoding changes.

Run:
  uv run --with pylibjpeg --with pylibjpeg-openjpeg python scripts/fetch_cq500_thin.py \
      --case CQ500-CT-401 --series 3 --out data/public/CQ500/head-ct-thin
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import sys
import urllib.request
import zipfile

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian

BASE = "https://s3.ap-south-1.amazonaws.com/qure.headct.study"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="CQ500-CT-401")
    parser.add_argument("--series", type=int, default=3, help="SeriesNumber of the thin series to keep")
    parser.add_argument("--out", default="data/public/CQ500/head-ct-thin")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"{out} already populated; nothing to do (never overwritten)")
        return 0
    url = f"{BASE}/{args.case}.zip"
    data = urllib.request.urlopen(url, timeout=600).read()
    print(f"{url}: {len(data) / 1e6:.1f} MB, sha256 {hashlib.sha256(data).hexdigest()}")
    archive = zipfile.ZipFile(io.BytesIO(data))
    kept = []
    for name in archive.namelist():
        if name.endswith("/"):
            continue
        ds = pydicom.dcmread(io.BytesIO(archive.read(name)))
        if int(ds.SeriesNumber) != args.series:
            continue
        pixels = ds.pixel_array  # decodes JPEG 2000 lossless
        ds.PixelData = np.ascontiguousarray(pixels).tobytes()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.PhotometricInterpretation = "MONOCHROME2"
        kept.append((float(ds.ImagePositionPatient[2]), ds, pixels))
    if not kept:
        print(f"series {args.series} not found in {args.case}", file=sys.stderr)
        return 1
    thickness = float(kept[0][1].SliceThickness)
    if thickness > 1.25:
        print(f"series {args.series} slice thickness {thickness} mm > 1.25 mm", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for index, (_, ds, pixels) in enumerate(sorted(kept, key=lambda item: item[0]), start=1):
        path = out / f"{index:05d}.dcm"
        ds.save_as(path, enforce_file_format=True)
        # Round-trip check: the rewritten file must carry the identical pixel array.
        if not np.array_equal(pydicom.dcmread(path).pixel_array, pixels):
            raise RuntimeError(f"pixel mismatch after rewrite: {path}")
    print(f"wrote {len(kept)} slices ({thickness} mm) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
