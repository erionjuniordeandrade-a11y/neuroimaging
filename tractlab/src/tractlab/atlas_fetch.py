"""Fetch population atlases with sha256-pinned integrity (A2).

Schaefer-200 / Yeo-7 from the CBIG release mirror. Files are cached under
``~/.cache/tractlab/atlases/`` and never vendored into the repo (publication
posture: runtime fetch, hash-checked).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

# CBIG Schaefer2018 200 parcels / 7 networks, FSL MNI152 1mm (master pin via sha256)
SCHAEFER_200_7_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/MNI/"
    "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
)
# Official order LUT (parcel index → network name string)
SCHAEFER_200_7_LUT_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/MNI/freeview_lut/"
    "Schaefer2018_200Parcels_7Networks_order.txt"
)

# Pinned after first successful download — update only with deliberate review.
# Filled at first run if env TRACTLAB_ATLAS_PIN=1 recomputes; else these must match.
SCHAEFER_200_7_SHA256 = os.environ.get(
    "TRACTLAB_SCHAEFER200_SHA256",
    # CBIG master Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz
    "a47808f907f175f2e83f2c941bc68a766a8991b238e98457e9d27a8760ea21ec",
)
SCHAEFER_200_7_LUT_SHA256 = os.environ.get(
    "TRACTLAB_SCHAEFER200_LUT_SHA256",
    # freeview_lut/Schaefer2018_200Parcels_7Networks_order.txt
    "42927810db2d272189c63c08927f8b2b4c9eac402f26e08e80d5ce70e7dd29f9",
)

CACHE_DIR = Path(os.path.expanduser("~/.cache/tractlab/atlases"))

# Yeo-7 didactic network order (1-indexed as in Schaefer order files)
YEO7_NETWORKS: dict[int, str] = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Ventral Attention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — pinned URL constants only
    tmp.replace(dest)


def fetch_schaefer200_yeo7(
    *,
    cache_dir: Path | None = None,
    expected_nii_sha: str | None = None,
    expected_lut_sha: str | None = None,
) -> tuple[Path, Path]:
    """Return (nii_path, lut_path). Refuse on hash mismatch when pin provided."""
    cache = cache_dir or CACHE_DIR
    nii = cache / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
    lut = cache / "Schaefer2018_200Parcels_7Networks_order.txt"
    if not nii.is_file():
        _download(SCHAEFER_200_7_URL, nii)
    if not lut.is_file():
        _download(SCHAEFER_200_7_LUT_URL, lut)

    nii_sha = sha256_file(nii)
    lut_sha = sha256_file(lut)
    exp_nii = expected_nii_sha or SCHAEFER_200_7_SHA256
    exp_lut = expected_lut_sha or SCHAEFER_200_7_LUT_SHA256
    if exp_nii and nii_sha != exp_nii:
        raise ValueError(
            f"Schaefer NIfTI sha256 mismatch: got {nii_sha}, expected {exp_nii}"
        )
    if exp_lut and lut_sha != exp_lut:
        raise ValueError(
            f"Schaefer LUT sha256 mismatch: got {lut_sha}, expected {exp_lut}"
        )
    return nii, lut


def load_schaefer_lut(lut_path: Path) -> dict[int, dict]:
    """Parse Schaefer order file → {label: {name, network_id, network_name, hemi}}.

    Format lines like:
      1 7Networks_LH_Vis_1  ...
    Network id inferred from Yeo order substring match; hemi from LH/RH.
    """
    out: dict[int, dict] = {}
    text = lut_path.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            lab = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        hemi = "?"
        if "_LH_" in name or name.startswith("LH_"):
            hemi = "L"
        elif "_RH_" in name or name.startswith("RH_"):
            hemi = "R"
        net_id = _network_id_from_name(name)
        out[lab] = {
            "name": name,
            "network_id": net_id,
            "network_name": YEO7_NETWORKS.get(net_id, "Unknown"),
            "hemi": hemi,
        }
    if len(out) != 200:
        raise ValueError(f"Schaefer LUT expected 200 labels, got {len(out)}")
    return out


def _network_id_from_name(name: str) -> int:
    # Order matters: more specific tokens first
    tokens = [
        (1, ("Vis", "Visual")),
        (2, ("SomMot", "Somatomotor")),
        (3, ("DorsAttn", "DorsalAttention", "DorsAttn")),
        (4, ("SalVentAttn", "VentAttn", "SalVentAttn", "VentralAttention")),
        (5, ("Limbic",)),
        (6, ("Cont", "Frontoparietal", "Control")),
        (7, ("Default", "DMN")),
    ]
    for nid, keys in tokens:
        for k in keys:
            if k in name:
                return nid
    return 0


def label_to_network_map(lut: dict[int, dict]) -> dict[int, int]:
    """parcel label → network id (1–7)."""
    return {lab: int(meta["network_id"]) for lab, meta in lut.items()}
