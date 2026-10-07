"""Source-hash sidecar for along-tract FA/MD products (casemask lineage).

scripts/make_scalar_maps.sh records the DWI and case-mask sha256 next to
the casemask products. A rerun whose sources drifted must refuse to reuse
those products unless ``--force``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .profile import sha256_file

SIDECAR_NAME = "casemask.sources.json"
PRODUCTS = (
    "mask_casemask.mif",
    "dt_casemask.mif",
    "fa_casemask.nii.gz",
    "md_casemask.nii.gz",
)
SOURCE_KEYS = ("dwi_sha256", "mask_sha256")


class StaleScalarMaps(ValueError):
    """Casemask products do not match the current DWI/mask hashes."""


def sidecar_path(out_dir: str) -> str:
    return os.path.join(out_dir, SIDECAR_NAME)


def current_sources(dwi_path: str, mask_path: str) -> dict[str, str]:
    return {
        "dwi_sha256": sha256_file(dwi_path),
        "mask_sha256": sha256_file(mask_path),
        "dwi_path": dwi_path,
        "mask_path": mask_path,
    }


def products_exist(out_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(out_dir, name)) for name in PRODUCTS)


def read_sidecar(out_dir: str) -> dict[str, Any] | None:
    path = sidecar_path(out_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_sidecar(out_dir: str, dwi_path: str, mask_path: str) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    record = current_sources(dwi_path, mask_path)
    path = sidecar_path(out_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    return record


def mismatch_lines(recorded: dict[str, Any] | None, current: dict[str, str]) -> list[str]:
    if recorded is None:
        return ["sidecar missing"]
    lines = []
    for key in SOURCE_KEYS:
        got = recorded.get(key)
        want = current[key]
        if got != want:
            lines.append(f"{key}: recorded {got} != current {want}")
    return lines


def decide(
    out_dir: str,
    dwi_path: str,
    mask_path: str,
    *,
    force: bool = False,
) -> str:
    """Return ``reuse`` or ``build``. Raise ``StaleScalarMaps`` when hashes drifted."""
    sources = current_sources(dwi_path, mask_path)
    if force or not products_exist(out_dir):
        return "build"
    recorded = read_sidecar(out_dir)
    bad = mismatch_lines(recorded, sources)
    if bad:
        raise StaleScalarMaps(
            "casemask products are stale vs current DWI/mask; "
            "rebuild with --force. " + "; ".join(bad)
        )
    return "reuse"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 4 or args[0] not in ("decide", "write"):
        sys.stderr.write("usage: scalar_maps decide|write OUT DWI MASK [--force]\n")
        return 2
    cmd, out_dir, dwi, mask = args[0], args[1], args[2], args[3]
    force = "--force" in args[4:]
    if cmd == "write":
        rec = write_sidecar(out_dir, dwi, mask)
        print(f"sidecar dwi_sha256={rec['dwi_sha256']} mask_sha256={rec['mask_sha256']}")
        return 0
    try:
        action = decide(out_dir, dwi, mask, force=force)
    except StaleScalarMaps as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2
    print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
