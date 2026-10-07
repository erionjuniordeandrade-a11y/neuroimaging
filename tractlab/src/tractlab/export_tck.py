"""Write streamlines to a MRtrix .tck file for download (research export)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram


def write_tck(streamlines: list[np.ndarray], out_path: str | Path) -> int:
    """Write streamlines to ``out_path``. Returns count written."""
    kept = []
    for s in streamlines:
        arr = np.asarray(s, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] == 3:
            kept.append(arr)
    if not kept:
        raise ValueError("no streamlines to export")
    tg = Tractogram(kept, affine_to_rasmm=np.eye(4))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.streamlines.save(tg, str(path))
    return len(kept)
