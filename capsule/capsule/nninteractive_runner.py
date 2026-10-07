"""Standalone subprocess adapter for nnInteractive; exchange files are temporary NPZ/JSON only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import torch

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.model_management import ensure_model_available, get_default_model_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    args = parser.parse_args(argv)

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    model_id = get_default_model_id()
    model_folder = Path(ensure_model_available(model_id))
    license_file = model_folder / "LICENSE"
    weights_licence = license_file.read_text(encoding="utf-8").splitlines()[0].strip()
    session = nnInteractiveInferenceSession(
        device=torch.device(args.device), use_torch_compile=False, verbose=False,
        torch_n_threads=os.cpu_count(), do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(str(model_folder))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as data:
        for index, prompt in enumerate(request["prompts"]):
            if index:
                session.reset_interactions()
            image = data[prompt["image_key"]].astype(np.float32, copy=False)
            session.set_image(image[None])
            target = torch.zeros(image.shape, dtype=torch.uint8)
            session.set_target_buffer(target)
            for point in prompt["points_kji"]:
                session.add_point_interaction(tuple(point["kji"]), include_interaction=point["positive"])
            np.savez_compressed(output_dir / f"mask_{index:04d}.npz", mask=(target.numpy() > 0).astype(np.uint8))

    try:
        version = importlib.metadata.version("nnInteractive")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    Path(args.result).write_text(json.dumps({
        "model_id": str(model_id), "version": version, "weights_licence": weights_licence,
    }, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
