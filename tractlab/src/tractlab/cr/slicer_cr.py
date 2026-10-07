"""3D Slicer ``--python-script`` loader for a manifest-driven CR peel case.

Research visualization only · enhancing vessels · not navigation, not a device
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import qt
import slicer
import vtk


HONESTY_WALL = "Research visualization only · enhancing vessels · not navigation, not a device"


def _case_root_from_argv() -> Path:
    for raw in reversed(sys.argv[1:]):
        if raw.startswith("-"):
            continue
        candidate = Path(raw).expanduser()
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    raise RuntimeError("usage: Slicer --python-script slicer_cr.py <case_root>")


def _resolve_under(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{label} must be a non-empty relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes its root") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{label} not found: {resolved}")
    return resolved


def _depth_tag(depth: float) -> str:
    if math.isclose(depth, round(depth), abs_tol=1e-8):
        return f"d{int(round(depth)):02d}"
    text = f"{depth:.6f}".rstrip("0").rstrip(".")
    whole, fraction = text.split(".")
    return f"d{int(whole):02d}p{fraction}"


def _configure_colours(model) -> None:
    display = model.GetDisplayNode()
    display.SetActiveScalar("RGB", vtk.vtkAssignAttribute.POINT_DATA)
    display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDirectMapping)
    display.SetScalarVisibility(True)
    display.SetColor(1.0, 1.0, 1.0)
    display.SetAmbient(0.30)
    display.SetDiffuse(0.72)
    display.SetSpecular(0.05)
    display.SetPower(10)
    display.SetBackfaceCulling(False)
    display.SetVisibility2D(True)
    display.SetSliceIntersectionThickness(2)
    display.SetVisibility(False)


def _add_watermark(renderer):
    annotation = vtk.vtkCornerAnnotation()
    annotation.SetText(vtk.vtkCornerAnnotation.UpperLeft, HONESTY_WALL)
    annotation.SetLinearFontScaleFactor(1.5)
    annotation.SetNonlinearFontScaleFactor(0.25)
    annotation.SetMaximumFontSize(16)
    text = annotation.GetTextProperty()
    text.SetColor(0.95, 0.95, 0.95)
    text.SetBold(True)
    renderer.AddViewProp(annotation)
    return annotation


CASE_ROOT = _case_root_from_argv()
CR_ROOT = CASE_ROOT / "cr"
MANIFEST = json.loads((CASE_ROOT / "manifest.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((CR_ROOT / "summary.json").read_text(encoding="utf-8"))
if MANIFEST.get("deid") is not True:
    raise RuntimeError("manifest field 'deid' must be true")
if SUMMARY.get("schema") != "tractlab.cr.summary/1":
    raise RuntimeError("unsupported or missing CR summary schema")
if SUMMARY.get("honesty_wall") != HONESTY_WALL:
    raise RuntimeError("CR summary honesty wall mismatch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Refuse stale/tampered bundles: every artifact the receipt names must hash-match, and
# nothing that is not in the receipt is loaded (a mesh without a receipt has no lineage).
_receipt_outputs = SUMMARY.get("outputs") or {}
if not isinstance(_receipt_outputs, dict) or not _receipt_outputs:
    raise RuntimeError("CR summary lists no outputs; refusing to load unreceipted meshes")
for _rel, _expected in _receipt_outputs.items():
    _path = _resolve_under(CASE_ROOT, _rel, f"receipt output {_rel}")
    _actual = _sha256(_path)
    if _actual != _expected:
        raise RuntimeError(f"CR output hash mismatch for {_rel}: receipt {_expected[:12]}… disk {_actual[:12]}… "
                           "(stale or tampered bundle — re-run the CLI)")
_receipted = {Path(rel).name for rel in _receipt_outputs}

input_spec = MANIFEST["inputs"]["t1c_unstripped"]
input_root = Path(input_spec["root"]).expanduser().resolve()
t1_path = _resolve_under(input_root, input_spec["path"], "inputs.t1c_unstripped.path")

layout_manager = slicer.app.layoutManager()
layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
volume = slicer.util.loadVolume(str(t1_path))
if volume is None:
    raise RuntimeError(f"Slicer could not load T1: {t1_path}")
volume.SetName("t1c_unstripped")
slicer.util.setSliceViewerLayers(background=volume)

CHANNELS = {
    "grey": "grey reformat (single sample)",
    "vmax": "±1 mm slab max (overlay channel shown as image — NOT the honest reformat)",
    "vessel": "enhancing vessels — research overlay",
}

_mesh_paths = []
for item in SUMMARY["per_depth"]:
    depth = float(item["depth_mm"])
    tag = _depth_tag(depth)
    paths = {}
    for channel in CHANNELS:
        path = CR_ROOT / f"{tag}_{channel}.ply"
        if path.name in _receipted and path.is_file():
            paths[channel] = path
    if "grey" not in paths or "vessel" not in paths:
        raise RuntimeError(f"missing receipted grey/vessel mesh for {depth:g} mm")
    _mesh_paths.append((depth, paths))

models = {}
for depth, paths in _mesh_paths:
    loaded = {}
    for channel, path in paths.items():
        model = slicer.util.loadModel(str(path))
        if model is None:
            raise RuntimeError(f"Slicer could not load CR mesh {path.name}")
        model.SetName(f"{CHANNELS[channel]} · {depth:g} mm")
        _configure_colours(model)
        loaded[channel] = model
    models[depth] = loaded
_available_channels = [c for c in CHANNELS if all(c in pair for pair in models.values())]

if not models:
    raise RuntimeError("CR summary contains no depths")

_watermarks = []
for index in range(layout_manager.threeDViewCount):
    renderers = layout_manager.threeDWidget(index).threeDView().renderWindow().GetRenderers()
    _watermarks.append(_add_watermark(renderers.GetFirstRenderer()))
for name in layout_manager.sliceViewNames():
    renderers = layout_manager.sliceWidget(name).sliceView().renderWindow().GetRenderers()
    _watermarks.append(_add_watermark(renderers.GetFirstRenderer()))

_channel = "grey"
_current_depth = None


def _nearest_depth(depth: float) -> float:
    requested = float(depth)
    return min(models, key=lambda available: abs(available - requested))


def show_depth(d):
    """Show the available depth nearest ``d`` millimetres."""
    global _current_depth
    selected = _nearest_depth(float(d))
    for depth, pair in models.items():
        for channel, model in pair.items():
            model.GetDisplayNode().SetVisibility(depth == selected and channel == _channel)
    _current_depth = selected
    slicer.app.processEvents()
    print(f"CR depth {selected:g} mm; channel = {CHANNELS[_channel]}")
    return selected


def set_channel(channel: str):
    """'grey' (honest reformat, default) · 'vessel' (enhancing-vessel overlay) · 'vmax'
    (slab-max shown as the image — the prototype's look; for the owner's comparison only)."""
    global _channel
    if channel not in _available_channels:
        raise ValueError(f"channel must be one of {_available_channels}")
    _channel = channel
    return show_depth(_current_depth if _current_depth is not None else SUMMARY["marker_depth_mm"])


def set_tint(on: bool):
    """Back-compat: True = enhancing-vessel overlay, False = grey reformat."""
    if not isinstance(on, bool):
        raise TypeError("set_tint(on) requires a bool")
    return set_channel("vessel" if on else "grey")


def screenshot(path):
    """Save the primary 3-D view, including the honesty-wall watermark."""
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = CASE_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    view = layout_manager.threeDWidget(0).threeDView()
    view.forceRender()
    slicer.app.processEvents()
    if not view.grab().save(str(destination)):
        raise RuntimeError(f"could not save screenshot: {destination}")
    print(f"CR screenshot: {destination}")
    return str(destination)


def contact_sheet(depths):
    """Render the requested depths into ``cr/contact-sheet.png`` and return its path."""
    requested = list(depths)
    if not requested:
        raise ValueError("contact_sheet(depths) requires at least one depth")
    previous = _current_depth
    view = layout_manager.threeDWidget(0).threeDView()
    images = []
    labels = []
    for depth in requested:
        selected = show_depth(depth)
        view.forceRender()
        slicer.app.processEvents()
        images.append(view.grab())
        labels.append(f"{selected:g} mm")
    width, height = images[0].width(), images[0].height()
    columns = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / columns))
    sheet = qt.QImage(width * columns, height * rows, qt.QImage.Format_ARGB32)
    sheet.fill(qt.QColor(5, 5, 8))
    painter = qt.QPainter(sheet)
    painter.setPen(qt.QColor(255, 255, 255))
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * width
        y = (index // columns) * height
        painter.drawImage(x, y, image)
        painter.drawText(x + 12, y + 24, label)
    painter.end()
    destination = CR_ROOT / "contact-sheet.png"
    if not sheet.save(str(destination)):
        raise RuntimeError(f"could not save contact sheet: {destination}")
    if previous is not None:
        show_depth(previous)
    print(f"CR contact sheet: {destination}")
    return str(destination)


show_depth(float(SUMMARY["marker_depth_mm"]))
layout_manager.threeDWidget(0).threeDView().resetFocalPoint()
print(
    f"CR ready: show_depth(d), set_channel({_available_channels}), set_tint(True|False), "
    "screenshot(path), contact_sheet(depths)"
)
