"""Inline the NiiVue bundle and the viewer2 sources into viewer2/template.html."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "src"
VENDOR = HERE.parent / "vendor"
NIIVUE = VENDOR / "niivue-0.69.0.umd.js"
LICENSE = VENDOR / "NIIVUE_LICENSE.txt"
PARTS = (
    "styles.css",
    "i18n.js",
    "state.js",
    "decode.js",
    "view.js",
    "render3d.js",
    "tracts.js",
    "masks.js",
    "anatomy.js",
    "tools.js",
    "trajectory.js",
    "seg-prompts.js",
    "tour.js",
    "save.js",
    "ui.js",
)


def niivue_block() -> str:
    bundle = NIIVUE.read_text(encoding="utf-8")
    for forbidden in ("</script", "<!--", "CAPSULE_PAYLOAD"):
        if forbidden in bundle:
            raise ValueError(f"NiiVue bundle contains {forbidden!r}; cannot inline safely")
    license_text = LICENSE.read_text(encoding="utf-8").strip().replace("*/", "* /")
    notice = ("/*! NiiVue 0.69.0 — Copyright (c) NiiVue contributors — BSD 2-Clause License\n\n"
              + license_text + "\n*/\n")
    return notice + bundle


def build() -> str:
    html = (SOURCE / "index.html").read_text(encoding="utf-8")
    marker = "<!--INLINE:niivue-->"
    if html.count(marker) != 1:
        raise ValueError(f"expected one {marker}")
    for name in PARTS:
        part = f"<!--INLINE:{name}-->"
        if html.count(part) != 1:
            raise ValueError(f"expected one {part}")
        text = (SOURCE / name).read_text(encoding="utf-8")
        if "</script" in text or "<!--" in text:
            raise ValueError(f"{name} contains a sequence that breaks inline script parsing")
        html = html.replace(part, text)
    html = html.replace(marker, niivue_block())
    if html.count("<!--CAPSULE_PAYLOAD-->") != 1:
        raise ValueError("expected exactly one payload marker")
    return html


if __name__ == "__main__":
    (HERE / "template.html").write_text(build(), encoding="utf-8")
