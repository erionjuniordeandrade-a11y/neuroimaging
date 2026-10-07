"""Inline the dependency-free viewer source into its distributable HTML template."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "src"
PARTS = (
    "styles.css",
    "i18n.js",
    "state.js",
    "decode.js",
    "mpr.js",
    "render3d.js",
    "segmentation.js",
    "tools.js",
    "tour.js",
    "save.js",
    "ui.js",
)


def build() -> str:
    html = (SOURCE / "index.html").read_text(encoding="utf-8")
    for name in PARTS:
        marker = f"<!--INLINE:{name}-->"
        if html.count(marker) != 1:
            raise ValueError(f"expected one {marker}")
        html = html.replace(marker, (SOURCE / name).read_text(encoding="utf-8"))
    if html.count("<!--CAPSULE_PAYLOAD-->") != 1:
        raise ValueError("expected exactly one payload marker")
    return html


if __name__ == "__main__":
    (HERE / "template.html").write_text(build(), encoding="utf-8")
