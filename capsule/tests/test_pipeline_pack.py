import numpy as np
import pytest

from capsule.pack import read_capsule, write_capsule


def _manifest():
    return {"schema": "case-capsule/1", "grid": {"dims": [3, 4, 2]},
            "volumes": [{"blob": "ct", "dtype": "int16"}], "masks": []}


@pytest.mark.parametrize("placeholder_count", [0, 2])
def test_template_requires_exactly_one_placeholder(tmp_path, placeholder_count):
    template = tmp_path / "template.html"
    template.write_text("<html>" + "<!--CAPSULE_PAYLOAD-->" * placeholder_count + "</html>")
    with pytest.raises(ValueError, match="exactly one"):
        write_capsule(template, tmp_path / "output.capsule.html", _manifest(), {"ct": np.zeros((2, 4, 3), dtype=np.int16)})
    assert not (tmp_path / "output.capsule.html").exists()


def test_blob_round_trip_and_no_overwrite(tmp_path):
    template = tmp_path / "template.html"
    template.write_text("<!doctype html><!--CAPSULE_PAYLOAD-->")
    target = tmp_path / "output.capsule.html"
    expected = np.arange(24, dtype=np.int16).reshape((2, 4, 3))
    write_capsule(template, target, _manifest(), {"ct": expected})
    decoded, blobs = read_capsule(target)
    assert decoded == _manifest()
    np.testing.assert_array_equal(blobs["ct"], expected)
    with pytest.raises(FileExistsError):
        write_capsule(template, target, _manifest(), {"ct": expected})
