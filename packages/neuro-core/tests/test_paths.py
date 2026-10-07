import os

import pytest

from neuro_core.paths import resolve_case_path


def test_resolves_inside_root(tmp_path):
    assert resolve_case_path(str(tmp_path), "a/b.json", what="x") == os.path.join(
        os.path.realpath(tmp_path), "a", "b.json")


@pytest.mark.parametrize("rel", ["", "/etc/passwd", "../escape", "a/../../escape"])
def test_refuses_escape_and_absolute(tmp_path, rel):
    with pytest.raises(ValueError):
        resolve_case_path(str(tmp_path), rel, what="x")


def test_refuses_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        resolve_case_path(str(root), "link/outside", what="x")
