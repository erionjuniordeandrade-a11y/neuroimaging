"""Shared manifest path containment checks."""

from __future__ import annotations

import os


def resolve_case_path(case_root: str, rel: str, *, what: str) -> str:
    """Resolve one manifest-relative path without allowing root escape."""
    root = os.path.realpath(case_root)
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        raise ValueError(f"{what}: path must be a non-empty relative path")
    path = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath((root, path)) != root:
        raise ValueError(f"{what}: path escapes case_root")
    return path
