"""tractlab package.

Under ~/fsl/bin/python neuro_core is not installed, so the monorepo copy is put on
the path. In the uv environment neuro_core is installed and this does nothing.
"""

import importlib.util as _util
import os as _os
import sys as _sys

if _util.find_spec("neuro_core") is None:
    _core = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "packages", "neuro-core", "src")
    if _os.path.isdir(_core):
        _sys.path.insert(0, _os.path.realpath(_core))
