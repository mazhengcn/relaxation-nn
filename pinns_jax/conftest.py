"""
Stub out heavy optional dependencies (tensorflow, clu) that are not required
for the unit tests in this package but are imported at the module level by
train.py.
"""

import importlib.machinery
import sys
import types
from unittest.mock import MagicMock


def _make_module_stub(name: str) -> types.ModuleType:
    """Create a bare ModuleType stub with a proper __spec__ so that
    importlib.util.find_spec() does not raise ValueError when inspecting it."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


# tensorflow stubs — needs proper __spec__ because orbax calls
# importlib.util.find_spec('tensorflow') to detect whether TF is installed.
_TF_MODULES = [
    "tensorflow",
    "tensorflow.core",
    "tensorflow.core.framework",
    "tensorflow.python",
    "tensorflow.python.framework",
]
for _mod in _TF_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = _make_module_stub(_mod)

# clu stubs — only need to prevent ImportError, plain MagicMock is fine.
_CLU_MODULES = [
    "clu",
    "clu.metric_writers",
    "clu.metric_writers.tf",
    "clu.metric_writers.tf.summary_writer",
    "clu.metric_writers.summary_writer",
    "clu.periodic_actions",
]
for _mod in _CLU_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
