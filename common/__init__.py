import os

# Shim package so imports like `from common.registry import registry` resolve
# to the implementation under agentboard/common.

__all__ = []

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_agentboard_common = os.path.join(_root, "agentboard", "common")
if os.path.isdir(_agentboard_common):
    if _agentboard_common not in __path__:
        __path__.insert(0, _agentboard_common)
