"""Compatibility shim for structured PDDL evaluation.

Provides EvalPddlSt expected by `agentboard/tasks/__init__.py`.
This file delegates to the existing EvalPddl implementation as a fallback.
"""
from .pddl import EvalPddl as EvalPddlSt

__all__ = ["EvalPddlSt"]
