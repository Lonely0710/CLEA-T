"""Compatibility shim exposing EvalBabyaiTrain.

Some parts of the code import `EvalBabyaiTrain` from `tasks`. The project
only contains `babyai.py` (EvalBabyai). Provide a small alias so imports
succeed without changing the rest of the codebase.
"""
from .babyai import EvalBabyai as EvalBabyaiTrain

__all__ = ["EvalBabyaiTrain"]
