"""
Backend array-namespace switch point for the DD2 physics kernel.

eos_ref uses numpy; the M9 eos_fast backend swaps in jax.numpy here.
Physics modules must import `xp` from this module, never numpy directly.
"""
import numpy as xp  # noqa: F401
