"""Physics invariants of the chiral colour-dielectric model.

`run_full_check.py` is the single entry point (CLAUDE.md section 13):

    python -m eos.ccdm.verify.run_full_check            # everything
    python -m eos.ccdm.verify.run_full_check --no-csc   # skip the paired states
    python -m eos.ccdm.verify.run_full_check --onset    # + the onset scans
"""
from eos.ccdm.verify.run_full_check import run_all

__all__ = ["run_all"]
