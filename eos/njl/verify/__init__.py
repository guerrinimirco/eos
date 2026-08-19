"""Physics invariants of the NJL model.

`run_full_check.py` is the single entry point (CLAUDE.md section 13):

    python -m eos.njl.verify.run_full_check            # everything
    python -m eos.njl.verify.run_full_check --no-csc   # skip the paired states
    python -m eos.njl.verify.run_full_check --sound    # + the sound-speed sweep
"""
from eos.njl.verify.run_full_check import run_all

__all__ = ["run_all"]
