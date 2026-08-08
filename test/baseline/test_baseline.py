"""Regression net: every model must still compute exactly what it computed
before the refactor began.

Each test re-runs the cases in `generate_baseline.py` and compares against the
stored `<model>.npz` at rtol = 1e-10. That tolerance is not a physics
statement — it is the assertion that a refactor is a NO-OP. A move, a rename,
a re-layering or a de-duplication must not shift the tenth significant digit
of anything.

A failure here means one of two things:

  1. The refactor changed physics by accident. Fix the code.
  2. The change was intentional (a bug fix, a tightened convergence gate).
     Then regenerate ONLY the affected model,

         python test/baseline/generate_baseline.py <model>

     in its own commit, and quote the before/after delta in the commit body.
     Never regenerate everything to make a red suite go green.

If the .npz files are missing, the test skips rather than fails: they are
gitignored along with the rest of test/, so a fresh clone legitimately has
none. See the module docstring of generate_baseline.py for how to rebuild.
"""
import pathlib
import sys

import numpy as np
import pytest

# `generate_baseline` sits beside this file. pytest's default import mode
# happens to put that directory on sys.path, but saying so explicitly keeps
# the import working under `--import-mode=importlib` and from a plain
# `python -m pytest` in any directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from generate_baseline import CASES, flatten, path_for  # noqa: E402

RTOL = 1e-10


@pytest.mark.parametrize("model", sorted(CASES))
def test_baseline(model):
    stored_path = path_for(model)
    if not stored_path.exists():
        pytest.skip(f"no stored baseline at {stored_path}; "
                    f"run generate_baseline.py {model}")

    stored = np.load(stored_path)
    fresh = CASES[model]()

    missing = sorted(set(stored.files) - set(fresh))
    added = sorted(set(fresh) - set(stored.files))
    assert not missing, (
        f"{model}: {len(missing)} baseline quantities are no longer produced, "
        f"e.g. {missing[:5]}")
    assert not added, (
        f"{model}: {len(added)} quantities are new, e.g. {added[:5]}. "
        f"If the extra output is wanted, regenerate the baseline.")

    mismatched = []
    for key in stored.files:
        want = stored[key]
        got = np.asarray(fresh[key], dtype=float)
        if got.shape != want.shape:
            mismatched.append(f"{key}: shape {got.shape} != {want.shape}")
            continue
        if not np.allclose(got, want, rtol=RTOL, atol=0.0, equal_nan=True):
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.nanmax(np.abs((got - want) / np.where(want == 0,
                                                               np.nan, want)))
            mismatched.append(f"{key}: max rel. change {rel:.3e}")

    assert not mismatched, (
        f"{model}: {len(mismatched)} quantities changed at rtol={RTOL:g}\n  "
        + "\n  ".join(mismatched[:20]))


def test_flatten_keeps_convergence_flags():
    """A silently-unconverged point must not slip through as 'no change'.

    The solvers return `converged` as a numpy bool, which subclasses neither
    `bool` nor `np.number`; an isinstance check that misses that type drops
    every convergence flag, and a refactor could then mark every point
    unconverged with the baseline still green. Both spellings are pinned.
    Strings are skipped on purpose — an `error` *message* is prose and may
    legitimately be reworded.
    """
    class FakeResult:
        def __init__(self):
            self.converged = np.bool_(True)
            self.also_converged = True
            self.message = "some prose that may drift"
            self.n_B = 0.16
            self.composition = {"n": 0.14, "p": 0.02}

    flat = flatten(FakeResult())
    assert flat["converged"] == 1.0
    assert flat["also_converged"] == 1.0
    assert "message" not in flat
    assert flat["composition.p"] == 0.02


def test_stored_baselines_carry_convergence_flags():
    """Every model whose solvers report convergence has those flags frozen."""
    reporting = {"vmit", "zl", "alphabag", "sfho", "mixed", "zlvmit"}
    for model in sorted(reporting):
        path = path_for(model)
        if not path.exists():
            pytest.skip(f"no stored baseline for {model}")
        keys = np.load(path).files
        flags = [k for k in keys if k.split(".")[-1] == "converged"]
        assert flags, f"{model}: no convergence flags were captured"
