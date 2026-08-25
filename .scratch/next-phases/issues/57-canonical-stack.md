# Which Python/numpy/scipy stack is canonical, and do the baselines regenerate on it?

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

[Ticket 47](47-dd2-nmp-inversion.md) established that this machine carries two
Python stacks, that `pyproject.toml:5` (`requires-python = ">=3.9"`) admits
both and picks neither, and that **all fourteen recorded suite failures are
that undeclared choice**:

| | anaconda3 (`python`, `python3`) | python.org 3.14 |
|---|---|---|
| Python | 3.9.7 | 3.14.2 |
| numpy | 1.26.4 | 2.3.5 |
| scipy | 1.13.1 | 1.17.0 |
| numba | 0.60.0 | 0.63.1 |
| matplotlib | 3.4.3 | 3.10.9 |

Every `test/baseline/*.npz` was produced on anaconda 3.9; every file in
`output/_audit/` was produced on 3.14. The fourteen node ids that fail on 3.14
pass on 3.9 in one invocation (`14 passed in 182.66s`).

**The ruling decides whether 13 golden reference files move**, so §12 makes it
an authorization question, not a preference.

### (A) anaconda 3.9 is canonical

The suite is green today. Every audit file in `output/_audit/` is a measurement
from the wrong interpreter and its failure counts must be retracted — including
the map's "14 failed" Suite-status block and every "0 added failures" claim
measured against it. No golden reference moves; no test changes.

Cost: Python 3.9 reached end-of-life in October 2025. And scipy 1.13's
`root(..., method="hybr")` returns the seed unchanged on dd2's 5x5 isoscalar
closure (residual 2.2e-3, zero iterations) where 1.17 solves it to 6.7e-11 —
so this option pins the project to a root-finder that does not converge on the
repository's own model and is not detected doing it.

### (B) python.org 3.14 is canonical

Correct long-term and matches where the audits already run. Cost, all of it
§12-gated:

- all 13 `test/baseline/*.npz` regenerate on 3.14;
- `test_api.py:127` and `:143`'s `abs=0.2` on Q_sat are re-derived from a noise
  floor MEASURED on that stack, not loosened to fit (ticket 47 Q3, and
  `nmp.py:85`'s requirement that `h` move in the forward and inverse maps
  together);
- `test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion`'s
  (K_sat, Q_sat) = (240, 300) premise is re-measured — the seed-limited/
  infeasible distinction it guards is exactly what changed;
- `test/tov/test_solver_fast_robustness.py`'s three cases get a sample the 6x6
  closure can actually reach on 3.14 (today it returns `None` at isoscalar
  residual 8.12e-02);
- the DD2 published NMP/TOV values and the CompOSE HS(DD2) slices are
  re-checked, since those are §12 ground truth independent of any `.npz`.

### Not optional either way

Pin the dev stack somewhere tracked, so the next session cannot rediscover
this: which interpreter produces the baselines and the audits. `pyproject.toml`
declares a floor for *consumers* and that is correct; what is missing is a
statement of the stack the golden references are made on. Note `test/` is
gitignored, so it cannot live there.

## Repro

    /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "
    import sys; sys.path.insert(0,'.')
    from eos.dd2.parameters import Parameters
    from eos.dd2.nmp import compute_nmp
    print(repr(compute_nmp(Parameters.default())['Q_sat']))"      # 169.00335695659044

    python -c "
    from eos.dd2.parameters import Parameters
    from eos.dd2.nmp import compute_nmp
    print(repr(compute_nmp(Parameters.default())['Q_sat']))"      # 168.65250604853313
