# dd2's NMP inversion misses its targets — regression, or a tolerance asserting below the noise?

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

Six suite failures reduce to one function, `eos/dd2/nmp.py::invert_nmp`, not
reaching its target:

    test/dd2/test_api.py::test_inversion_without_Q_sat_predicts_it
    test/dd2/test_api.py::test_inversion_with_Q_sat_still_imposes_it
    test/dd2/test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion
    test/tov/test_solver_fast_robustness.py  x3  (build_parametrization
        returns None, so the test dies on `'NoneType' has no attribute n_sat`)

A seventh symptom is the `dd2` baseline's `nmp.K_sat` / `nmp.Q_sat` /
`nmp.K_sym` drift. Same function, same cause.

**It is not a regression.** Measured, not assumed:

- The three failure shapes are **bit-identical at every one of the 13 commits**
  that touched dd2's NMP path since `72881cf` created `eos/dd2/nmp.py`, up to
  and including HEAD. Walked in a detached worktree with a standalone repro
  that imports `eos/` only, since `test/` is gitignored and cannot be walked.
  At every commit: predicted Q_sat off by 51.5090, (240, 300) not recovered by
  restarts, (250, 100, 30) failing at isoscalar residual 8.12e-02.
- `compute_nmp` is deterministic run to run, and unchanged by
  `NUMBA_DISABLE_JIT=1` or by pinning threads to one.
- The stack did not move: numpy 2.3.5, scipy 1.17.0, Python 3.14.2, all
  installed Feb 2026 — **before** every `test/baseline/*.npz` (Aug 2026).

So no committed change and no library change produced this. What is left is
that the tests and the `.npz` were produced against a state that is not any
commit on this line — and `test/` being gitignored (`.gitignore:75`) is
exactly the hazard the map already records under "Several real fixes now live
outside version control".

**Two of the three shapes are a tolerance below the noise floor.** Sweeping
the stencil step at fixed code:

    stored dd2.npz    Q_sat 168.6525    K_sat 242.7240507853
    h = 5e-5          Q_sat 166.2932    K_sat 242.7242622586
    h = 1e-4 (ship)   Q_sat 169.0034    K_sat 242.7240212246
    h = 2e-4          Q_sat 168.7631    K_sat 242.7241263927
    h = 3e-4          Q_sat 168.7828    K_sat 242.7242150119
    h = 5e-4          Q_sat 168.7619    K_sat 242.7245619593
    h = 1e-3          Q_sat 168.6947    K_sat 242.7261959878

Both the stored 168.65 and the current 169.00 sit inside that band, and the
**shipped default h = 1e-4 is the outlier** against the 2e-4..1e-3 cluster —
precisely what `nmp.py`'s own docstring says ("the h = 1e-4 step used here and
in the forward map is just past the truncation/roundoff optimum (~3e-4 to
1e-3): Q_sat carries ~0.1 MeV of stencil noise at h = 1e-4"). Yet
`test_api.py` asserts a predicted Q_sat to `abs=0.2` and `test/baseline/`
pins it at `rtol=1e-10`.

**The third shape is NOT noise and is the real question.** The default 5x5
closure, given the DD2 point's own six NMPs, returns a parametrization whose
Q_sat is **117.49 against the seed's 169.00** — off by 51.5, and
self-consistently so: the round trip `compute_nmp(inv)["Q_sat"]` is 117.25,
agreeing with the prediction, while the six imposed NMPs round-trip to
3.4e-05. The inversion is not failing; it is landing in a different basin.
`nmp.py:70` predicts exactly this ("The residual surface has a spurious basin
in which the cross-constraint is satisfied but Q_sat is wrong, and which basin
a solve lands in is a property of where it started").

So the questions, in order:

1. **Should the default 5x5 closure, seeded from the published DD2 couplings
   and given DD2's own NMPs, be required to return to DD2's own basin?** The
   test asserts it must. The module docstring says which basin you land in is
   a property of the seed. One of the two is wrong, and it is a physics
   ruling, not a tolerance ruling.
2. If yes: is the fix a better seed, the pin (`PINNED_COEFF = "c_omega"`), or
   `N_RESTARTS`? Note the restart loop runs ONLY on a gate miss, and here the
   gate is MET (`ok=True`) — so restarts never fire on this case at all.
3. **What tolerance is honest for a third finite difference of the output of a
   nonlinear solve?** CLAUDE.md §12 forbids loosening a tolerance to make a
   test pass, so this needs the answer "here is the measured noise, and the
   tolerance should have been that all along", with the measurement above as
   the evidence — or the h default moves to the optimum the docstring names,
   in BOTH the forward map and the inverse together (`nmp.py:85` requires
   that), and the baseline is regenerated.
4. **Does `Q_sat` belong in a frozen `rtol=1e-10` baseline at all?** This is
   the same question ticket 40 settled for `mu_S` and `mu_e`, and the map
   already lists it open under "Whether an underdetermined potential belongs
   in a frozen baseline at all". A quantity with a documented 0.1+ MeV noise
   floor pinned at ten digits is the same defect wearing a different hat.

Resolved when each of the four is ruled, the ruled fixes applied, and the
added-failure count reported. **Blocks nothing formally, but ticket 44 renames
`compute_nmp` / `invert_nmp` and everything around them**, so running 44 first
means measuring its added-failure count against seven known-bad tests in the
exact code being renamed.

## Repro

`test/` is gitignored, so the repro imports `eos/` only and can be walked
across history:

    from eos.dd2.parameters import Parametrization
    from eos.dd2.nmp import compute_nmp, invert_nmp

    nmp = compute_nmp(Parametrization.from_dd2_defaults())
    six = {k: nmp[k] for k in ("n_sat", "E_sat", "m_eff_ratio",
                               "K_sat", "E_sym", "L_sym")}
    inv, status = invert_nmp(six)
    assert status.ok                                    # passes
    assert abs(status.predictions["Q_sat"] - nmp["Q_sat"]) < 0.2   # 51.5090
