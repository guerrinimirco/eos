# dd2's NMP inversion misses its targets — regression, or a tolerance asserting below the noise?

Type: grilling
Status: resolved
Assignee: session 9616271c
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


## Resolution

**A report, not a fix — and the ticket's central premise is false.** All six
failures, the seventh `dd2` baseline symptom, and the eight failures the map
assigns to [ticket 56](56-baseline-empty-sector-gate.md) are ONE cause: the
audits were run on an interpreter that did not produce the golden references.

### There are two Python stacks on this machine

| | anaconda3 (`python`, `python3`) | python.org 3.14 |
|---|---|---|
| Python | 3.9.7 | 3.14.2 |
| numpy | 1.26.4 | 2.3.5 |
| scipy | 1.13.1 | 1.17.0 |
| numba | 0.60.0 | 0.63.1 |
| matplotlib | 3.4.3 | 3.10.9 |

`python` in a bare shell resolves to anaconda 3.9; `output/_audit/` was
produced with `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`.
`pyproject.toml:5` says `requires-python = ">=3.9"`, which admits both and
picks neither. **That undeclared choice is the defect.** `eos` is not installed
into either site-packages, so `conftest.py`'s `sys.path` insert is what both
interpreters import — no shadowing is involved.

**Measured, decisively.** The fourteen failing node ids of
`pytest_after_ticket45.txt`, run as a single invocation on the anaconda stack:

    14 passed, 1 warning in 182.66s

and every one of them fails on 3.14. The forward map is bit-stable *within* a
stack and differs *between* them:

    compute_nmp(Parameters.default())
      3.9  / scipy 1.13.1   Q_sat 168.65250604853313   <- equals the stored dd2.npz
      3.14 / scipy 1.17.0   Q_sat 169.00335695659044   <- equals the audit

Repeated runs, `OMP_NUM_THREADS=1` and `NUMBA_DISABLE_JIT=1` all reproduce
168.65250604853313 exactly on 3.9. So the ticket's determinism checks were
sound but uninformative: they varied every axis except the one that mattered.
The install-date check compared the 3.14 packages against the `.npz` dates —
but BOTH stacks were installed in Feb 2026 and both predate every `.npz`
(Aug 2026), so that check could not have discriminated. And the 13-commit walk
held the stack fixed while varying the code, which is the wrong axis.

### The four questions

**1. Should the default 5x5 closure return to DD2's own basin? NO, and it
cannot.** This is the physics ruling the ticket asked for, and it is
stack-independent.

The published DD2 couplings violate the closure's own cross-constraint
`f''_sigma(1) = f''_omega(1)` by **2.200718e-03** — measured identically on
both stacks. The 5x5 system has five exact conditions; the published table
satisfies four and misses the fifth. **The published point is therefore a
stationary point of the residual norm, not a zero of it**, and no seed
recovers it because it is not a root.

    scipy 1.13.1   iso residual 2.201e-03   couplings bit-identical to published
    scipy 1.17.0   iso residual 6.686e-11   cross row -1.23e-12
                     b_sigma -0.0250   c_sigma -0.0269   (Q_sat's coefficients)
                     -> predicted Q_sat 117.4944

The old behaviour was `hybr` returning the seed after zero iterations, and
`ISO_GATE = 2e-2` — sized to admit exactly this 2.2e-3 — waving it through as
`ok=True`. So `test_api.py:127` encodes a solver artifact as a requirement,
and `nmp.py`'s "the round trip returns the published couplings unchanged" was
never a statement about the closure. **Corrected in `5644ed0`** (docstrings and
comments only; no number moves).

`nmp.py:70`'s "spurious basin" is also mis-described for this case: 117.49 is
not a spurious basin, it is *the* root of the stated 5x5 system.

**2. Better seed, the pin, or N_RESTARTS? None of them.** Not a seed problem.
The restart loop is irrelevant twice over — it fires only on a gate miss, and
the gate is MET on both stacks.

**3. What tolerance is honest for a third finite difference?** This is the
"report, not a fix" shape, and it is the **6x6** test
(`test_inversion_with_Q_sat_still_imposes_it`: 168.6459 against 169.0034,
diff 0.357, asserted `abs=0.2`). The ticket's own h-sweep spans 166.29–169.00
— a 2.7 MeV band — while the docstring names ~0.1 MeV at h = 1e-4 and the
shipped h is the outlier of that sweep. `abs=0.2` asserts below the floor.
**Not touched**: §12 forbids loosening it, and the honest alternative (moving
h to the 3e-4–1e-3 optimum in BOTH directions together per `nmp.py:85`, then
regenerating) is a golden-reference change. It belongs to the stack ruling
below, since the tolerance must be re-derived on whichever stack wins.

**4. Does Q_sat belong in a frozen rtol=1e-10 baseline? No** — ticket 40's
`mu_S` defect in a different hat: a quantity with a documented >=0.1 MeV noise
floor pinned at ten digits. But this is now **secondary**: on a fixed stack
`dd2.npz` reproduces exactly, so the ten-digit pin is only violated when the
stack moves. A latent defect, not what broke the suite.

### One correction the 6x6 path earns

A worry raised and **disproved**: if `hybr` never left the seed, the tov tests
would be integrating a DD2-default star while believing it soft and
Delta-rich. Measured on 3.9 for the tov sample (K_sat 250, Q_sat 100,
L_sym 30): the 6x6 path does move, returning K_sat 249.187 and Q_sat 100.995
— within the gate's 1e-2-scaled 2 MeV. The stall is specific to the 5x5 path
at DD2's own NMPs, where four rows are already exactly zero. On 3.14 the same
call returns `None` with `isoscalar residual 8.12e-02`, which is the tov
failure the ticket quotes.

### What changed

`eos/dd2/nmp.py` docstrings and comments only (`5644ed0`). No equation,
tolerance, `.npz` or returned number touched. **0 added failures** on the
anaconda stack: `test/dd2` + `test/test_imports.py` = 394 passed.

### Graduated

The stack ruling is now sharp enough to ticket and is NOT mine to make — it
decides whether 13 `.npz` files regenerate. See
[ticket 57](57-canonical-stack.md).
