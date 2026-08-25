# Four missing verify/ invariants, and who owes the delivery gate

Type: task
Status: resolved
Blocked by: 11
Parent: ../map.md

## Question

[Ticket 11](11-conformance-triage.md) ruled these closed now rather than in
Phase 5: they are pure additions, no number moves, and two of them are real holes
in the models most likely to hide a defect.

1. **dd2's verify suite checks neither the free energy nor the rearrangement
   placement — and dd2 carries Σ^R** (finding 25).
   `eos/dd2/verify/run_full_check.py:134-156` registers golden points, thermo
   identities, responses, coeff analytic~FD, backend parity and CompOSE. Euler
   appears only inside `_check_identities` (`:70-78`, via `p.euler_residual()`).
   There is no `f = eps - T s` check and **no rearrangement check**, though
   `dd2/thermodynamics.py:492` returns `Sigma_R` and `dd2/solver.py:174,398,666`
   consume it. §8 makes "Σ^R enters mu and P, never eps" an invariant, and it is
   exactly the invariant that catches a wrong density-dependent RMF.
   `eos/ccdm/verify/run_full_check.py:263-297` is the model implementation — it
   asserts both the identity and that the term is non-trivial. Copy it.

2. **`eos/mixed`'s verify suite has the same gap** (finding 26).
   `mixed/verify/run_full_check.py` has Euler/HVH (`:106`), causality and
   monotonicity (`:187-194`) and backend parity (`:173`), no free-energy identity.
   It couples DD2, which carries Σ^R. Downstream of item 1.

3. **`eos/ccdm`'s verify suite has no causality or monotonicity check at all**
   (finding 27). `grep -n "cs2\|sound\|monoton"
   eos/ccdm/verify/run_full_check.py` returns nothing. Every other model checks
   `0 <= cs^2 <= 1` somewhere. CCDM is a colour-superconducting model where a
   wrong gap contribution shows up in the sound speed first — and it sits next to
   the best rearrangement check in the repository.

4. **`eos/njl`'s causality check is disabled by default** (finding 28).
   `njl/verify/run_full_check.py:545 check_sound_speed` exists but `:602 if
   include_sound` gates it on the `--sound` CLI flag, so `run_all()` with no
   arguments runs no causality check. An invariant that does not run by default
   is not one. If cost is the reason, turn it on anyway and mark it `slow` using
   `pyproject.toml`'s existing marker convention, and say so in the ledger.

5. **dd2 and did adopt the delivery gate** (finding 29, the (a) half).
   §8 scopes P-monotonicity to tables "DELIVERED to a structure solver", and it
   runs in only 3 of 12 suites (`sfho:234`, `enjl:889`, `mixed:191`). Eight
   models never build a table, so their absence is correct by the letter — but
   `eos/dd2/table.py:316 build_core_table` and `eos/did/table.py:203` do build
   one and do not check it. `eos/enjl/verify/run_full_check.py:866
   check_delivered_table` is the model implementation.
   The (b) half — §8 naming the delivery gate as belonging to whoever builds a
   table — is [ticket 22](22-phase5-claudemd.md)'s, and this ticket assumes that
   ruling rather than waiting on it.

Resolved when all five run in their suites by default and pass, and the
added-failure count against `output/_audit/pytest_before_with_crust.txt` is
reported. **No golden reference is at risk** — nothing here changes a computed
quantity. If one of the new checks *fails* on shipped physics, that is a finding:
report it, do not weaken the check and do not loosen a tolerance (§12).

## Answer

**All five shipped. Every verify suite passes: 12 entry points, 134 checks, 0
failures** (was 121 across eleven before this ticket and
[ticket 64](64-general-verify-suite-missing.md)).

| # | item | where | result |
|---|---|---|---|
| 1 | dd2 free energy | `dd2/verify:_check_free_energy` | ok 1.49e-15 |
| 1 | dd2 rearrangement | `dd2/verify:_check_rearrangement` | ok 1.87e-16, Sigma^R carries **4.1%** of eps |
| 2 | mixed free energy | `mixed/verify:_check_free_energy` | ok 9.67e-15 over 14 rows |
| 3 | ccdm causality + monotonicity | `ccdm/verify:check_causality` | ok, c_s^2 in [0.241, 0.358] |
| 4 | njl causality on by default | `njl/verify:run_all` | ok, `--sound` is now `--no-sound` |
| 5 | dd2 delivery gate | `dd2/verify:_check_delivered_table` | ok, 150 rows, c_s^2 in [0.008, 0.815] |
| 5 | did delivery gate | `did/verify:check_delivered_table` | ok, 200 rows, c_s^2 in [0.006, 0.711] |

**Nothing outside `eos/*/verify/` was touched.** No `api.py`, no `solver.py`, no
`thermodynamics.py`, no notebook. No number moved.

### The one design decision worth reading — item 5's gate is stated against n_B

`np.diff(P) >= 0` on a delivered table **is a check that cannot fail**, and both
`build_core_table` implementations are why: each ends

    order = np.argsort(P)          # TOV interpolation needs P increasing
    return EOSTable_for_TOV(P=P[order], epsilon=eps[order], nB=nB[order])

so the rows arrive sorted by P and the pressure column is monotone by
construction whatever the branch did. The sort does not repair the DENSITY
column — a branch whose P falls with n_B comes back with n_B permuted — so the
gate is `np.diff(nB) >= 0`, which is also what section 8 actually says ("P
non-decreasing **in n_B**"). Proved by falsification below: injecting a
softening at one row and re-sorting exactly as the builder does passes a
`diff(P)` test and fails this one. `enjl`'s `check_delivered_table` is the model
implementation and does not have the problem, because a CONSTRUCTED table is
already ordered by density; the two builders that sort do.

### The rearrangement check (item 1), copied from ccdm rather than reinvented

`ccdm/verify:check_rearrangement_placement`'s shape, with DD2's assembly in
place of CCDM's — two identities taken apart, plus the size gate:

    P   - (S_P + P_field + P_gas)       = Sigma^R n_B      the term IS in P
    eps - (S_eps + eps_field + eps_gas) = 0                and is NOT in eps

The kinetic sums are rebuilt from the state's own fields and potentials —
`baryon_kinetics` at mu~_B = mu_B - Sigma^R, the kinetic potential — so the
check does not read back the assembly it tests. Both lines hold to 1.9e-16 and
the term is 0.3% of eps at n_B = 0.1 rising to 4.1% at 0.5, so neither passes by
being small.

### Item 4: cost was NOT the reason, so there is no `slow` marker and no ledger entry

The ticket allows marking it `slow` if cost is why it was off. Measured first:
`check_sound_speed` is **0.6 s** and `check_saturation_density` **5.4 s**,
against a 3.2 s suite. That is not a `slow` marker's problem, so
`include_sound` simply defaults to `True`, the CLI flag inverts to `--no-sound`
(kept so a caller re-running the cheap invariants in a loop can still opt out),
and `docs/DEFERRED.md` gains nothing — there is nothing deferred.

### Item 2's honest scope

At T = 0, which is the window `mixed/verify` sweeps, the free-energy identity IS
the Euler relation rearranged. What it adds is the side it is read from: the sum
is rebuilt species by species by `_mu_dot_n` (extracted from the existing
`_euler_resid`, which now calls it), at each phase's own potentials and each
lepton domain's own mu_e. The docstring says this rather than implying
independence it does not have. Same for dd2's item-1 free energy, which
deliberately reads the public `free_energy_density` property — the one a
notebook or `nucleation` actually calls — instead of recomputing eps - T s.

### Every new check was proved able to fail

| break | result |
|---|---|
| dd2 free energy: property returns the MATTER's f, not the totals | FAIL 1.66e-02 |
| dd2 rearrangement: Sigma^R moved INTO eps | FAIL 4.25e-02 |
| dd2 rearrangement: Sigma^R dropped from P | FAIL 4.08e-02 |
| dd2 rearrangement: term identically zero | FAIL — **the size gate**, identities held |
| dd2 delivered: P falls with n_B, re-sorted as the builder does | FAIL 9.12e-02 |
| dd2 delivered: c_s^2 = 2 at one row | FAIL 1.00e+00 |
| mixed free energy: mu.n off by **1 ppm** | FAIL 1.33e-06 |
| mixed free energy: f built with +P | FAIL 3.35e-01 |
| ccdm causality: c_s^2 = 1.4 | FAIL 4.00e-01 |
| ccdm causality: c_s^2 = nan | FAIL inf, **names the density** (not absorbed) |
| ccdm monotonicity: P falls above 1.6 fm^-3 | FAIL 2.71e+02 |
| did delivered: P falls with n_B, re-sorted | FAIL 1.65e-01 |
| did delivered: c_s^2 = 2 at one row | FAIL 1.00e+00 |
| (controls) all unbroken | PASS |

The nan row is [ticket 63](63-verify-causality-nan-silent-pass.md)'s guard,
written into the new check from the start rather than retrofitted.

### Ticket 69 was not touched

The causality check reads **`cs2_isothermal`**, which is the key `ccdm`'s and
`njl`'s `eos_response` already return — neither is among ticket 69's four
models. Nothing was renamed.

### Gates

- **Verify sweep, all twelve entry points**: `abpr` 9, `alphabag` 10, `ccdm`
  17, `dd2` 9, `did` 14, `enjl` 17, `mixed` 10, `njl` 17, `sfho` 8, `vmit` 8,
  `zl` 10, `general` 5 — **134 ok, 0 FAIL**. (A thirteenth exists,
  `eos/astro/gmode/verify/run_full_check.py`; it is not a model suite and was
  not in this ticket's scope.)
- `python3 -m pytest test/dd2 test/did test/ccdm test/njl test/mixed
  test/general test/test_imports.py` — **1032 collected, 1029 passed, 3 failed**
  in 954 s. The three are `test/dd2/test_api.py::test_inversion_without_Q_sat_predicts_it`,
  `::test_inversion_with_Q_sat_still_imposes_it` and
  `test/dd2/test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion` —
  **the same three node ids, verbatim, as `output/_audit/pytest_after_ticket61_dd2_py314.txt`
  (3 failed, 203 passed) and as `pytest_after_ticket56_py314.txt`.** Zero added.
  They are ticket 47's stack artifact, owned by
  [ticket 47](47-dd2-nmp-inversion.md).
- `python3 -m pytest test/baseline` (rtol = 1e-10) — **6 failed, 10 passed**:
  `ccdm, dd2, enjl, njl, tov, zlvmit`, identical to
  `pytest_after_ticket61_baseline_py314.txt`. Zero added.
- Interpreter **python.org 3.14.2** (numpy 2.3.5, scipy 1.17.0), never prefixed
  with `timeout`. A full-suite number is unmeasurable while three sessions hold
  `notebooks/*.py` (`notebooks/enjl_eos.py` and `notebooks/hybrid_eos.py` were
  modified in the tree throughout), so the targeted suites above are what this
  ticket can stand behind; they cover every file it touched and every test that
  imports one.

### Finding — reported, not fixed

`eos/dd2/verify/run_full_check.py`'s module docstring lists as item 3 a "TOV
cross-check — M_max >= 2 M_sun", which the suite does not run and which
`run_full_check`'s own docstring correctly says lives in `test/dd2/`. The list
was already wrong; this ticket renumbered around it rather than editing a line
it was not asked to touch. One-line fix for whoever owns the docstring sweep.
