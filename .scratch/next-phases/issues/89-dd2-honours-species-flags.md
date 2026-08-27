# `dd2.solver.solve` honours `flags.photons`

Type: task
Status: resolved
Blocked by: -
Parent: ../map.md

## Question

Ruled by [ticket 81](81-second-default-solver-kwargs.md), section 1. Split out
because this is **the only commit in that ruling that moves frozen values**;
everything else it decided moves none, and mixing them makes a bisect useless
the next time a baseline moves.

`dd2.solver.solve(par, n_B, flags, ..., include_photons=True)` accepts a
`SpeciesFlags` and never reads `.photons`. Only `api.py:112` and `table.py`
translate it, so the solver's own callers get a photon gas whatever the flag
says. Measured at n_B = 0.32, T = 30:

    solve(par, 0.32, SpeciesFlags(photons=False), T=30)      P = 36.84136685
    solve(par, 0.32, SpeciesFlags(photons=True),  T=30)      P = 36.84136685
    solve(par, 0.32, SpeciesFlags(photons=False), T=30,
          include_photons=False)                             P = 36.81824551
    one photon gas at T = 30                                     0.02312133

## Work

1. `solve` reads `flags.photons` at `eos/dd2/solver.py:757`. Delete
   `include_photons` from `solve` and from its four wrappers —
   `solve_beta_eq_neutrinoless`, `solve_hadronic`, `solve_fixed_yc`,
   `solve_beta_eq_neutrino_trapped` — and from `sweep`.
2. `api.py:112` and `table.py:63,217,227,307` stop translating and pass the
   flags through.
3. The three callers that relied on the kwarg are seeds whose P/eps/s are
   discarded — `solver.py:409` (`default_guess`, via `solve_beta_eq`) and
   `eos/mixed/adapters.py:224,583`. They extract fields and potentials, which
   photons do not touch, so `mixed.npz` must not move. Set `photons=False` on
   the `replace(...)` flags there rather than deleting the intent.
   `solve_beta_eq` keeps its own `include_photons`: it takes no flags object.
4. Regenerate `test/baseline/dd2.npz` under ticket 65's measure-then-regenerate
   gate.

## Gate

- **456 of 976 dd2 rows move**: 81 at T=10, 321 at T=30, 27 each at T=40 and
  T=60. Measure the moved set BEFORE regenerating and confirm it is exactly
  that set — a row moving at T=0 means something other than photons changed.
- The 429 T=0 rows, the **DD2 golden SNM point** and the **published NMP/TOV
  values** must NOT move: photons vanish at T = 0. §12 makes these ground truth.
- Every moved row moves in P, eps and s only. Photons carry no conserved
  charge, so the composition — every n_i, every mu_i, Y_C, Y_S — is unchanged.
  A moved density is a bug, not a regeneration.
- `mixed.npz` unmoved at rtol = 1e-10.
- Full suite green before the commit (§12).


## Resolution

Executed, commit `992fd9c`, on python.org 3.14.2 (numpy 2.3.5, scipy 1.17.0). `solver.solve` reads
`flags.photons`; `include_photons` is gone from `solve`, from
`solve_beta_eq_neutrinoless`, `solve_hadronic`, `solve_fixed_yc`,
`solve_beta_eq_neutrino_trapped` and from `sweep`. `solve_beta_eq` keeps its
own, as ruled — it takes no flags object. `api.py` and `table.py` stop
translating.

### The measurement, taken BEFORE regenerating

`dd2.npz` has **4692 keys**. Re-evaluating `case_dd2()` against the frozen file:

    moved at rtol = 1e-10          162   (54 points x P, eps, s)
    unmoved                       4530   every one BIT-identical
    worst |delta - photon gas|    0.89 ulp of the total
    T = 0 and NMP keys moved         0 of 2454

Moved points: 15 at T = 10 (9 `beta.nuc`, 6 `trapped`), 33 at T = 30
(9 `beta.nuc`, 24 `yc`), 3 at T = 40 and 3 at T = 60 (`mesons`). Only P, eps
and s: no density, no potential, no Y_C, no Y_S. Every moved key moved by
**exactly one photon gas**, and the residue is under one ulp of the total —
which is what `st["P"] += ph.P * hc3` rounds to, so the move is the addition
itself and nothing else.

`mixed.npz` was ASSERTED, not assumed: **2497 keys, 0 moved, 2497
bit-identical.**

### This ticket's own gate numbers were wrong

The gate above, inherited from [ticket 81](81-second-default-solver-kwargs.md)
§1, predicted "**456 of 976** dd2 rows move: 81 at T=10, 321 at T=30, 27 each
at T=40 and T=60". Neither number reproduces: the file holds 4692 keys, not
976, and 162 moved, not 456. The per-temperature split is not a constant
multiple of the measured one either, so it is not a difference in what a "row"
counts. Every SUBSTANTIVE clause of the gate does hold and is verified above —
T = 0 unmoved, golden SNM point and NMP/TOV unmoved, composition unmoved, every
move exactly one photon gas. Ticket 81 §1's arithmetic is simply wrong and
should not be quoted again.

### Three call sites the ticket's work item 3 missed

Work item 3 named three callers "whose P/eps/s are discarded" and said the
seeds are the whole of it. Three more would have silently gained a photon gas:

- **`eos/mixed/adapters.py:537` (`_dd2_frozen_block`)** is NOT a seed. Its P
  and eps ARE the frozen sound speed, and its own docstring says "matter only:
  no leptons, no photons". Given `replace(flags, photons=False)`.
- **`eos/dd2/verify/compose.py:73`** adds its own photon gas below the solve,
  to match the CompOSE general-purpose table content. Passing the flags
  through would have double-counted it against a golden comparison.
- **`eos/dd2/backends/responses_jac.py:38,110`** solve for a state whose
  potentials feed the Jacobian; photonless by intent.

One site is left following the caller's flag deliberately: `dd2_phase`'s
`wing_sweep`. Its rows are stitched into `build_hybrid_table` as they stand,
with no mixture layer above them to contribute the radiation, so the wing must
carry whatever the mixed window carries. Before this change it took the bare
`include_photons=True` default and had a photon gas even at
`SpeciesFlags(photons=False)`; now it agrees with the window and with the
model's own call at the same conditions, which is what
`test/mixed/test_hybrid_modes.py` asserts. `eos/mixed/species.py` and
`mixed.md` say so where they described the old hardcoded `False`.

### Gate

    python.org 3.14.2   test/dd2 + test/mixed + test/baseline
    492 collected, 492 passed, 0 failed, 0 skipped   (624.99 s)

**0 added failures.** The hyperonic branch of `verify/compose.py` and the
`responses_jac` susceptibility path are not covered by those directories and
were smoke-tested by hand; both return finite numbers.
