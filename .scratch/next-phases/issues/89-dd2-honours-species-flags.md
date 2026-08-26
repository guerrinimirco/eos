# `dd2.solver.solve` honours `flags.photons`

Type: task
Status: open
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
