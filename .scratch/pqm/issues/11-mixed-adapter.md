# `pqm_phase`: the `eos/mixed` adapter, with the analytic Jacobian block

Type: task
Status: open
Blocked by: 10
Parent: ../map.md

## Question

Make `pqm` a quark phase for `eos/mixed`. This is the application the whole
effort is for: a mixed phase needs each pure phase at **imposed, non-neutral,
non-symmetric potentials**, which is precisely what `pqm` was built never to
assume.

**The contract does not need to change.** `pqm` needs LESS of it than
`njl_phase` does, not more, because its `thermo_from_mu` is a pure evaluation
with no internal solve.

### The shape

`eos/mixed/adapters.py`, templated on `alphabag_phase`
(`adapters.py:1038-1141`) -- the closest existing case, and chosen for exactly
that reason: its docstring says `alphabag.thermodynamics.thermo_from_mu` is a
pure evaluation with no internal solve, which is `pqm`'s situation too.

```python
def pqm_phase(par, flags=None, patterns=None):
    def thermo(mu, mu_C, mu_S, T, n_B_guess=None, x0=None, return_state=False):
        mu_u, mu_d, mu_s = quark_potentials(mu, mu_C, mu_S)   # general/basis
        ...
        return (th, None) if return_state else th
    ...
    return Phase(name="pqm", thermo=thermo, potential_kind="physical",
                 cold_start=..., wing_sweep=..., frozen_thermo=...,
                 jacobian_block=...)
```

with `potential_kind="physical"` (no rearrangement self-energy, so the slot
carries `mu_B` itself), and `PhaseThermo` assembled inline as `alphabag_phase`
does at `adapters.py:1094-1107`.

**What goes on `fields`.** `alphabag` passes `{}` -- "couplings explicit in mu:
no field". `pqm` has no self-consistent field either, but it DOES have the
per-pattern gaps and the selected pattern, which a caller reading a hybrid row
genuinely wants. `njl_phase` puts its gaps and colour potentials there
(`adapters.py:1231-1234`). Follow `njl_phase`: `{"Delta_1..3": ..., "pattern":
...}`. It costs nothing and it is what makes a mixed row legible.

**The flag refusals RAISE**, in the shape of `adapters.py:1072-1088`, never a
silent no-op. Photons are phase-common and threaded in by the pairing
(`mixed/species.py:22-48`), so an adapter's hardcoded `photons=False` is
correct by construction.

### The one thing `pqm` can give that the existing quark adapters cannot

**A real `jacobian_block`.** The mixed residual is finite-differenced by
default, and `eos/mixed/backends/jacobian.py:211-221` will use an analytic
block when BOTH phases supply one. Measured: `njl_phase` carries none, and
neither does `did_phase`, *"which disables `analytic_jac` for the whole pair"*.

`pqm`'s Hessian `d^2 Omega/dmu_a dmu_b` is closed form -- the basis functions
are elementary and ticket 01 already wrote their first derivatives -- so this
is one of the few places the surrogate is not merely faster but strictly more
capable than the model it replaces.

Temper the expectation with what was already measured: ticket 04 of the
njl-speed map found the FD Jacobian is only **9-24% of residuals** because
`hybr` forms it ONCE per `root` call and Broyden-updates thereafter, so an
analytic block is a **1.1-1.3x** ceiling, not the ~6x an earlier estimate
assumed. Worth having, not worth over-engineering.

### The budget to measure against

`.scratch/njl-speed/issues/04-count-the-mixed-loop.md` counted the loop and the
counts are deterministic to the call:

| what | count | wall (njl) |
|---|---|---|
| one hybrid row | 40 residuals / 41 `thermo` / 123 NJL solves | 12.8-28.8 s |
| `locate_window`, hinted | 5,700 NJL solves | 3,385 s |
| a 200-point build | ~12,100 solves | ~4,460 s |
| **the per-call budget for a 60 s build** | -- | **15.2 ms / `thermo`, 4.95 ms / `thermo_from_mu`** |

A `pqm` `thermo` call is a handful of elementary functions -- microseconds --
so the budget is met by a factor of thousands and the interesting question is
no longer speed but **whether the mixed solve still converges** when the quark
side is a smooth polynomial instead of a gap-equation solve. Two things could
go either way: the polynomial has no branch terminus, so it returns finite
numbers in regions where NJL correctly fails (which may help the solver, or may
let it wander into nonsense); and its `n_B <= 0` region is exactly where the
71-of-90 silent point loss came from.

### The comparison that makes it meaningful

Build the SAME hybrid two ways -- DD2+njl and DD2+pqm at matched NJL parameters
-- and compare row by row. `eos/mixed/api.py:171-211` returns `(rows, windows)`;
compare both. If the quark side is good to a few percent, the hybrid's `chi`,
its window edges and its `M_max` should follow at a comparable level, and any
place they do not is a place the mixed solve amplifies the fit error.

## Gate

- `pqm_phase(par, flags=None, patterns=None)` in `eos/mixed/adapters.py`,
  exported from `eos/mixed/__init__.py`, with `jacobian_block` implemented.
- **A DD2+pqm hybrid row matching a DD2+njl row** at matched parameters: `chi`,
  `P`, `eps` and the per-phase charge split within the quark-side fit error,
  and the window edges `n_onset`/`n_offset` within a few percent.
- **A full 200-point hybrid build, timed**, against njl's ~4,460 s -- n = 3,
  median, cpu beside wall, interpreter named, never wrapped in `timeout`.
- `analytic_jac=True` demonstrated working on a pair where both sides supply a
  block, with the measured speedup stated against the 1.1-1.3x ceiling rather
  than against an assumption.
- A statement of what the adapter does **below the quark branch terminus**,
  where the polynomial returns a number and NJL correctly does not. `ccdm_phase`
  RAISES there and `docs/DEFERRED.md` records that as an open question for the
  contract; `pqm` must make a deliberate choice and say which.
- `test/mixed/` passes. Run `test/njl` and `test/mixed` **one directory at a
  time** -- both hold a `test_jacobian.py` and collide at collection.
