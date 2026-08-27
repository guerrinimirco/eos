# `eos.vmit.EoSPoint.Y_S` is never assigned, and the baseline froze the zero

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

Found on 2026-08-27 while building [ticket 99](99-quark-ea-at-zero-pressure.md).

`eos/vmit/solver.py` never writes `result.Y_S` in **three of its four**
solvers. `solve_beta_eq_neutrinoless`, `solve_beta_eq_neutrino_trapped` and
`solve_fixed_yc` all set `result.Y_C = q_thermo.Y_C` and leave `Y_S` at the
dataclass default of `0.0`. Only `solve_fixed_yc_ys` has it, and only because
the mode's own `Y_S` argument goes into the constructor.

So a caller reading `eos.vmit.eos_point(...).point.Y_S` gets `0.0` at every
density, on every set, in every beta-equilibrium and fixed-`Y_C` state. At
n_B = 0.5 fm^-3 the shipped set has `n_s = 0.4255 fm^-3` -- `Y_S = 0.851` --
and the point reports `Y_S = 0.0000` beside it.

`eos.alphabag` sets the field correctly (`solver.py:313`), so this is vMIT's
alone.

### It is frozen into the baseline

`test/baseline/vmit.npz` holds **`beta.T0.n0.45.Y_S = 0.0` next to
`beta.T0.n0.45.Y_s = 0.8402`** -- the contradiction, pinned at rtol = 1e-10.
Every `*.Y_S` key of the beta, trapped and `yc` families is zero; only the
`ycys` family is right. So fixing the field MOVES A FROZEN BASELINE, which is
why ticket 99 did not fix it: that ticket's Gate says "no number moves
anywhere else", and this is the other side of the same coin from
[ticket 62](62-regenerate-baselines-py314.md).

### Why nothing caught it

`eos/vmit/table.py:88` recomputes `Y_S=result.n_s / n_B` for its rows, so
every TABLE is right and every table-driven check passes. The defect lives
only on the point object, which the baselines sample directly.

### What has to be decided

- Whether the field is fixed and `vmit.npz` regenerated, or the field is
  removed from the point in favour of the basis map every reader should use.
  Ticket 99 took the second position for its own reader without ruling on
  this one: `zero_pressure_point` computes `Y_S` from
  `eos.general.basis.quark_charges` on the solved flavour densities in EVERY
  quark model, which is both uniform and immune to a cached field going
  stale. If that is the right reader everywhere, a cached `Y_S` on the point
  is a second home for a derived quantity and CLAUDE.md section 7 argues
  against it.
- Whether any OTHER model caches a conserved fraction that its solver forgets
  to fill. `Y_C`, `Y_L` and `Y_S` are all cached fields on nine point classes
  and only `vmit.Y_S` was checked.
- The blast radius on read: `eos/mixed/backends/jacobian.py:148` reads
  `st.Y_S`, but of a MIXED state rather than a vMIT point, so it is not
  affected. Nothing else in the package reads a vMIT point's `Y_S`.

### It already cost a wrong finding

Ticket 99's own measured table recorded "**`vmit`'s default set is already
two-flavour at its surface** (Y_S = 0.0000 at n_B = 0.4404): the s quark is
unpopulated below its threshold". That is false. Read through the basis map
the same surface has **Y_S = 0.8379**, and the two-flavour E/A is a genuinely
different number, 1236.02 MeV against the three-flavour 1155.75 MeV. The
finding survived into the ticket as a warning to implementers and had to be
retracted when the flag it was arguing about was built.

## Gate

- Every quark model's point reports the strangeness fraction its own solved
  densities carry, or does not carry the field at all.
- Whichever way it goes, the `vmit.npz` keys that change are named, and the
  regeneration is the deliberate act [ticket 62](62-regenerate-baselines-py314.md)
  made it.
- A check that would have caught this: a cached fraction on a point agrees
  with `eos.general.basis` on the same point, in every model.
