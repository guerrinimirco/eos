# The branch-selecting construction, and the section 8 gate at the table boundary

Type: prototype
Status: open
Blocked by: 07
Parent: ../map.md

## Question

Milestone 1 gives three separately fitted patterns. This ticket combines them
into one model that respects the mode's conserved charges -- and the whole
difficulty is that **which potential is extremized depends on the mode**, and
the two answers disagree exactly where it matters.

`eos/njl/solver.py:36-53` is the authority and it is worth quoting, because it
is the single most load-bearing fact on this map:

> *"fixed mu_B (which is what `thermo_from_mu` and therefore `eos.mixed` do) is
> by pressure instead. The two pick the same phase wherever one phase is
> stable, and they CANNOT agree inside a first-order transition, because there
> no pure phase is the ground state at fixed n_B -- a mixture is. ... measured
> on `rg_njl1` in beta equilibrium at T = 0, the 2SC and CFL branches have
> equal f at n_B = 0.653 fm^-3, while equal pressure at equal mu_B puts the
> transition at mu_B = 1370 MeV with the window n_B = 0.583 -> 0.736 fm^-3
> (3.6 -> 4.6 n_0, the fixed-potential number the MUSES module and Kunkel et
> al. both report). So the switch in a fixed-n_B sweep is NOT the transition,
> and the jump in P across it is not physics."*

### The three surfaces, and what each extremizes

**1. At fixed `(mu_B, mu_C, mu_S, T)`** -- what an `eos/mixed` phase adapter
does, and what ticket 11 needs. The grand potential density is `Omega = -P`, so
the stable phase is the one of **largest total pressure**. No inversion, no
ambiguity, microseconds. The prototype already does this
(`notebooks/csc_bag_map.py:641-661`, `eos_from_fit`), and its docstring already
makes the point that matters: *"`csc_type` is an OUTPUT here, never an input."*

**2. At fixed `(n_B, Y_C, T)` and in beta equilibrium.** The mode fixes
densities, so the ranking is by **`f = eps - T s`**. Each pattern is solved for
the potentials that satisfy the mode's charge rows, then compared. The prototype
does this too (`csc_bag_map.py:563-639`, `phase_state`), one root find per
pattern:

- `beta_eq_neutrinoless`: `mu_S = 0`, solve `n_C = n_e(-mu_C)` for `mu_C`
- `fixed_YC`: `mu_S = 0`, solve `n_C/n_B = Y_C`
- `fixed_YC_YS`: 2-D solve in `(mu_C, mu_S)`

**Keep `scan_root` (`csc_bag_map.py:531-554`) and do not replace it with a
plain `brentq`.** Measured: *"a fixed bracket is not safe here: the polynomial
pressure has no meaning where it predicts n_B <= 0, and at the ends of a wide
mu_C interval it does exactly that ... which is how the fixed-Y_C sweep
silently lost 71 of its 90 points."* Scan for a sign change between two
MEANINGFUL points, then refine.

**3. CFL refuses modes its physics does not contain**, and this is already
right in the prototype (`csc_bag_map.py:503-508`, `571-577`): locking fixes
`Y_C = 0` and `Y_S = 1` identically, so a mode demanding `Y_C = 0.5` or
`Y_S = 0` **has no CFL solution at all** and the code returns `None` rather
than a number. Its own comment notes this is CLAUDE.md section 3's statement --
*"`cfl` is not a choice of equilibrium condition but a statement about which
phase the model describes"* -- arriving on its own rather than being written
in. That is why `pqm` exposes **no `cfl` mode**: the pure branch is reached by
a `patterns=("CFL",)` restriction, exactly as `eos.njl` does.

### Representing the first-order jump without breaking section 8

Section 8 says a table DELIVERED to a structure solver has `P` non-decreasing
in `n_B` and `0 <= c_s^2 <= 1`, while *"a raw model branch MAY violate this
inside a first-order transition region (mechanical instability is real physics
and branch mapping must be able to represent it); the violation is resolved by
a construction -- Maxwell, Gibbs, or the eta-mixed phase -- before the table
reaches TOV."*

So the design is:

- `eos_table` returns **per-branch tables plus the fixed-mu window**
  (`n_onset`, `n_offset` per temperature and fraction combination), the same
  shape `eos.njl` and `eos/mixed` already use. The window is **located at fixed
  mu**, from where the branch pressures cross at equal potentials -- never from
  the fixed-`n_B` `f`-crossing, which lies INSIDE the window rather than at
  either edge.
- The `f`-envelope over candidates is still reported, because it is what a
  fixed-`n_B` sweep naturally produces and what a caller comparing against
  `eos.njl`'s own table will see -- but it is labelled as the envelope, not as
  the transition.
- The monotonicity and causality gate runs on the **CONSTRUCTED** table.
  Section 8 is explicit that the gate belongs to whoever builds a table a
  structure solver consumes, so it lives in `pqm`'s `verify/` (ticket 10) and
  not in someone else's suite.

### One temperature-specific terminus to handle

Ticket 05 adds `Delta(T) = Delta(0) sqrt(1 - (T/T_c)^2)`, so above `T_c` a
paired branch does not exist -- it becomes the unpaired one. That is a branch
terminus in T rather than in `n_B`, and the selection must return `None` for
that pattern there rather than a gap of zero dressed as a paired phase. It is
the same failure the CFL locking filter catches in the training data, arriving
on the other axis.

### What ticket 07 changes about this ticket

If 07 says the disagreement is mostly **gapless**, then no construction fixes
it and this ticket's job shrinks to representing the jump correctly and
REPORTING the ambiguity, with the gate renegotiated. If 07 says mostly
**degenerate**, the model should return the margin and the runner-up so a
caller can see the branch is undetermined. If mostly **fit error**, this ticket
is where more terms or a reweighted fit earn their place. **Read 07's verdict
before starting.**

## Gate

- One entry point that, given a mode and its conditions, returns the selected
  branch, the margin over the runner-up, and the runner-up's name.
- **The transition `n_B` within a few percent of `eos.njl`'s fixed-mu window**,
  measured on at least `rg_njl1` in beta equilibrium at T = 0, where the answer
  is known independently: `mu_B = 1370 MeV`, window `0.583 -> 0.736 fm^-3`.
- **Phase agreement >= 90% in every benchmarked mode** -- the blocking gate --
  or 07's renegotiated target with the reason.
- The constructed table **monotone in P and `0 <= c_s^2 <= 1`**, asserted; the
  RAW per-branch table allowed to violate it inside the window, which is the
  point.
- A paired pattern correctly refused above its `T_c`, and CFL correctly refused
  in modes demanding `Y_C != 0` or `Y_S != 1`.
- `scan_root`-style bracketing retained, with a count of points found, so the
  71-of-90 silent loss cannot recur unnoticed.
