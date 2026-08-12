# mixed — the hadron-quark mixed phase, Gibbs to Maxwell in one parameter

The full description, with equations and bibliography, is `mixed.tex`
(compiled against `../../docs/eos.bib`). This file is the plain-text summary.

**What it is.** A *composite engine*, not a model: it supplies no matter of
its own. It couples DD2 (hadronic, Typel et al., PRC 81, 015803 (2010); Hempel
and Schaffner-Bielich, NPA 837, 210 (2010)) to vMIT (quark, Chodos et al., PRD
9, 3471 (1974); Gomes et al., ApJ 877, 139 (2019)) across a first-order
deconfinement transition. A continuous parameter `eta` in [0, 1] selects the
construction — `eta=0` Gibbs (Glendenning, PRD 46, 1274 (1992)), `eta=1`
Maxwell, intermediate values interpolating in the sense of Constantinou et
al., PRD 107, 074013 (2023).

At a given `(n_B, T, eta)` a hadronic phase of volume fraction `1-chi`
coexists with a quark phase of fraction `chi`. `chi` is solved for, not given,
and is deliberately NOT clamped: `chi <= 0` means the density is still pure
hadronic, `chi >= 1` that it is already pure quark, `0 < chi < 1` that it is
inside the window. The phase boundaries are located by exactly that sign
change.

**The phase-adapter contract** (`adapters.py`). Everything this engine knows
about DD2 and vMIT passes through one interface:

    adapter(the phase's conserved-charge potentials, T) -> PhaseThermo

which solves that phase's *own* internal self-consistency — meson fields for a
density-dependent RMF, the vector fixed point for a bag — at the given
potentials, and reports `{n_i}`, `n_B`, `n_C`, `n_S`, `P`, `eps`, `s`,
`mu_B`, `mu_C`, `mu_S`. There is no `eta`, no mixing and no neutrality in it:
those are conditions on the *pair* of phases. Pairing a different hadronic or
quark model is writing an adapter, not editing the solver.

Two properties are load-bearing. Both adapters report the same projection
`mu_i = B_i mu_B + C_i mu_C + S_i mu_S`, with `C` the non-leptonic charge and
`S = +1` per s quark, so the coexistence conditions can be written without
either side knowing which engine produced the other. And an adapter must be a
deterministic function of its arguments — the residual is differentiated
numerically, so an adapter that remembered the previous trial point would make
the Jacobian differentiate that memory along with the physics.

The hadronic adapter takes the KINETIC potential `mu_tilde_B = mu_B - Sigma_R`.
`Sigma_R` depends on the phase density, which is itself an unknown of the
phase-internal solve; carrying `mu_tilde_B` keeps that circularity inside the
adapter instead of in the outer iteration.

**The eta split.** The lepton gas is written as a *local* population (one
potential per phase, weight `eta`) plus a *global* one (a single potential,
weight `1-eta`). Neutrality is imposed where each population lives:

    local  (eta > 0):  n_C^H = n_l^H   and   n_C^Q = n_l^Q
    global (eta < 1):  (1-chi) n_C^H + chi n_C^Q = n_l^G

At `eta=1` only the local rows survive — each phase separately neutral, no
charge exchanged, one coexistence pressure, a plateau with a density jump. At
`eta=0` only the global row survives — each phase may be charged, charge is
exchanged freely, the pressure rises through the window. Intermediate `eta`
stands in for the surface tension and Coulomb cost of the mixed-phase
structures; it is a controlled interpolation, not a derivation from a surface
tension.

Only the local leptons sit inside the structures whose pressures must balance,
so mechanical equilibrium reads `P^H + eta P_l^H = P^Q + eta P_l^Q`. The
global leptons, photons and trapped neutrinos are common to both phases and
cancel from it identically.

**Charges as a declaration** (`equilibrium/charges.py`). Each of
`{B, C, S, L_e}` is GLOBAL (shared potential, conserved on the volume
average), LOCAL (per-phase potential, conserved inside each phase), or
NOT_CONSERVED (potential eliminated). `B` is GLOBAL in every mode —
`mu_B^H = mu_B^Q` is what makes the phases coexist at all.

| mode | B | C | S | L_e | independent variables |
|------|---|---|---|-----|-----------------------|
| `beta_eq_neutrinoless`     | global | — | — | — | `(n_B, T)` |
| `beta_eq_neutrino_trapped` | global | — | — | global | `(n_B, Y_Le, T)` |
| `fixed_YC`                 | global | global | — | — | `(n_B, Y_C, T)` |
| `fixed_YC_YS`              | global | global | global | — | `(n_B, Y_C, Y_S, T)` |

("—" is NOT_CONSERVED.) Nothing else in the engine enumerates modes: the
unknown vector, the residual list and the analytic Jacobian are all assembled
by reading the regimes off a `ChargeSpec`, so the four named modes are four
configurations of one solver and an unnamed combination needs no new code.
`S` LOCAL is not wired and raises; `L_e` LOCAL is not a defined mode at all —
the neutrino mean free path is far larger than the mixed-phase structures.

A regime is not a third kind of declaration. A `ChargeSpec` is a `ModeSpec`
— the same object every single-phase model takes, saying which charges are
held and at what fractions — plus one `Locality` per charge, which is the only
axis a second phase adds. The regime is the two composed: a charge the mode
does not hold is NOT_CONSERVED; one it holds is GLOBAL or LOCAL according to
its locality. So the four modes are declared once, in `eos/general/modes.py`,
and this engine says only *where* each conserved charge is conserved.

**The system.** Because the adapters absorb each phase's internal
self-consistency, the unknowns are only potentials, `chi`, and the split
lepton potentials — four to nine numbers:

    always            mu_tilde_B_H, mu_B_Q, chi
    C global          mu_C_H, mu_C_Q  (with leptons)  or  mu_C  (leptonless)
    S global          mu_S
    L_e global        mu_nue
    eta > 0           mu_eL_H, mu_eL_Q
    eta < 1           mu_eG

Residual rows: baryon potentials match; the volume average reproduces `n_B`;
mechanical equilibrium; then by regime the average charge, strangeness and
lepton-number conditions, the eta-shifted charge matching
`mu_C^H + eta mu_eL^H = mu_C^Q + eta mu_eL^Q` (which at `eta=0` is the Gibbs
statement `mu_C^H = mu_C^Q`), and the neutrality rows above.

In beta equilibrium `mu_C` is not an unknown: it is eliminated by
`mu_C^I = mu_nue - [eta mu_eL^I + (1-eta) mu_eG]`, which for transparent matter
is the repository's `mu_C + mu_e = 0`. Strangeness self-equilibrates,
`mu_S = 0`.

**Totals.** Matter phases volume-averaged; local leptons weighted `eta` and
themselves volume-averaged; global leptons weighted `1-eta`; photons and
trapped neutrinos uniform and counted once. The pressure is read off one phase
rather than averaged, because the mechanical row has made the two equal. The
Euler relation `eps + P = T s + sum_i mu_i n_i` is NOT an algebraic identity
here — it holds only if every thermal and lepton term is weighted consistently
— which makes it the sharpest single check on the assembly; it is asserted at
1e-8 relative on every solved point.

**Two sound speeds.** `c_eq^2 = dP/deps` along the solved sequence, with `chi`
free: a compression is answered by converting hadrons into quarks, so through
a Maxwell window it vanishes and through a Gibbs window it dips but stays
finite. This is the one TOV takes. `c_ad^2` holds `chi` fixed — the mixture is
compressed faster than the phases can convert, so the pressure has to rise.
Freezing `chi` is the part that matters; freezing only the charge fractions
would let the solve readjust `chi` and return to the plateau. Each phase is
then compressed by the same factor, keeps its own `Y_C` and `Y_S`, and the
leptons are re-neutralised against the frozen charge; with `chi=0` and
`leptons=False` it reduces exactly to `eos.dd2`'s adiabatic sound speed. The
gap between the two drives a composition g-mode across the transition.

**What a table returns.** `(rows, windows)`. The windows — onset and offset
per isotherm and fraction combination — are part of the result, not something
recovered afterwards by scanning for where `chi` crossed 0 and 1. Rows carry
each conserved charge resolved by phase, on the volume-weighted convention
`Y_j^I = w_I n_j^I / n_B`, giving three cheap invariants:
`Y_B^H + Y_B^Q = 1`, `Y_C^H + Y_C^Q = Y_C`, `Y_S^H + Y_S^Q = Y_S`. The
partition is what the global sums cannot show: at fixed total `Y_C` the
hadronic phase can be far more positively charged than the average while the
quark phase carries the compensating negative charge, and how far that goes is
what `eta` controls.

**Numerics.** Every residual row is made dimensionless before it is judged —
density rows by `n_B`, potential equalities by a potential scale, the
mechanical row by the larger phase pressure — and gated on the largest scaled
component at 1e-10.

Warm starts dominate the cost, since the adapters run once per residual
evaluation. Three levels: the hadronic starting field configuration is a
constant of the solve and is computed once; each density seeds the next; each
isotherm seeds the next. The mixed system is solved only *inside* the located
window (outside it the pure-phase solvers answer the same question far more
cheaply), and the window search on one isotherm is told where the previous one
found its boundaries — with the hint discarded and the full search repeated if
a boundary lands on the edge of the hinted range, so a window that genuinely
disappears is still reported as gone.

The solver's numeric Jacobian is the reference and the correctness oracle; a
hand-assembled analytic Jacobian is the fast path, validated against finite
differences and required to reach the same root. A trial point that drives a
phase solve out of its domain gets a large penalty residual, so the outer
iteration backs off instead of aborting.

Non-convergence is a return value at the public boundary, never an exception:
the engine runs inside parameter scans that walk into regions where no hybrid
equation of state exists. A converged point with `chi` outside [0, 1] is not a
failure — it is how the engine says which side of the transition a density is
on.

**Not implemented** (see `docs/DEFERRED.md`): the frozen-per-species and
frozen-conserved-fraction response freezes; `S` LOCAL; combining a fixed `Y_C`
with a fixed `Y_Le`.
