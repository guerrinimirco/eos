# Phase 2 — DD2+vMIT η-mixed-phase engine: specification and build plan

**Read first**: `/CLAUDE.md` (repo conventions), then
`docs/phase2/STEP0_AUDIT.md` (which equilibrium conditions are verified, and
the one place the physics admits two readings).

Physics ground truth for Phase 1 is `DD2_EoS_Physics_Report.md`. Physics ground
truth for the mixed phase is the thesis Ch. 3 (η framework) and Appendix A
(general local/global rule); the equations cited by number below are from there,
and `docs/phase2/STEP0_AUDIT.md` records which of them have been re-derived and
symbolically verified rather than merely quoted.

---

## 0. Scope and non-goals

**In scope**: a mixed-phase solver coupling the existing DD2 hadronic engine to
the existing vMIT quark engine through a first-order transition, with a
continuous local/global charge-neutrality parameter `η ∈ [0,1]`; T=0 and T>0;
the four equilibrium modes; table generation; TOV integration of the result.

**Not in scope** (do not build these, do not leave hooks that constrain them):
Bayesian inference layers, nucleation dynamics, microscopic surface/Coulomb
energies, alternative quark models beyond vMIT.

**Reuse, do not rewrite**: `eos/dd2/` (hadronic), `eos/vmit/` (quark),
`eos/general/` (JEL integrals, leptons, constants), `eos/tov/` (structure).
These are validated. Phase 2 is a new package that *consumes* them.

`eos/zlvmit/` is the first-generation nucleons+quarks mixed-phase code. Read it
to understand behaviour and to harvest its bracketing/continuation heuristics.
Do **not** copy its structure: its per-mode branch duplication is precisely
what this rewrite exists to eliminate. It stays in the tree, untouched, as a
regression oracle for the nucleons-only β-equilibrium case.

## 1. Physics specification

### 1.1 The η split

`η` is the fraction of electrons enforcing charge neutrality **locally**, in
each phase separately; `1-η` enforce neutrality only **globally**, across the
mixed phase. Physically it emulates the net effect of finite-size effects
(surface tension, Coulomb, Debye screening) without modelling them: `η → 1`
corresponds to structures large compared to the screening length (electrons
tied to individual lumps), `η → 0` to structures small compared to it.

Two electron populations therefore exist per phase: `e_L` and `e_G`. They are
the *same particle with the same EoS*; the split is bookkeeping over which
neutrality constraint each fraction answers to.

`χ = V_Q/(V_H + V_Q)` is the quark volume fraction. The mixed phase is defined
by `0 < χ < 1`; clamp to the pure phases outside that.

### 1.2 Free energy

```
f_M = (1-χ) f_H + χ f_Q                                          (3.3)

f_H = Σ_{h}  f_h(n_h^H, T) + η f_e(n_{e_L}^H, T)
                            + (1-η) f_e(n_{e_G}^H, T) + f_γ(T)   (3.4)

f_Q = Σ_{q}  f_q(n_q^Q, T) + η f_e(n_{e_L}^Q, T)
                            + (1-η) f_e(n_{e_G}^Q, T) + f_γ(T)   (3.5)
```

where `h` now runs over the **full active hadronic content** (nucleons,
hyperons, Δ, per `SpeciesFlags`) rather than just `p,n` as in the thesis, and
`q` over `u,d,s`.

### 1.3 Constraints

Global conservation of baryon number and non-leptonic charge:

```
n_B = (1-χ) n_B^H + χ n_B^Q                                      (3.6)
n_C = (1-χ) n_C^H + χ n_C^Q                                      (3.7)
```

Neutrality, split by η:

```
n_C^H - n_{e_L}^H = 0                                            (3.8)
n_C^Q - n_{e_L}^Q = 0                                            (3.9)
(1-χ)(n_C^H - n_{e_G}^H) + χ(n_C^Q - n_{e_G}^Q) = 0             (3.10)
```

`n_B^H`, `n_C^H`, `n_S^H` are sums over the active baryons weighted by their
`B`, `Q`, `S` quantum numbers — **`S = +1` per s-quark**, so `Λ` has `S=+1`,
`Ξ` has `S=+2` (see `/CLAUDE.md` §2). Quark side: `n_B^Q = (n_u+n_d+n_s)/3`,
`n_C^Q = (2n_u - n_d - n_s)/3`, `n_S^Q = n_s`.

Note that with hyperons active `n_S^H ≠ 0`, unlike the thesis case.

### 1.4 Equilibrium conditions

Verified in `STEP0_AUDIT.md`. Baryon and global-electron matching hold in every
mode:

```
μ_B^H = μ_B^Q                                                    (3.24)
μ_{e_G}^H = μ_{e_G}^Q   ⟹   n_{e_G}^H = n_{e_G}^Q                (3.33)
```

The second implication follows because electrons share one EoS across phases;
exploit it to eliminate a variable.

Charge condition, **mode-dependent**:

- `C` globally conserved (fixed `Y_C` modes) — the η-shifted matching:
  ```
  μ_C^H + η μ_{e_L}^H = μ_C^Q + η μ_{e_L}^Q                      (3.27)
  ```
- `C` not conserved (β-equilibrium) — per-phase, with `μ_{e_G}` reappearing:
  ```
  μ_C^H + η μ_{e_L}^H + (1-η) μ_{e_G}^H = 0                      (3.56)
  μ_C^Q + η μ_{e_L}^Q + (1-η) μ_{e_G}^Q = 0                      (3.57)
  ```
  (thesis writes these as `μ_p + η μ_{e_L} + (1-η) μ_{e_G} = μ_n` etc.)

The asymmetry — `μ_{e_G}` present in the second, absent from the first — is
correct and is a consequence of `C` being conserved in one case and not the
other. Do not "symmetrize" it.

Strangeness:

- self-equilibrating (default): `μ_S^H = 0` and `μ_S^Q = 0` **independently**
  in each phase (3.28–3.30) — weak equilibrium w.r.t. non-leptonic
  flavour-changing reactions.
- `Y_S` fixed globally: `μ_S^H = μ_S^Q`, one shared unknown. **This is the
  default for Mode D** — see `STEP0_AUDIT.md` §2 for why, and for the
  alternative per-phase reading which gives a different equation
  (`μ_B^H + Ȳ_S μ_S^H = μ_B^Q + Ȳ_S μ_S^Q`, with no separate `μ_S` matching).
  Implement D-global first; D-local behind a flag, and only if asked.

Mechanical equilibrium, **mode-independent and η-independent**:

```
P_H = P_Q                                                        (3.46)
```

This was verified to survive nonzero `n_S^H` and all three mode families. Write
it once. Photon and `e_G` contributions cancel between phases identically.

### 1.5 The regime-driven formulation (the central design requirement)

Each conserved quantity in `{B, C, S, L_e}` independently sits in one of three
regimes, and the regime determines its contribution to the unknown vector and
residual list:

| regime | condition | potential |
|---|---|---|
| globally conserved | `μ_j^H = μ_j^Q` (A.94) | one shared unknown |
| locally conserved | absorbed into baryon relation (A.93) | per-phase unknowns |
| not conserved | `μ_j^I = 0` in each phase separately | eliminated |

**Requirement**: assemble the unknown vector and residual list *from the regime
assignment*. Do not enumerate mode combinations. The four named modes must be
four configurations of one solver, and an unnamed combination must work without
new code. This is the single most important architectural constraint in this
document; `zlvmit`'s branch explosion is the anti-pattern.

### 1.6 The two flavors of "fixed `Y_C`" — do not conflate

Phase 1 already distinguishes these and Phase 2 must preserve the distinction
(`eos/dd2/solver.py`, `charge_mode` / `yc_leptons`):

- **leptonless**: `Y_C` fixes the hadronic charge fraction with **no leptons
  present**. This is the CompOSE general-purpose `(n_B, T, Y_q)` convention.
- **with neutralizing leptons**: leptons present and neutralizing; the muon
  sector promotes an extra unknown closed by leptonic equilibrium.

Carry an explicit flag. Silent conflation of these is the most common error in
this domain.

### 1.7 Neutrinos — stated assumption, not a derived result

The thesis neglects neutrinos entirely (Ch. 3 preamble), so the η framework's
extension to trapped neutrinos is **not derived anywhere**. The assumption to
implement:

> Trapped neutrinos are treated as **purely global**: `μ_ν^H = μ_ν^Q`, with no
> local component and no `η` weighting.

Rationale: the physical basis of the `e_L`/`e_G` split is the mean free path
relative to structure size, and the neutrino mean free path is long compared to
any plausible lump scale — so the local component is not physically motivated.
This is a modelling choice. Mark it as such in the code and in any output
metadata; do not present it as following from the thesis.

Muons: electron-family only for now. The state vector, flags surface, and
residual assembly must be written so an independent muon `e_L`/`e_G` split is a
targeted addition, not a rewrite. Do not build it yet.

## 2. The four modes

| mode | fixed inputs | `B` | `C` | `S` | `L_e` |
|---|---|---|---|---|---|
| A — β-equilibrium | `(n_B, T, η)` | global | not conserved | not conserved | not conserved |
| B — β-eq + trapped ν | `(n_B, Y_L, T, η)` | global | not conserved | not conserved | global |
| C — fixed `Y_C` | `(n_B, Y_C, T, η)` | global | global | not conserved | not conserved |
| D — fixed `Y_C` + `Y_S` | `(n_B, Y_C, Y_S, T, η)` | global | global | global | not conserved |

Every row shares `μ_B^H = μ_B^Q` and `P_H = P_Q`; every row with leptons shares
the `(μ_{e_G}` matched, `μ_{e_L}^{H,Q}` neutrality-slaved) machinery.

## 3. Architecture

### 3.1 Package layout

```
eos/mixed/
    __init__.py          # public API only
    charges.py           # ChargeSpec, regimes, quantum-number tables
    modes.py             # the four named modes as ChargeSpec factories
    state.py             # MixedState (unknown vector <-> named quantities)
    residual.py          # regime-driven residual assembly
    jacobian.py          # hand-coded analytic Jacobian, FD-verified
    phases.py            # thin adapters over eos/dd2 and eos/vmit
    solver.py            # solve_mixed(), bracketing, boundary location
    continuation.py      # warm start, sweep, boundary refinement
    table.py             # table driver
    verify/
        identities.py    # Euler, f = eps - Ts, Sigma^R placement
        limits.py        # eta -> 0/1 Gibbs/Maxwell endpoint checks
        regression.py    # vs zlvmit; mode-to-mode consistency
```

### 3.2 Key objects

**`ChargeSpec`** — the regime assignment. Something equivalent to:
```python
@dataclass(frozen=True)
class ChargeSpec:
    B: Regime = Regime.GLOBAL          # always
    C: Regime = Regime.NOT_CONSERVED
    S: Regime = Regime.NOT_CONSERVED
    L_e: Regime = Regime.NOT_CONSERVED
    targets: Mapping[str, float] = ...  # Y_C, Y_S, Y_L where applicable
    yc_leptons: bool = False            # §1.6
```
`modes.py` provides `mode_A()`, `mode_B(Y_L)`, `mode_C(Y_C)`, `mode_D(Y_C, Y_S)`
as factories returning `ChargeSpec`. Arbitrary combinations must be
constructible directly.

**`MixedState`** — maps the flat unknown vector used by the solver to named
physical quantities and back. All indexing logic lives here; nothing else may
index the vector positionally. This is what makes the muon split a targeted
addition later.

**`phases.py`** — adapters presenting a uniform interface over the two engines:
given `(potentials, T, flags)` return `(densities, P, eps, s, μ_i)` plus
derivative blocks. The hadronic adapter wraps `eos/dd2`; the quark adapter
wraps `eos/vmit`. Per the kickoff, derivative blocks are **mandatory** for
these two models but the abstract interface must declare them
optional-with-FD-fallback so a future quark model is not blocked.

**Follow the Phase-1 unknown-vector convention**: solve in **kinetic**
potentials (`ν_i = μ_i - Σ0_i`), not `μ`. `eos/dd2/physics/residual.py`
documents why — `Σ^R` and the `Γ_ω ω_0` shift cancel out of the iteration, and
kinetic potentials warm-start well across a density sweep where `μ`-based
unknowns throw the trial state into the `k_F = 0` plateau. This matters more,
not less, in the mixed phase.

### 3.3 Residual assembly

Always present: baryon-number conservation; `μ_B` matching (plain, or the
(A.93) combined form if any charge is in the LOCAL regime); `P_H = P_Q`;
per-phase local neutrality (3.8, 3.9); global neutrality (3.10);
`μ_{e_G}` matching (3.33).

Per charge, by regime: GLOBAL → conservation equation + `μ_j` matching (with
the η shift for `C`, Eq. 3.27); NOT_CONSERVED → `μ_j^I = 0` in each phase, or
the β-equilibrium form (3.56–3.57) for `C`; LOCAL → per-phase target
constraints, and `μ_B` matching takes the combined form.

Scale residuals dimensionlessly, as Phase 1 does — field equations by
`m_nucleon`, density constraints by `n_B`, pressures by a characteristic
pressure. Never mix raw magnitudes across a residual vector.

### 3.4 Solver

MINPACK `hybrj` with the analytic Jacobian, as Phase 1. Warm start from the
previous sweep point, then a documented fallback guess sequence. Bracket the
phase boundaries (`χ → 0`, `χ → 1`) explicitly and clamp to pure phases
outside; the pure-phase solves are the existing engines called directly.

Post-solve gates, mandatory, matching Phase 1's `RESIDUAL_TOL = 1e-10` and
`HVH_RTOL = 1e-8`: residual norm below tolerance; `0 < χ < 1` or clamped;
Euler relation per phase; `P_H = P_Q` to tolerance; every density
non-negative.

## 4. Build plan with validation gates

Each milestone ends with tests in `test/mixed/` in the existing style. Do not
start a milestone before the previous gate is green.

1. **P0 — scaffolding.** `charges.py`, `modes.py`, `state.py`, `phases.py`
   adapters. Gate: adapters reproduce, to round-off, the values obtained by
   calling `eos/dd2` and `eos/vmit` directly at a set of golden points.
2. **P1 — nucleons only, Mode A, T=0, η endpoints.** Gate: `η=1` reproduces a
   Maxwell construction (constant `P` across the mixed phase in the
   β-equilibrated case); `η=0` reproduces Gibbs; both agree with `eos/zlvmit`
   at matched parameters. This is the regression oracle and it must be exact
   to solver tolerance, not "close".
3. **P2 — continuous `η`.** Gate: monotone, continuous behaviour of `χ(n_B)`
   and `P(n_B)` in `η`; the curves for different `η` intersect near the density
   where the pure-phase free energies cross (thesis §3.2.1); mixed phase
   shrinks in density as `η` grows.
4. **P3 — T > 0.** Gate: `f = eps - T s` and the Euler relation per phase to
   tolerance; smooth `T → 0` limit recovering P1/P2 values.
5. **P4 — Mode C (fixed `Y_C`), both lepton flavors of §1.6.** Gate: at `η=1`
   with `Y_C` fixed, pressure is **not** constant across the mixed phase — this
   is the physics point of the framework (thesis §3.1.5) and a test that
   asserts constant `P` here is asserting the bug. Mode C at the appropriate
   `Y_C` must reproduce Mode A's composition where they coincide.
6. **P5 — Mode D (`Y_S` global) and Mode B (trapped ν, §1.7).** Gate:
   mode-to-mode regression — Mode D with `Y_S` set to the value Mode C
   self-consistently produces must return Mode C's state; likewise Mode B with
   `Y_L` set to Mode A's value must return Mode A's state. These cross-checks
   are required by the kickoff and are the strongest available test that the
   regime machinery is right.
7. **P6 — hyperons and Δ on.** Gate: `n_S^H ≠ 0` handled correctly; the
   verified `P_H = P_Q` reduction holds numerically; `SpeciesFlags` gating works
   as in Phase 1.
8. **P7 — analytic Jacobian.** Gate: agreement with a finite-difference
   Jacobian across every mode (exact at T=0, JEL-floor-limited at T>0), and
   backend parity — analytic-Jacobian and numeric-Jacobian solves reach the
   same root. Mirror `test/dd2/test_dd2_m9.py`.
9. **P8 — tables and TOV.** Gate: tables generate over the full grid without
   holes; `eos/tov/` integrates them; `M(R)` curves are smooth in `η` and
   reproduce published behaviour qualitatively.
10. **P9 — verification suite.** All of `eos/mixed/verify/` green, wired into
    the existing verification runner pattern.

## 5. What to ask about rather than decide

- Any case where the two readings of a constraint give different equations and
  `STEP0_AUDIT.md` does not already pick one.
- Whether to extend the η split to muons (deferred — do not decide it by
  implementing it).
- Surface tension: Appendix A carries `P_H = P_Q - 2ε/R`; the bulk
  construction drops it. Keep the residual written so the term could be
  restored, but do not add it.
- Any convention not stated in `/CLAUDE.md`. Document it there rather than
  burying it in a docstring.
