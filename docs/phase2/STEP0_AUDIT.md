# Phase-2 Step-0 audit — verified against the thesis derivation

Status of each equilibrium condition in the η-mixed-phase framework, checked
against Ch. 3 (the η framework) and Appendix A (the general local/global rule)
of the thesis, with symbolic verification where the algebra is non-obvious.

Notation: `η ∈ [0,1]` is the fraction of electrons enforcing local charge
neutrality separately in each phase; `1-η` enforce only global neutrality.
`χ = V_Q/(V_H+V_Q)` is the quark volume fraction. `C` is the **non-leptonic**
electric charge. Phases are `H` (hadronic) and `Q` (quark).

---

## 1. Confirmed: mechanical equilibrium is `P_H = P_Q` in every mode

The thesis derives `P_H = P_Q` (Eq. 3.46) from `∂f_M/∂χ = 0` for the nucleons-only,
β-equilibrated, strangeness-free case. The concern was whether this survives
(a) nonzero *hadronic* strangeness once hyperons are active, and (b) the
fixed-`Y_C` and fixed-`Y_C`+`Y_S` modes.

**It does.** Carrying out the χ-stationarity variation with a hadronic
strangeness density `n_S^H ≠ 0` present, and imposing each mode's own chemical
conditions, the residual reduces identically to `P_H - P_Q` in all three cases:

| mode | chemical conditions imposed | `∂f_M/∂χ = 0` reduces to |
|---|---|---|
| A — β-equilibrium | Eqs. 3.56–3.57, `μ_S^H = μ_S^Q = 0` | `P_H = P_Q` |
| C — fixed `Y_C` | Eq. 3.27, `μ_S^H = μ_S^Q = 0` | `P_H = P_Q` |
| D — fixed `Y_C` + `Y_S` | Eq. 3.27, `μ_S^H = μ_S^Q` | `P_H = P_Q` |

The reason is structural rather than accidental: the η-weighted electron terms
enter `f_I` and the Euler relation `f_I = -P_I + Σ_j w_j n_j μ_j` with the *same*
weights, so every `μ·∂n/∂χ` term cancels against its counterpart in `-f_H + f_Q`
once the chemical conditions hold. The `η`-dependence drops out of mechanical
equilibrium entirely. Photons and the `e_G` component cancel because
`μ_{e_G}^H = μ_{e_G}^Q` forces `n_{e_G}^H = n_{e_G}^Q` (electrons share one EoS)
and `f_γ` is phase-independent.

**Implementation consequence**: the mechanical-equilibrium residual is one
function, `P_H - P_Q`, independent of mode and of `η`. Do not write per-mode
variants of it.

## 2. Correction: the Mode-D strangeness condition depends on a choice not yet made

This is the one place where the prepared hypothesis (§5c-ii, "plain-matched
`μ_S`") is **underdetermined rather than wrong**. Fixing `Y_S` admits two
physically distinct readings, and they give different equations.

### Variant D-global — `Y_S` fixed on the *total* system

`n_S = (1-χ) n_S^H + χ n_S^Q` with `n_S/n_B = Ȳ_S` prescribed. Strangeness is
then a **globally** conserved charge, and Appendix A's rule for global charges
applies: one common potential across the two phases,

```
μ_S^H = μ_S^Q        (plain-matched)
μ_B^H = μ_B^Q
```

This is what the hypothesis's §5c-ii asserts, and it is correct **for this
reading**. Strangeness is free to redistribute between phases; only the total
is held.

### Variant D-local — `Y_S` fixed *per phase*

`n_S^I = Ȳ_S n_B^I` separately in each phase. Strangeness is now **locally**
conserved, and Appendix A Eq. (A.93) applies instead: a locally conserved
charge gets *no continuity condition of its own* and is absorbed into the
baryon relation, weighted by its fixed fraction. Carrying out the variation
with respect to `n_B^H` under the local constraints gives

```
-(χ-1) · ( μ_B^H + Ȳ_S μ_S^H − μ_B^Q − Ȳ_S μ_S^Q ) = 0
```

so that, for `χ ≠ 1`,

```
μ_B^H + Ȳ_S μ_S^H = μ_B^Q + Ȳ_S μ_S^Q       (A.93)
```

and there is **no** separate `μ_S^H = μ_S^Q`. Substituting this back into the
χ-stationarity residual again yields exactly `P_H = P_Q`, so both variants are
internally consistent — they are different physics, not one right and one wrong.

### Why this matters practically

The two variants differ in the *count and identity of unknowns*, not just in a
residual expression:

- D-global: `μ_S` is a single shared unknown; one global constraint
  (`Σ Y_S = Ȳ_S`). Strangeness can migrate across the interface.
- D-local: `μ_S^H` and `μ_S^Q` are **separate** unknowns; two local constraints;
  the baryon condition is the combined (A.93) form rather than `μ_B^H = μ_B^Q`.

Getting this wrong produces a solver that converges to a thermodynamically
consistent state that is not the state intended — the failure mode is silent.

**Recommendation**: implement **D-global** as the default. It is the reading
consistent with how `Y_C` is already treated in this framework (globally
conserved, per Eq. 3.7), it keeps `μ_B^H = μ_B^Q` uniform across all modes, and
it matches the tabulated-EoS convention where `Y_S` is a table axis for the
whole system. Expose D-local behind a flag if a per-phase-strangeness study is
wanted later; the residual assembly should be driven by the charge's
local/global regime, so this is a small addition rather than a new branch.

**Do not** hard-code the assumption. The regime of each charge in
`{B, C, S, L_e}` — globally conserved / locally conserved / not conserved —
must be an input to the residual assembly, exactly as the hypothesis's
meta-rule proposes. That meta-rule is confirmed correct by Appendix A; the
three regimes map onto (A.94), (A.93), and "potential → 0 in each phase
separately" respectively.

## 3. Confirmed: the four modes are one solver

Appendix A's classification supports the hypothesis's central architectural
claim. Each conserved quantity independently falls into one of three regimes:

| regime | condition across phases | potential |
|---|---|---|
| globally conserved | `μ_j^H = μ_j^Q` (A.94) | unknown, shared |
| locally conserved | absorbed into baryon relation (A.93) | unknown, per-phase |
| not conserved | `μ_j^I = 0` in each phase separately | eliminated |

Every named mode is a choice of regime per charge, plus the two conditions that
are always present (`μ_B` matching in some form, and `P_H = P_Q`). The `zlvmit`
branch explosion is therefore avoidable: assemble the unknown vector and
residual list from the regime assignment, do not enumerate mode combinations.

The η-specific content sits entirely in the charge condition — Eq. 3.27,
`μ_C^H + η μ_{e_L}^H = μ_C^Q + η μ_{e_L}^Q` — and in the β-equilibrium form
Eqs. 3.56–3.57. Note that `μ_{e_G}` appears in the β-equilibrium condition with
weight `1-η` but drops out of the fixed-`Y_C` charge condition; this asymmetry
is correct and follows from `C` being conserved in one case and not the other.

## 4. Still to derive, not yet verified

- **Neutrino trapping inside the η framework.** The thesis neglects neutrinos
  entirely (Ch. 3 preamble). Whether the `e_L`/`e_G` split should extend to
  neutrinos, or whether trapped neutrinos are necessarily global (their mean
  free path being the physical basis of the split, and being long), is a
  physics question the thesis does not answer. The mean-free-path argument
  favours treating neutrinos as purely global — i.e. `μ_ν^H = μ_ν^Q` with no
  local component — but this needs to be stated as a modelling assumption
  rather than derived.
- **Muon split.** Deferred by instruction; the state vector must accommodate it
  as a targeted addition.
- **Surface tension.** Appendix A carries `P_H = P_Q - 2ε/R`; the bulk
  constructions here drop it, with `η` emulating its net effect instead. Keep
  the residual written so the `-2ε/R` term could be restored.

---

*Symbolic verification of §1 and §2 was performed with SymPy: the
χ-stationarity residual and the `n_B^H` variation were expanded and simplified
under each mode's substitutions, confirming the reductions quoted above.*
