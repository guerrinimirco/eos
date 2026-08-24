# notebooks/hadronic_eos — the six figure families and the TOV pass

Type: task
Status: open
Blocked by: 12
Parent: ../map.md

## Question

Stage 1, figures. All styling from `eos/general/figure_style.py` and nothing else
(§10); overlays from `eos/general/constraints/` via
`overlay(ax, plane, ...)` (`eos/general/constraints/__init__.py:403`). **Every
panel selectable for with/without hyperons and with/without deltas.**

1. `P` vs `n_B` in beta equilibrium, all models overlaid
2. `P` vs `n_B` at `Y_C = 0.5`, `Y_S = 0`, `leptons=False` — symmetric nuclear
   matter — against the heavy-ion constraints (FOPI, Danielewicz)
3. mass–radius against the M–R constraints
4. mass–tidal-deformability against the M–Λ constraints
5. speed of sound squared vs `n_B` — the label says **which one it is**
   (`cs2_isothermal` vs `cs2_adiabatic`, §5), never a bare `c_s^2`
6. composition: particle fractions `Y_i` vs `n_B`, one panel per model, species
   colours from `figure_style.particle_style`

Structure work goes through `eos.astro.tov`. The P-monotonicity and
`0 ≤ c_s² ≤ 1` check runs **before** integration and reports status rather than a
meaningless mass (§8).

Figures written to `output/hadronic/`. Done when every figure file exists and the
notebook executes clean.
