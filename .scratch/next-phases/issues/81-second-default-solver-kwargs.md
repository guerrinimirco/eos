# Should the bare solver `include_*` kwargs follow §4's flags to False?

Type: grilling
Status: in progress
Assignee: session 9a857509
Blocked by: -
Parent: ../map.md

## Question

Raised by [ticket 65](65-species-flag-defaults.md), which unified §4's six
`SpeciesFlags` defaults on all-False and was explicitly scoped to the
dataclass. It exposed that the same sectors carry a **second default one layer
below**, and the two now disagree.

`zl`, `vmit` and `alphabag` take bare solver keyword arguments defaulting
**True**:

    solve_beta_eq_neutrinoless(n_B, T, params=None, include_photons=True, ...)
    # alphabag also: include_gluons=True, include_thermal_neutrinos=True

and `dd2`'s `solver.solve` carries `include_photons=True` and
`include_muons=True`. `SpeciesFlags` reaches these only through `api.py` /
`table.py`, which translate `species.photons -> include_photons` correctly.

So §4's "a sector that is off is off because its flag says so" is honoured at
the dataclass and violated below it. Measured: `eos_point` with no `species`
gives ΔP = 2.85e-04 MeV/fm³ against the bare `solve_beta_eq_neutrinoless` at
T = 10 — exactly the photon pressure — which is what turned
`test/vmit/test_uniform_api.py` and `test_tables.py` red in ticket 65. Both
were made explicit on **both** sides as a holding fix; neither now inherits
any default, so this ticket is free either way.

**A coverage gap rides along.** `zl`, `vmit` and `alphabag` moved **0**
baseline keys under ticket 65 *because the generator calls their raw solvers
and never constructs a `SpeciesFlags`*. Their public default did move —
`alphabag.eos_point` at T = 30 went 156.3823 -> 156.2985 MeV/fm³. So
`test/baseline/` does not exercise the `SpeciesFlags` -> solver wiring for
those three models at all, and whatever is ruled here should say whether that
gap is closed by a baseline case or left as a `verify/` entry.

**The candidates.**

1. **Flip them to False**, so there is one default per sector everywhere. The
   consistent reading of §4, and it moves numbers: `zl`, `vmit` and `alphabag`
   baselines would move for the first time, needing the same measure-then-
   regenerate gate ticket 65 used.
2. **Delete the kwargs**, making the flags object the only way to name a
   sector. Largest diff; the raw solvers are called directly in `verify/`
   suites and the baseline generator, so every call site would have to route
   through `SpeciesFlags`.
3. **Rule them internal**, on the ground that §5's public boundary is
   `eos_point` / `eos_table` / `eos_response` and those are already correct.
   Costs no numbers; requires saying so where a physicist calling
   `solve_beta_eq_neutrinoless` directly would read it, because the baseline
   generator itself calls them that way.

Whichever is chosen, the drift check belongs beside
`test_the_six_species_flags_all_default_to_off` in `test/test_imports.py`,
which is the precedent ticket 65 set.
