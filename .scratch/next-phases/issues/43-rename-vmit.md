# Apply the approved renames — eos/vmit

Type: task
Status: open
Blocked by: 10, 42
Parent: ../map.md

## Question

The worst package on the list. Ticket 07: **`eos/vmit` was never converted at
all below `Parameters`** — every §13 vocabulary name in it is wrong — while
`docs/DEFERRED.md:320` calls the conversion DONE. Fix that ledger line in the
same change.

**Rule 3, drop `compute_` where it carries nothing (14):**

    compute_quark_thermo              -> kinetic_thermo
    compute_quark_density             -> quark_density
    compute_vector_field              -> vector_field
    compute_vector_pressure           -> vector_pressure
    compute_vector_energy             -> vector_energy
    compute_bag_pressure              -> bag_pressure
    compute_bag_energy                -> bag_energy
    compute_mu_effective              -> effective_potential
    compute_effective_mu_quarks       -> effective_potentials
    compute_mu_physical               -> physical_potentials
    compute_quark_densities_for_solver -> effective_state
    compute_vmit_thermo_from_mu_n     -> thermo_from_mu_n
    compute_quark_matter_thermo_from_n -> thermo_from_n
    compute_quark_matter_thermo_from_mu -> thermo_from_mu

**Rule 1, the name repeats the package (2):**

    VMITEOSResult -> EoSPoint          (the record eight other models already use)
    VMITThermo    -> MatterThermo

**Rule 2, the shared vocabulary (7):**

    get_vmit_default()            -> Parameters.default()
    solve_vmit_beta_eq            -> solve_beta_eq_neutrinoless
    solve_vmit_fixed_yc           -> solve_fixed_yc
    solve_vmit_fixed_yc_ys        -> solve_fixed_yc_ys
    solve_vmit_trapped_neutrinos  -> solve_beta_eq_neutrino_trapped
    result_to_guess               -> warm_start
    get_default_guess_{beta_eq,fixed_yc,fixed_yc_ys,trapped_neutrinos}
                                  -> default_guess(mode, ...)

**FROZEN, do not rename** (ticket 10 Q4): `VMITTableSettings`,
`compute_vmit_table`, `save_vmit_results` in `compute_tables.py`. Their only
consumer is `notebooks/ZLvMIT_hybrid.ipynb`, which the map rules out of scope and
[ticket 41](41-corrupt-notebooks.md) records as unopenable.

NOT in this ticket: deleting `get_vmit_custom()` — [ticket 46](46-api-changes.md).

Resolved when vmit is renamed, `DEFERRED.md:320` no longer claims it was already
done, and the added-failure count is reported. `test/baseline/` must not move.
