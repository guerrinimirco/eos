# Apply the approved renames — eos/dd2

Type: task
Status: open
Blocked by: 10, 42
Parent: ../map.md

## Question

The largest blast radius on the list, ~240 call sites, over half of them in
`test/`. Ticket 07 calls dd2 the second-worst package.

**Rule 2, the parameter object (3):**

    Parametrization                    -> Parameters       (52 eos / 144 test / 43 nb)
    Parametrization.from_dd2_defaults() -> Parameters.default()
    Parametrization.from_dd2y_defaults() -> Parameters.named("DD2Y")

`from_dd2_defaults` says "dd2" twice in `eos.dd2.Parametrization.from_dd2_defaults`.

**Rule 2, the solvers named after the §3 modes rather than after `octet` (6):**

    solve_octet          -> solve                            (35 eos / 66 test / 2 nb)
    solve_beta_eq_octet  -> solve_beta_eq_neutrinoless
    solve_fixed_yc_octet -> solve_fixed_yc
    solve_yl_octet       -> solve_beta_eq_neutrino_trapped
    sweep_octet          -> sweep
    sweep_beta_eq_octet  -> fold into sweep

**Rule 2, the warm start (2):**

    beta_warm_start  -> warm_start
    octet_warm_start -> warm_start

Two functions collapsing to one name: confirm they are the same job before
merging, and if they are not, the second keeps a name saying how it differs.

**RULED KEEP** (ticket 10 Q2): `solve_composition`, `solve_snm` and their `_t0`
twins are NOT renamed. They are not §3 modes — symmetric matter at saturation is
what `nmp.py` needs, not a mode a caller selects — so rule 2 does not bind them.

Already done: `notebook_api.py` deleted under [ticket 03](03-stage0-removals.md).

Resolved when dd2 is renamed and the added-failure count is reported. The §12
golden references bind hardest here: the DD2 golden SNM point at
n_B = 0.16 fm^-3, the published NMP/TOV values pinned in `dd2/verify` and
`test/dd2/`, and the CompOSE HS(DD2) slices. A rename moves NO number.
