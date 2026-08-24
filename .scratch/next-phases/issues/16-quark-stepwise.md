# notebooks/quark_eos — the NJL and CCDM step-by-step section

Type: task
Status: open
Blocked by: 15
Parent: ../map.md

## Question

Stage 2, second half, for `njl` and `ccdm` specifically. One cell per step, each
**printing the quantity it just computed**:

1. the model **without colour superconductivity**: parameters, the gap/field
   equations, the grand potential, and the thermodynamic quantities at one
   `(n_B, T)` point — each labelled with its symbol and units
2. the same point with pairing on, one pairing pattern at a time
3. **unpaired vs 2SC vs CFL** compared: grand potential per phase at fixed
   `(mu_B, T)`, which phase is favoured where, `P` and `eps` vs `n_B` per phase
4. `Delta(n_B, T)` as a 2-D map per pairing pattern, plus `Delta` vs `n_B` at
   fixed `T` and vs `T` at fixed `n_B`
5. the quantities that go with it: quark and electron fractions, `c_s²`, the
   phase boundary in the `(mu_B, T)` plane

`docs/njl_csc_implementation.md` and `docs/ccdm_implementation.md` are the
references for what these models implement. **Where the notebook and those
documents disagree, the code decides and the disagreement is reported** in the
answer.
