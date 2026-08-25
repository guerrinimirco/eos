# ccdm.md and ccdm.tex to §11 standard

Type: task
Status: resolved
Blocked by: 09
Parent: ../map.md

## Question

`ccdm.md` scored **3/16** — the joint worst. It has no Lagrangian and no grand
potential, and `B_eff = [U(0)−U(φ₀)] + [V(0,0)−V(f_π,ζ₀)]` names `U` and `V`
without ever defining either: §11's first prohibition, at the centre of the
document. No integrals at either temperature, no unknown vector, no ordered
residual, no `s`, no scalar density; tier-1 parameters listed by name with no
numbers. `ccdm.tex` is at 14/16 and is the content source for most of it.

`docs/ccdm_implementation.md` is the reference for what the model implements.

Bring both files to CLAUDE.md §11's test — a physicist reproduces the
model from the document without opening the source — with the `.md` and the
`.tex` carrying the **same information**, each written natively for its format
([ticket 09](09-tex-or-md.md)).

§11 requires, explicitly, in both files: the Lagrangian or grand potential; every
parameter and the reference it is fitted to; the field or gap equations; the
residual **row by row in the order the solver assembles it**, with the unknown
vector; the single-species thermodynamics at `T = 0` and `T > 0` **written out
rather than cited** — the Fermi and Bose integrals are shared code but each
document states them anyway; **every** returned quantity, including `s` and `n_s`
and the identities they come through (`n_s = (eps − 3P)/m*`,
`s = (eps + P − Σ_i mu_i n_i)/T`); the terms that differ between `P` and `eps`;
and which rows each mode changes.

Per-document gaps are named in
[document-audit.md](../research/document-audit.md). The code decides wherever a
document and the source disagree; report any disagreement rather than silently
following one.

**Carried in from [ticket 27](27-document-defects.md)**: `ccdm.md` uses
`R1..R5` to label **modes** while `ccdm.tex` uses `R1..R4` to label **residual
rows**, and `ccdm.md` then uses `R_4` in both senses. Ticket 27 did not fix it
because the fix is to choose one labelling across the pair, which is this
ticket's job. §11 wants the residual rows numbered in solver order; give the
modes a different label entirely.

## Resolution

**Both files rewritten to §11; `ccdm.tex` compiles clean.** Committed as
`9008537`: `eos/ccdm/ccdm.tex`, `eos/ccdm/ccdm.md` and `docs/eos.bib`, by
explicit path. 22 pages, `grep -c undefined ccdm.log` = 0. Build artifacts
cleaned; nothing else staged.

### The label collision (carried in from ticket 27)

Discharged by giving the **modes** the new label, since §11 requires the
residual rows to be numbered in solver order and the rows were already
`R_1..R_4` in the code (`thermodynamics.internal_rows`, `state_at`'s
`field_residual`). Both files now say, in one sentence each: the specification
labels its closure rows `R1..R5`, and they are relabelled **`M1..M5`** here
because `R_1..R_4` are the residual rows. `solver.py`'s module docstring still
uses the specification's `R1..R5` — out of scope for a document ticket, and
reported rather than changed.

### Four defects the code overturned

Each checked against the source before writing, per ticket 27's lesson:

1. **`rho_s` sign.** `ccdm.tex` §pairing wrote `rho_s,j = -dOmega/dM*_j`. The
   per-mode identity three sections earlier is `R_s = -dP/dM*`, and `Omega =
   -P`, so it is `+dOmega/dM*`. `pairing.py:468` (`delta_rho_s *= -inv`, with
   the comment "rho_s = +dOmega/dM") confirms the code has it right.
2. **The gapless test.** The `.tex` said `min_k E < 0`. It is
   `min_energy < GAPLESS_FRACTION * max|Delta|` with `GAPLESS_FRACTION = 1e-3`
   (`pairing.py:312, 478`).
3. **Which potential ranks the candidates.** The `.tex` said "compared by
   `Omega`" in three places. `solver.solve` ranks by `min(..., key=p.f_total)`
   — `f = eps - Ts`, the right potential at fixed density; the mixed adapter
   (`adapters.py:1281`) ranks by `st.P`, i.e. `Omega`, at fixed potential. Both
   files now state the split.
4. **The thermal-neutrino sector.** `solver.thermal_sectors` adds `N_nu` massless
   `mu = 0` flavours (3 free-streaming, 2 trapped, since the trapped flavour is
   already counted at its own potential) to `P`, `eps` and `s`. It appeared in
   neither document — the same class of omission ticket 27 found in `sfho`.

### What was added beyond the audit's gaps

C2: tier-1 table with all nine vacuum constants and their source, the derived
block with its eleven numbers recomputed from `parameters.py`
(`zeta_0 = 94.0452`, `lambda = 16.3866`, `v^2 = 7486.83`, `lambda_z = 31.4135`,
`v_zeta^2 = -4039.30`, `C_0 = 2.43517e9`, `phi_0 = 56.25`), tier 2, the eight
tier-3 fields with shipped value / prior / pinned-or-knob status, the four
published sets, and `mu_ceiling = 607.47 MeV`. `B_eff` recomputed:
`B_chi^(1/4) = 229.89`, `B_eff = (239.66 MeV)^4 = 429.39 MeV/fm^3`.

C6: the ~45 `EoSPoint` fields grouped by what they are, the ~40 fields of the
matter block, the 38 table-row columns (including the `chi` / `chi_diel`
collision), the progress dictionary, and all six `eos_response` outputs with
their formulas and the one implemented freeze. **`n_s = (eps - 3P)/m*` is
declared inapplicable with its reason** — three distinct `M*_f`, and `eps` and
`P` carry `U`, `V`, `W` and `Sigma_R n_q` — with the per-mode relations
`N = dP/dmu*`, `R_s = -dP/dM*`, `S = dP/dT` as the stronger audit that replaces
it; and the `n_S`-is-strangeness collision is stated.

Also newly written out, in both files: the explicit Hellmann-Feynman forms of
`dn_j`, `drho_s,f` and `ds_pair`; the `phi(x) = x + 2T ln(1+e^-x/T)` kernel the
pairing potential is built from; the mode charge rows as equations; the
`P_log` vs `P_k4` surface term; `k_max`, the `60 T` absent-mode threshold and
the `Phi` guard with its reasons; the row scales and the `1e-10` gate against
the root finder's `1e-13`; the cold start with all eight seeds; the internal
system `eos.mixed` consumes; the branch and pattern seed tables with numbers;
and the species-flag table with the three that raise.

The Euler audit is quoted with a measured number: `-7.0e-16` relative at
`n_B = 1.5 fm^-3`, `T = 0`, `beta_eq_neutrinoless` (branch `partial`, pattern
`unpaired`, scaled residual `2.5e-14`).

### `docs/eos.bib`

`ccdm.tex` had a hand-written `\begin{itemize}` reference list — the only model
document besides `njl.tex` not citing the shared bib. It now uses
`\bibliography{../../docs/eos}`, which every key it needed already satisfied.
Doing so surfaced **two latent defects in the shared bib**: `Steiner2002` and
`deCarvalho2010` carry bare `_` in their `note` fields, which halts pdflatex.
Both escaped. `ParticleDataGroup2024` (Navas et al., PRD 110 (2024) 030001)
appended for the tier-1 vacuum constants — the only key absent, re-checked
against the file immediately before appending.

### Reported, not fixed

- `solver.py`'s module docstring and the four `solve_*` docstrings still call
  the modes `R1..R5`, which is the specification's labelling and now differs
  from both documents' `M1..M5`.
- `species.py` imports `DEGENERACY` under `# noqa: F401` and never uses it.
