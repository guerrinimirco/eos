# Ship `eos/pqm` in the section 5 layout

Type: task
Status: open
Blocked by: 06, 08, 09
Parent: ../map.md

## Question

Turn the proven notebook into a model subpackage. Milestone 1 deliberately kept
this last: the basis changed three times on the way here (closed-form
derivatives, the conformal fix, finite T), and a section 5 package whose
`parameters.py` has to be reworked twice is worse than a notebook that moves.

### The layout

CLAUDE.md section 5: **the names are mandatory, the existence is conditional.**

```
eos/pqm/
  __init__.py          re-exports + the layout map in the docstring
  parameters.py        Parameters(eta_D, G_V0_over_GS, M_g, vector_form,
                       tc_coeff, + the fitted coefficient surfaces)
                       .default() / .named() = the published sets
  species.py           SpeciesFlags + the model's quantum numbers
  thermodynamics.py    basis(), pressure/Omega, closed-form densities and s,
                       thermo_from_mu
  solver.py            the mode closures and the branch selection
  table.py             build_table: the warm-started sweep + progress callback
  api.py               eos_point / eos_table / eos_response
  verify/run_full_check.py
  pqm.tex, pqm.md      (ticket 12)
```

**Not present, each for a stated reason:** no `couplings.py` (a coupling is a
function of the STATE; these coefficients are functions of the PARAMETERS, so
they go in `parameters.py`); no `nmp.py` (no nuclear sector); no `backends/`
(the reference path IS the fast path -- there is no second implementation, and
section 5 defines `backends/` by the property that deleting it changes no
number); no `responses.py` unless `eos_response` outgrows `api.py`.

Templates to read rather than invent: `eos/alphabag/` is the closest existing
quark model with a `table.py`, and `eos/abpr/` shows the shorter shape. Both
define their own flat point record (`eos/alphabag/solver.py:80-133`,
`eos/abpr/solver.py:107-163`) rather than using a shared one; follow that. The
`__init__.py` re-export pattern is `eos/alphabag/__init__.py:25-69`. Register in
`MODELS` at `eos/__init__.py:45-46`.

### Section 1, which is the one that can actually be broken here

**`eos/pqm` imports `eos/general/` and nothing else in this repository.** The
notebook currently violates that twice (`notebooks/csc_bag_map.py:95-96`):

```python
from eos.alphabag.thermodynamics import fermi_thermo
from eos.mixed import njl_phase
```

- `fermi_thermo` is a four-line wrapper over
  `eos.general.fermi_integrals.solve_fermi_jel(mu, T, m, G_QUARK,
  include_antiparticles=True)` (`eos/alphabag/thermodynamics.py:218-230`).
  Take the general call directly.
- `njl_phase` is the DATA GENERATOR and must not cross into the package at all.
  The fitting code stays in `notebooks/`; the package consumes the fitted
  surfaces as data. That is what `docs/csc_bag_mapping.md:450-454` already
  prescribed: *"the fitting itself is not model code: it consumes `eos.njl` and
  produces parameters, so it belongs with the study that needs it, not inside
  either model."*
- `eos.general.pqcd.alpha_s` (ticket 02) is legal -- `general/` is the layer
  both may import.

`test/test_imports.py` enforces this and will catch a slip.

### Section 5's other requirements, each a thing to actually do

- **`thermodynamics.py` never knows which mode it is in.** The test is
  literal: grep it for `beta`, `Y_C`, `neutral`, `trapped` and find nothing.
  Mode conditions live in `solver.py`.
- **The public boundary is fm-based**: `n` in fm^-3, `T` and every `mu` in MeV,
  `eps` and `P` in MeV/fm^3, and `s` and `n_s` in fm^-3. Natural units stay
  inside and never cross a module boundary; a natural-units working record
  carries a leading underscore.
- **`par` first and never optional; `mode` required.** `pqm` has three modes,
  so it does not get `abpr`'s single-mode default.
- **The progress callback dict is the same in every model**:
  `{mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
  elapsed_s}`. Add keys, never rename them. Deep solver code never prints.
- **Non-convergence is a return value** at every public boundary, never an
  exception and never a hang -- a `PointResult(ok, message, point)` in the
  shape of `eos/alphabag/api.py:45-54`, with a bounded iteration count on every
  root find. This matters more than usual here: a sampler walking outside the
  fitted `(eta_D, G_V0)` box is a normal event (ticket 09), and so is a mode
  with no solution on a given branch.
- **No global mutable state**; the parameter object picklable so
  multiprocessing and MPI work.
- **Array in, array out** where the physics allows. `pressure()` over a grid of
  potentials is pure arithmetic and should vectorize completely; only the mode
  root finds are inherently per-point.

### Section 4: which flags, and which category each is

Every flag defaults to False. `photons`, `thermal_neutrinos` and `muons` are
ordinary defaults. `hyperons`, `deltas` and `thermal_mesons` RAISE -- this is a
quark model, the same statement `eos/njl/species.py:43-58` and
`eos/alphabag/species.py:90-100` make. `gluons` raises too: `alphabag` carries a
gluon gas and `pqm` does not, and section 4 says a sector that is off has its
flag False and a flag with one legal value is a STATEMENT.

**There is no `csc` flag and no pattern flag.** The pattern is an OUTPUT of
minimizing Omega, which is this model's whole premise; the pure branch is
reached by a `patterns=("CFL",)` restriction on the API, exactly as `eos.njl`
does, not by a flag and not by a mode.

### Section 12: the tests

`test/pqm/` named after the physics (`test_fixed_yc.py`, `test_branch_selection.py`,
`test_thermodynamic_identities.py`), plus a `test/baseline/pqm.npz` frozen at
rtol = 1e-10. `verify/run_full_check.py` carries the physics INVARIANTS rather
than unit behaviour:

- the Euler relation and both free-energy identities, which after ticket 01
  hold identically and should be asserted at 1e-12, not 1e-8;
- `0 <= c_s^2 <= 1` and P monotone on any CONSTRUCTED table -- section 8's
  gate, which belongs here because `pqm` builds tables a structure solver
  consumes, and which must NOT be asserted on a raw single-branch table inside
  a transition window;
- the conformal limit at two scales (ticket 02);
- the CFL locking identity `n_C = 0`, `n_S = n_B` as a consequence of the
  derivative, not a filter;
- the T = 0 limit of the finite-T forms as `T -> 0`.

**Before committing, run the suites the change can reach** -- `test/pqm/`,
`test/baseline/`, and `test/test_imports.py` -- and state which ran and why
those are the reachable set. Not the full suite: it is a landing measurement,
not a commit gate, and this checkout is shared and dirty.

## Gate

- The package exists in the layout above and imports clean.
- **`test/test_imports.py` passes**, with `eos/pqm` importing nothing outside
  `eos/general/`.
- `grep -E 'beta|Y_C|neutral|trapped' eos/pqm/thermodynamics.py` returns
  nothing.
- `test/pqm/` passes; `test/baseline/pqm.npz` frozen at rtol = 1e-10 and
  passing; `test/test_imports.py` and `test/baseline/` pass. What ran and why
  those are the reachable set, stated with the result.
- `verify/run_full_check.py` runs and asserts every invariant listed above.
- One measured point per mode reproducing the notebook's answer to 1e-10, so
  the port is proven to have changed no number.
- The section 9 speed numbers re-measured through the package API, since the
  notebook's per-point cost included notebook overhead.
