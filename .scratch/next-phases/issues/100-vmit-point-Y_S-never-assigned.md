# `eos.vmit.EoSPoint.Y_S` is never assigned, and the baseline froze the zero

Type: grilling
Status: resolved (2026-08-28)
Assignee: session e305b9b2
Blocked by: -
Parent: ../map.md

## Question

Found on 2026-08-27 while building [ticket 99](99-quark-ea-at-zero-pressure.md).

`eos/vmit/solver.py` never writes `result.Y_S` in **three of its four**
solvers. `solve_beta_eq_neutrinoless`, `solve_beta_eq_neutrino_trapped` and
`solve_fixed_yc` all set `result.Y_C = q_thermo.Y_C` and leave `Y_S` at the
dataclass default of `0.0`. Only `solve_fixed_yc_ys` has it, and only because
the mode's own `Y_S` argument goes into the constructor.

So a caller reading `eos.vmit.eos_point(...).point.Y_S` gets `0.0` at every
density, on every set, in every beta-equilibrium and fixed-`Y_C` state. At
n_B = 0.5 fm^-3 the shipped set has `n_s = 0.4255 fm^-3` -- `Y_S = 0.851` --
and the point reports `Y_S = 0.0000` beside it.

`eos.alphabag` sets the field correctly (`solver.py:313`), so this is vMIT's
alone.

### It is frozen into the baseline

`test/baseline/vmit.npz` holds **`beta.T0.n0.45.Y_S = 0.0` next to
`beta.T0.n0.45.Y_s = 0.8402`** -- the contradiction, pinned at rtol = 1e-10.
Every `*.Y_S` key of the beta, trapped and `yc` families is zero; only the
`ycys` family is right. So fixing the field MOVES A FROZEN BASELINE, which is
why ticket 99 did not fix it: that ticket's Gate says "no number moves
anywhere else", and this is the other side of the same coin from
[ticket 62](62-regenerate-baselines-py314.md).

### Why nothing caught it

`eos/vmit/table.py:88` recomputes `Y_S=result.n_s / n_B` for its rows, so
every TABLE is right and every table-driven check passes. The defect lives
only on the point object, which the baselines sample directly.

### What has to be decided

- Whether the field is fixed and `vmit.npz` regenerated, or the field is
  removed from the point in favour of the basis map every reader should use.
  Ticket 99 took the second position for its own reader without ruling on
  this one: `zero_pressure_point` computes `Y_S` from
  `eos.general.basis.quark_charges` on the solved flavour densities in EVERY
  quark model, which is both uniform and immune to a cached field going
  stale. If that is the right reader everywhere, a cached `Y_S` on the point
  is a second home for a derived quantity and CLAUDE.md section 7 argues
  against it.
- Whether any OTHER model caches a conserved fraction that its solver forgets
  to fill. `Y_C`, `Y_L` and `Y_S` are all cached fields on nine point classes
  and only `vmit.Y_S` was checked.
- The blast radius on read: `eos/mixed/backends/jacobian.py:148` reads
  `st.Y_S`, but of a MIXED state rather than a vMIT point, so it is not
  affected. Nothing else in the package reads a vMIT point's `Y_S`.

### It already cost a wrong finding

Ticket 99's own measured table recorded "**`vmit`'s default set is already
two-flavour at its surface** (Y_S = 0.0000 at n_B = 0.4404): the s quark is
unpopulated below its threshold". That is false. Read through the basis map
the same surface has **Y_S = 0.8379**, and the two-flavour E/A is a genuinely
different number, 1236.02 MeV against the three-flavour 1155.75 MeV. The
finding survived into the ticket as a warning to implementers and had to be
retracted when the flag it was arguing about was built.

## Gate

- Every quark model's point reports the strangeness fraction its own solved
  densities carry, or does not carry the field at all.
- Whichever way it goes, the `vmit.npz` keys that change are named, and the
  regeneration is the deliberate act [ticket 62](62-regenerate-baselines-py314.md)
  made it.
- A check that would have caught this: a cached fraction on a point agrees
  with `eos.general.basis` on the same point, in every model.

---

## Resolution (2026-08-28)

**The field is FIXED, not removed, and `vmit.npz` was regenerated — 39 keys,
all of them `.Y_S`, nothing else in the file moved.** Measure-then-regenerate,
in that order: the deltas were computed and read before a line of `vmit`
changed, so the regeneration confirmed a prediction rather than blessing
whatever the code happened to produce.

### 1. First, the sweep the ticket's second bullet asked for

Every model's point, every mode it exposes, default flags, n_B = 0.5 fm^-3,
T = 0, each cached fraction against `eos.general.basis` on that point's own
solved densities (`quark_charges` for the quark models, `charges_from_densities`
for `zl`, the point's own `n_C`/`n_S` for `did`). `dd2`, `sfho` and `enjl`
cache no fraction at all — they carry densities and nothing derived — so there
was nothing in them to be stale.

**On `Y_S` the sweep found `vmit` alone**, exactly as the ticket claimed:

| model | mode | cached `Y_S` | basis `Y_S` |
|---|---|---|---|
| `vmit` | `beta_eq_neutrinoless` | 0.00000000 | 0.85103848 |
| `vmit` | `beta_eq_neutrino_trapped` | 0.00000000 | 0.73437461 |
| `vmit` | `fixed_YC` | 0.00000000 | 0.70896633 |

Six others measure it correctly in every mode. **On `Y_L` it found three** —
`zl`, `vmit` and `alphabag`, all outside `beta_eq_neutrino_trapped` — and per
the instruction that is a SECOND TICKET and not a wider diff here:
[ticket 108](108-cached-lepton-fraction-three-models.md). It is a different
question, not the same one twice: `zl` and `alphabag` annotate the field
`# electron-family lepton fraction (trapped mode)`, under which reading their
zeros are correct, while `njl`, `ccdm` and `did` measure the lepton fraction in
every mode. Nobody has ruled which job the name names, so there was nothing to
fix — only something to decide.

### 2. Then the fix, four lines

`result.Y_S = q_thermo.Y_S` in all four solvers of `eos/vmit/solver.py`
(`:333`, `:489`, `:634`, `:744`), beside the `result.Y_C = q_thermo.Y_C` the
two beta solvers already had. `thermodynamics.py:361` already computes that
`Y_S` from `basis.quark_charges` on the solved flavour densities, so this
routes the point through the same map `table.py` and `zero_pressure_point`
were already using — no second implementation, and CLAUDE.md §7's single-home
rule is satisfied by the value having one source rather than by the field
being deleted.

`solve_fixed_yc_ys` is included on purpose although its constructor already
carried the requested `Y_S`. The Gate says the point reports what its own
solved densities carry, and that is now true in four modes out of four rather
than three; it is also what `alphabag` does, which reports 3.58e-12 where 0.0
was asked for. It moved nothing: the three `ycys` keys are at Y_S = 1.0 to
better than 1e-10 either way.

### 3. The regeneration, named key by key

`test/baseline/vmit.npz`, 1119 keys. **Before regenerating, the current code
was checked against the stored file and reproduced it exactly — 0 keys moving
with no code change** — so every delta below is attributable to this fix and
to nothing that drifted in beside it.

**39 keys moved, every one of them a `.Y_S`, all from 0.0:**

- `beta.T{0,10,30}.n{0.3,0.45,0.6,0.8,1,1.3}.Y_S` — 18 keys, 0.7909 to 0.9221
- `yc.{lep,nolep}.YC{0,0.3,0.5}.n{0.45,0.8,1.3}.Y_S` — 18 keys, 0.6056 to 0.9212
- `trapped.n{0.45,0.8,1.3}.Y_S` — 3 keys, 0.7242 to 0.8044

**Nothing else moved.** No potential, no density, no P, eps or s: the field was
write-only, so nothing downstream of it inside the model had ever read the
zero. `beta.T0.n0.45.Y_S` now reads 0.8402368306, next to the `Y_s = 0.8402`
that was already in the file — the contradiction the ticket named, gone.

**Nothing outside `eos` is affected.** `nucleation` couples to `eos.alphabag`,
never to `eos.vmit`, and `alphabag`'s field was always right;
`eos/mixed/backends/jacobian.py:148` reads a MIXED state's `Y_S`, as the ticket
said.

### 4. The check that would have caught it

`test/test_cached_fractions.py`, new, cross-model: one case per (model, mode),
asserting that a point's cached `Y_C` and `Y_S` equal `eos.general.basis` on
that same point's densities. **Verified red before green** — backing the four
lines out fails it with `Obtained: 0.0, Expected: 0.8510384763983517`.
21 passed, 3 skipped, the skips all pre-existing non-convergence reported with
the residual reached rather than hidden (`ccdm` in two modes below its
deconfinement onset, `did` in `fixed_YC_YS`, whose default flags carry no
strange species to hold Y_S = 0.2).

A second test in the file refuses to let a new model slip past the enumeration:
if a point carries `Y_C` or `Y_S` and the model is not in the table, it fails
saying so. The file's docstring states why `Y_L` is excluded, and names
ticket 108 as what would remove that paragraph.

Tolerance is 1e-8, not round-off, and deliberately so: a fixed-fraction mode
reports what it SOLVED (`alphabag` lands on 0.29999999998989824 for a
requested 0.3). What is being caught is a field never written, which misses by
the whole fraction. That is not the loosening CLAUDE.md forbids — there was no
prior assertion to loosen.

### 5. The wrong finding, retracted where it still lived

`eos/general/zero_pressure.py:106` still carried ticket 99's retracted example
as fact — "`eos.vmit`'s default set has the s quark below its threshold at its
own surface, so a three-flavour REQUEST returns Y_S = 0". Measured this
session through the public API: the surface is at n_B = 0.4404 fm^-3 with
**Y_S = 0.8379**, E/A = 1155.75 MeV, and the genuine two-flavour arm is a
different point, n_B = 0.3970, E/A = 1236.02. The docstring now states the
general case, says Y_S is measured through the basis map and never off a
cached field, and records that the example it used to give was wrong and why.

`eos/vmit/vmit.md:613` and `vmit.tex:865` listed `Y_C, Y_S, Y_L` together as
"the fractions the mode fixed", which was the sentence the code was
implementing. `Y_S` now has its own row saying it is measured on the solved
flavour densities in every mode.

### Gate

- [x] Every quark model's point reports the strangeness fraction its own
      solved densities carry — `vmit` now in all four modes, the six others
      already did, checked by test rather than by reading.
- [x] The `vmit.npz` keys that change are named — 39, all `.Y_S`, listed above
      — and the regeneration was deliberate: predicted first, then confirmed.
- [x] The check exists, is cross-model, and was verified to go red without the
      fix.
