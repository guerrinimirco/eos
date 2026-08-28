# `Y_L` on a point is two different quantities wearing one name

Type: grilling
Status: resolved (2026-08-28)
Assignee: session dc4b25ab
Blocked by: -
Parent: ../map.md

## Question

Surfaced by [ticket 100](100-vmit-point-Y_S-never-assigned.md)'s first
instruction: **establish whether any other model has the same unassigned-field
shape before touching vmit.** It does, on a different field, and the answer is
a ruling rather than a fix — which is why it is here and not in that diff.

Ticket 100 swept every model's point against `eos.general.basis` on that
point's own solved densities, at n_B = 0.5 fm^-3, T = 0, every mode, default
flags. On **`Y_S` the sweep found `vmit` alone**, exactly as ticket 100
claimed. On `Y_L` it found three:

| model | mode | cached `Y_L` | (n_e + n_mu + n_nu)/n_B |
|---|---|---|---|
| `zl` | `beta_eq_neutrinoless` | 0.00000000 | 0.11398814 |
| `zl` | `fixed_YC` | 0.00000000 | 0.30000000 |
| `vmit` | `beta_eq_neutrinoless` | 0.00000000 | 0.00003555 |
| `vmit` | `fixed_YC` | 0.00000000 | 0.30000000 |
| `vmit` | `fixed_YC_YS` | 0.00000000 | 0.30000000 |
| `alphabag` | `beta_eq_neutrinoless` | 0.00000000 | 0.00005545 |

In all six the leptons are present and the field reads the dataclass default.

## Why it is NOT the same ticket as 100

Because `Y_S` had one meaning and `Y_L` has two, and nobody has ruled which.

- `eos/zl/solver.py:79` and `eos/alphabag/solver.py:97` both annotate the
  field `# electron-family lepton fraction (trapped mode)`. Read that way the
  zeros above are not a bug: outside `beta_eq_neutrino_trapped` there is no
  Y_Le condition, the field is inapplicable, and both models expose the
  measured lepton content as `Y_e` beside it.
- `eos/njl/solver.py:473` and `eos/ccdm/solver.py:520` compute
  `Y_L = (n_e + n_nu)/n_B` in **every** mode, so `njl` reports Y_L = 0.3 in
  `fixed_YC`. Read that way the zeros are the same defect ticket 100 fixed.
- `eos/did/solver.py:135` calls the field `Y_Le` and measures it in every
  mode — a third spelling of the same slot.
- `eos/vmit/solver.py:72` says only `# Lepton fraction`, committing to
  neither.

CLAUDE.md §13 rule 2 — the same job carries the same name in every model — is
violated whichever way it goes, so the decision is what the job IS:

- **(a) `Y_L` is the trapped mode's condition, echoed back.** Then `njl`,
  `ccdm` and `did` are the ones that drift, and the field should be absent or
  documented as inapplicable outside the mode that fixes it. Cheapest, and it
  makes the zeros correct as they stand.
- **(b) `Y_L` is the measured lepton fraction, in every mode.** Then it is
  ticket 100's defect three more times, `zl`/`vmit`/`alphabag` assign it from
  their own lepton densities, and one spelling wins over `Y_Le`.
- **(c) the field is derived and should not be cached at all.** CLAUDE.md §7's
  single-home argument, the position ticket 99 took for its own reader. Every
  point already carries `n_e`, `n_mu` and `n_nu`, so `Y_L` is one division the
  caller can do — and a second home for a derived quantity is precisely what
  went stale in ticket 100.

Note that `Y_C` has a milder version of the same question: `vmit`'s and
`zl`'s fixed-fraction solvers echo the requested `Y_C` while `alphabag`'s
report the solved one (0.29999999998989824 against a requested 0.3). Ticket
100 left that alone deliberately — the values agree to 1e-8, so nothing is
wrong today — but whichever way `Y_L` is ruled should say what `Y_C` is too,
since they sit two lines apart in the same dataclass.

## Blast radius

**Arm (b) moves `test/baseline/vmit.npz` and `zl.npz` a second time.** Ticket
100 has already regenerated `vmit.npz` once (39 `.Y_S` keys). So this ticket
**must not run concurrently with [ticket 95](95-vmit-solver-flags.md)**, which
moves vMIT rows for its own reason, and the two regenerations should be
sequenced rather than interleaved.

`eos/mixed/backends/jacobian.py:148` reads `st.Y_S` of a MIXED state, not of
any point above, so it is unaffected. Nothing in the package reads a `Y_L`
off one of these points.

## Gate

- One meaning for `Y_L` across the ten models, stated in CLAUDE.md §13's
  vocabulary list or in each dataclass, and `Y_Le` reconciled with it.
- `test/test_cached_fractions.py` extended to cover the field. It skips
  `Y_L` today, and says in its module docstring exactly why: there is no
  convention to assert yet. That paragraph comes out when this ticket lands.
- Any baseline key that moves is named, and the regeneration sequenced
  against ticket 95.


## Resolution (2026-08-28) — arm (b), respelled: `Y_Le`, measured, every mode

**Neither camp was spelled right.** Section 2 already decides the question:
"Y_X = n_X / n_B for every charge: Y_C, Y_S, Y_Le, Y_Lmu", and L_e is a
conserved charge of that section's reduced basis (B, C, S, L_e, L_mu). So the
lepton fraction is a property of the SOLVED STATE, defined in every mode,
exactly as Y_C is — `beta_eq_neutrino_trapped` holds it the way `fixed_YC`
holds Y_C, and nobody claims Y_C is undefined outside `fixed_YC`. Arm (a)
would have made Y_Le the one fraction in the vocabulary that is not a state
property. Arm (c) was refused for the narrower reason that the trapped mode's
own condition would stop being readable off its result while Y_C and Y_S
stayed cached, splitting the vocabulary rather than unifying it.

`Y_L` appears **nowhere in CLAUDE.md**. The conformant models were the two the
ticket did not list among either camp — `did` and `dd2`, both already `Y_Le`
and both already measuring it — and `eos/mixed` had ruled the same way
already: `test/mixed/test_mixed_api.py` carries
`test_the_lepton_fraction_has_exactly_one_name`, which asserts `Y_Le` is the
condition name and `Y_L` is refused. This ticket brings the five stragglers to
where the engine already was.

### What changed

- **`eos/general/basis.py`** gains `lepton_charges(n_e, n_nue, n_mu, n_numu)
  -> (n_Le, n_Lmu)` — the single home (section 7) for the map from lepton
  densities to the two family charges, and where the convention is now stated.
- **`zl`, `vmit`, `alphabag`**: field `Y_L` -> `Y_Le`, and it is ASSIGNED from
  the point's own lepton densities in every mode instead of carrying the
  dataclass default. `alphabag`'s `point.Y_L = Y_Le` echo (solver.py:574) is
  gone — the trapped mode now reports what it solved, like every other mode.
- **`njl`, `ccdm`**: rename only. Both already measured the right quantity.
- **`vmit`**: the trapped solver's condition kwarg was also `Y_L`; it is
  `Y_Le`, which is what section 5 names and what `vmit/api.py` already spoke
  at its own boundary. All callers pass positionally, so nothing else moved.
- **`eos/vmit/compute_tables.py:184`** listed `'Y_L'` in the attribute names
  `results_to_arrays` reads — the one place the rename would have broken
  silently.
- **Documents** (section 11): `zl.md`, `vmit.md`, `alphabag.md`, `njl.md`,
  `ccdm.md`, `njl.tex`. `vmit.md` also had the row described as "the fractions
  the mode fixed", which the ruling makes false; it now says measured.

### On `Y_C`, which the ticket asked be settled alongside

Same ruling, and **no code change**: a fraction is measured on the state, and
an echo is legitimate only where it is exact by construction (`zl`'s fixed_YC
sets n_p = Y_C n_B, so echo and measurement agree to round-off). `alphabag`
reporting 0.29999999998989824 was always the correct behaviour rather than the
outlier, and the 1e-8 comparison in `test/test_cached_fractions.py` is what
keeps the distinction honest without forbidding the echo.

### Gate

- `test/test_cached_fractions.py` no longer skips the field: its module
  docstring's "there is no convention to assert yet" paragraph is replaced by
  the section 2 argument, `Y_Le` joins `Y_C`/`Y_S` in the comparison loop, and
  it joins the set in `test_no_model_was_left_out_of_the_table_above` so a new
  model caching it must join CHARGES. **21 passed, 3 skipped** (the three
  pre-existing non-convergence skips).
- The ticket's own six-row table now agrees on both sides —
  `zl` `beta_eq_neutrinoless` reports **0.11398814** against the recomputed
  0.11398814.
- `test/zl test/vmit test/alphabag test/njl test/ccdm test/mixed test/general
  test/test_cached_fractions.py test/test_imports.py`: **932 passed, 3
  skipped, 3 xfailed** (8m25s). `test/baseline`: **20 passed**.

### The baselines, key by key

Regenerated `zl`, `vmit`, `alphabag`, `njl`, `ccdm` (`test/` is gitignored, so
this is a local freeze, not a committed one). Audited against a before-image:

| model | `.Y_L` -> `.Y_Le` renamed | of those, value moved | OTHER keys moved |
|---|---|---|---|
| `zl` | 57 | **39** (beta 27, yc 12) | 0 |
| `vmit` | 42 | **29** (beta 18, yc 9, ycys 2) | 0 |
| `alphabag` | 36 | **23** (beta 18, yc 5) | 0 |
| `njl` | 105 | 0 | 0 |
| `ccdm` | 76 | 0 | 0 |

**316 keys renamed, 91 changed value, and NOTHING ELSE MOVED** — every removed
key had a `.Y_Le` twin added, and no other key in any of the five files
differs at rtol = 1e-10. Every one of the 91 is a frozen wrong value
corrected: the field was never written, so it shipped 0.0 while the point's
own leptons said otherwise. `njl` and `ccdm` move zero values because they
were already right.

Of the 91, **84 move by more than 1e-8** and **7 move from an exact 0.0 to a
number of size 1e-11 to 1e-21** — all seven are `Y_C = 0` states with leptons
on (`yc.lep.YC0.*`, `ycys.*`), where the electron density solves to the
solver's own residual rather than to a hard zero. They are correct measured
values, but they are also a relative comparison of a quantity that is zero by
construction, which is the shape ticket 76 warned about; noted below rather
than fixed here.

### Two corrections to this ticket's own framing

1. **The blast radius was understated.** It named `vmit.npz` and `zl.npz`; it
   missed `alphabag.npz` (23 value moves) and the 181 name-only moves in
   `njl.npz` and `ccdm.npz`.
2. **"Nothing in the package reads a `Y_L` off one of these points"** is true
   of the package and false of the suite: `test/ccdm/test_ccdm_modes.py:40`
   asserted `p.Y_L`, and `test/alphabag/test_alphabag_modes.py:64` asserted
   `r.Y_L == Y_Le` by exact equality — which the ruling turns into a measured
   comparison, so it is now `pytest.approx(..., abs=CLOSURE_TOL)`.

Also: the table's `0.30000000` column for the `fixed_YC` rows was recorded
before [ticket 91](91-leptons-default-and-drift-checks.md) made `leptons`
default to False. Under today's defaults those rows read 0.0 on BOTH sides,
because there are no leptons; at `leptons=True` both sides read 0.3. The
lepton arms are why only half the frozen `yc` zeros moved.

### Deliberately left alone

- The `'Y_L'` **column header** written by `eos/zl/table.py:320` and
  `eos/vmit/compute_tables.py:146`. Those label a written text table, take
  their value from the grid key rather than from the point, and `nucleation`
  carries its own unrelated `GRIDS['Y_L']` vocabulary. Renaming an output
  format is a separate decision with a downstream blast radius; it is on the
  map as fog.
- `eos/zlvmit/`, exempt under section 1, and its `Y_L_input` baseline keys.
- No model caches `Y_Lmu`. The unsuffixed name invited summing families while
  every implementation summed only the electron one; `Y_Le` is honest today
  because `zl`/`vmit`/`alphabag` carry no muon fields at all, but the muon
  family number is simply not reported anywhere. Recorded, not fixed.
