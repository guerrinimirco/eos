# The SU(6) vector ratios become parameters, and the Delta ratios become free
# variables

Type: task
Status: resolved (2026-08-29)
Blocked by: -   (103 ruled it; 106 folded in and closed as superseded)
Parent: ../map.md

## Question

Ruled by the user on [ticket 103](103-nmp-closures-four-models.md)
(2026-08-29). Today the hyperon VECTOR coupling ratios are hardcoded at their
SU(6) values (`sfho/parameters.py:37`, dd2's hyperon constructor) and only the
SCALAR ratios are inverted from the single-particle depths. SU(6) is an
assumption, not a measurement, and CLAUDE.md section 6 is unambiguous about
what that costs: "a parameter that can only be changed by editing a source
file makes inference impossible".

**The ruling.** For every RMF model with a hyperon sector:

- `x_sigma_H` stays fixed by the potential depth `U_H`, as now.
- `x_omega_H`, `x_rho_H`, `x_phi_H` become **SU(6) x y**, with `y` an ordinary
  parameter on `Parameters` defaulting to 1.0. **Nine factors**, one per
  (meson, multiplet) pair over M = omega, rho, phi and H = Lambda, Sigma, Xi
  -- named fields, not a dict, so a sampler varies them by name and section 13's
  explicit form is kept.
- The **Delta sector takes no factors**: its couplings are the free variables
  `x_Delta_sigma`, `x_Delta_omega`, `x_Delta_rho` directly.

The shipped sets are the test that nine is the right count and not three:
SFHoY is `y = 1.5` on omega AND phi for Lambda and Sigma, and `y = 1.875` on
both for Xi, with the rho ratios left at SU(6) (`sfho/parameters.py:391-410`);
SFHoY* is all ones. A per-meson factor shared across multiplets cannot express
that; a per-multiplet factor shared across mesons cannot express a set that
breaks omega and phi differently.

One documentation point the ruling names: `y_phi_H` multiplies a NEGATIVE
SU(6) ratio (-sqrt(2)/3 for Lambda and Sigma, -2sqrt(2)/3 for Xi), so a factor
above one makes the coupling more negative. Say so where the field is defined,
or a reader takes "1.5x" for "more repulsion".

## Gate

Every published hyperonic set reproduces its current couplings exactly through
the new fields; `SFHoY` and `SFHoY*` are expressed as factor sets rather than
as absolute numbers; the per-model documents state the nine factors and the
Delta free variables; every model's `verify` and `test/<model>` green with no
baseline moved. [Ticket 106](106-su6-breaking-rescaling.md) is unblocked by
this and is where the rescaling itself is decided.


## Resolution (2026-08-29)

**Shipped, and [ticket 106](106-su6-breaking-rescaling.md) folded in and closed
as superseded first.** 106 was this same work charted from the physics side a
day earlier and neither file pointed at the other; its three traps are the
substance of what follows, and its two proposals that the ruling had already
overtaken (three factors rather than nine, a new `Parameters.named(...)` rather
than re-expressing the sets that exist) are recorded there as not surviving.

### The nine factors, in both RMFs with a hyperon sector

`Parameters` in `dd2` and `sfho` each gain the same nine fields, defaulting to
1.0, one per (meson, multiplet) pair:

    y_omega_Lambda  y_omega_Sigma  y_omega_Xi
    y_rho_Lambda    y_rho_Sigma    y_rho_Xi
    y_phi_Lambda    y_phi_Sigma    y_phi_Xi

with `x_MY = y_M_Y * SU(6)` evaluated by `couplings.vector_ratios` (dd2) and
`parameters.vector_ratios` (sfho), and a `su6_breaking` property returning the
3x3 table. `x_sigma_H` is unchanged: it stays the one hyperon ratio fitted to a
measurement, and stays stored.

**They are not inert record-keeping, and that took a small structural change in
each model.** A field a sampler can set but that nothing reads is the second
way to say one thing §13 forbids, so the derived couplings were removed rather
than duplicated:

- `dd2`'s `hyperon_couplings` rows go from six columns to **three** --
  `(name, mass, x_sigma)`. `hyperon_coupling_map` still returns the same
  5-tuple `(mass, x_sigma, x_omega, x_rho, x_phi)`, now computed, so
  `thermodynamics.py` is untouched. `has_phi_coupling` reads the derived
  column.
- `sfho`'s `couplings_map` holds **only `'sigma'`** for a hyperon;
  `get_coupling` supplies omega/rho/phi from SU(6) x y. Deltas keep all four
  entries, which is also what keeps `species.check_couplings`' zero-coupling
  guard meaningful -- an entry's presence is still what declares the sector.

So `replace(par, y_omega_Lambda=1.5)` moves the coupling in both models, which
is what "a sampler varies them by name" has to mean.

### 106's trap 1: the factors are read out of Fortin et al., not inferred

Read at the source (arXiv:1711.09427v2 §2.2, Fortin, Oertel & Providencia,
PASA 35 (2018) e044): *"We therefore rescale the omega and phi-meson hyperon
couplings as follows: g_MLambda = 1.5 g_MLambda(SU(6)),
g_MSigma = 1.5 g_MSigma(SU(6)), g_MXi = 1.875 g_MXi(SU(6))."* Their Table 1
gives the result, and it is now a test fixture:

    Model     R_sL R_wL R_pL R_rL  R_sS R_wS R_pS R_rS  R_sX R_wX R_pX R_rX
    DD2Y      0.62  2/3 -0.47  0   0.48  2/3 -0.47  2   0.32 0.33 -0.94  1
    SFHoY     0.85  1   -0.71  0   0.58  1   -0.71  2   0.51 0.62 -1.77  1
    SFHoY*    0.61  2/3 -0.47  0   0.35  2/3 -0.47  2   0.30 0.33 -0.94  1

Three things the paper settles that the ticket could only guess at. **M ranges
over omega AND phi**, so the pattern really is per-(meson, multiplet) and
**rho is untouched** -- the row that makes nine the right count and not three.
**The Xi factor is 1.875 and not 1.5**, and 1.875 x 1/3 = 0.625 exactly, which
their table prints as 0.62; the test tolerance against Table 1 is 1e-2 and is
the paper's print precision, nothing else. And DD2Y's row is all-SU(6), which
is why every dd2 factor defaults to 1.

The documentation point the ruling asked for is made where each field is
defined and in both model documents: **y_phi multiplies a NEGATIVE ratio**, so
a factor above one makes g_phiY more negative rather than more repulsive-
looking.

### 106's trap 2: the inversion runs AFTER the rescaling

U_Y = -g_sigmaY sigma + g_omegaY omega (+ Sigma^R in dd2) holds both couplings,
so a rescaled vector coupling changes the scalar coupling that reproduces the
same depth. The API makes the wrong order unreachable rather than warning about
it: both constructors take a `base` carrying the factors and close the depths
on it, in one call --

    base = replace(Parameters.default(), y_omega_Lambda=1.5, y_phi_Lambda=1.5, ...)
    par  = from_hyperon_potentials(U_Xi=-14.0, base=base)      # dd2
    par  = from_potential_depths(U_Xi_N=-14.0, base=base)      # sfho

-- and each model's `verify/` gained an entry that rebuilds a broken set from
its own factors and requires the depths back. Measured: dd2 holds U_Y to
**6.4e-14 MeV** on a base broken at 1.5/1.875, and reproduces DD2Y's SU(6)
column to **0.0e+00**; sfho holds it to **1.4e-14 MeV** and lands within
**1.8e-04** of the shipped `SFHoY_Fortin` couplings, which is the rounding of
that set's published six-digit R_sigma column and not a disagreement. The
counter-case is tested too: `replace(par, y_omega_Lambda=1.5)` on a finished
set moves U_Lambda by more than 1 MeV, which is exactly the silent failure the
one-call shape prevents.

### 106's trap 3: "depths or ratios" is an argument

`sfho.invert_nmp` now **raises** on a base carrying hyperons unless told which:
`hold_hyperons='ratios'` holds g_sigmaH/g_sigmaN and lets U_Y move;
`hold_hyperons='depths'` holds U_Lambda/U_Sigma/U_Xi and re-inverts the scalar
couplings on the new nucleon sector. A nucleonic base has no such question and
needs no answer, so no existing call changed. On `SFHoY_Fortin` at SFHo's own
published NMPs:

    arm       R_sigma_Lambda      U_Lambda
    base      0.854315            -30.035
    ratios    0.854315  (held)    -29.864  (moved)
    depths    0.854915  (moved)   -30.000  (held)

`docs/DEFERRED.md`'s entry for this is rewritten from a deferral to a
description of the argument. **One real bug fell out of building it**:
`compute_hyperon_potentials` called `compute_saturation_fields()` with no
argument, so it read the depths at *nucleonic SFHo's* saturation point rather
than the parametrization's own -- invisible for the published sets, which are
all built on that base, and wrong by 0.06 MeV for an inverted one. Fixed;
no shipped number moves.

### What else moved, and why it belongs here

- **`from_hyperon_potentials(x_phi=...)` is gone**, replaced rather than
  doubled, as [ticket 102](102-retire-phi-field-flag.md)'s ping asked. The
  reachable way to say "no phi" is `y_phi_Lambda = y_phi_Sigma = y_phi_Xi = 0`,
  which is strictly more expressive (per-multiplet) and works by `replace` on
  any par. `test_imports.py`'s phi-sector check, `test_dd2_m4`'s, CLAUDE.md §4,
  `docs/STRUCTURE.md` and the `hybrid_eos` notebook's retired-run rebuild were
  all moved onto it.
- **`sfho.parameters.from_coupling_ratios` deleted.** It was a second
  constructor for the job `from_potential_depths` does, reached by no caller
  outside the module's own `__main__` demo, not exported from `eos.sfho`, and
  it hardcoded the paper's ROUNDED vector column (-0.71, 0.62, -1.77) behind a
  `use_scaled_vectors` boolean -- absolute numbers and a two-valued flag where
  this ticket puts factors. Its Delta arguments are the ones the ruling wanted
  freed, and they live on `from_potential_depths` now.
- **`x_wD` / `x_rD` renamed to `x_Delta_omega` / `x_Delta_rho`** in dd2's
  `from_delta_potential`, `SECTOR_KEYS` and `build_parametrization`. §13 rule 2:
  the same job carries the same name in every model, and this ticket was
  putting `x_Delta_omega` on sfho's side. The dd2 `Parameters` fields already
  carried those names, so this only aligns the constructor with them; sample
  dicts using the old keys are the one caller-visible break, and every in-repo
  site was updated including `test/baseline/generate_baseline.py`.
- **The Delta sector takes no factors**, as ruled: `x_Delta_sigma`,
  `x_Delta_omega`, `x_Delta_rho` are free variables directly in both models,
  and `MULTIPLET` deliberately contains no Delta entry.

### Documents

`dd2.tex`/`dd2.md` and `sfho.tex`/`sfho.md` state Eq. (su6break), the nine
factors, the sign point about y_phi, the after-not-before order of the
inversion, and the Delta free variables. `docs/STRUCTURE.md` §6's hyperon
paragraph is rewritten from "the vector ratios are taken from SU(6)" to the
factor form, with the depths-or-ratios sentence. Ticket 103's closure table is
untouched -- this sector was never a row in it.

### Gate

- **y = 1 reproduces every current number bit-for-bit.** Checked directly
  rather than asserted: HEAD's `sfho/parameters.py` and `dd2/parameters.py`
  were loaded side by side with the working tree's and every coupling of every
  published set compared with `==`. **dd2 (DD2, DD2Y) and sfho (SFHo_Nucleonic,
  SFHoY, SFHoY*, SFHo_2fam_phi, SFHo_2fam): bit-identical, zero mismatches.**
  The arithmetic is exact and not merely close -- `1.5 * (2/3) == 1.0` and
  `1.875 * (1/3) == 0.625` in IEEE doubles, which is why SFHoY could be
  re-expressed as a factor set without moving a bit.
- **SFHoY and SFHoY\* are factor sets**, not absolute vector numbers: their
  builders set the nine fields and store only the scalar couplings.
- **The named breaking set reproduces its published depths through
  `compute_hyperon_potentials`**: rebuilt from its factors it returns
  -30 / +30 / -14 MeV to 1e-14. (The SHIPPED `SFHoY_Fortin` returns
  -30.035 / +29.998 / -14.019, because its R_sigma column is the paper's
  six-digit rounding -- a property of the published table, not of this change.)
- **A rescaling moves g_omega_Y and the scalar follows at fixed depth**: one
  test per model, plus the naive-rescale counter-case.
- Per-model documents state the nine factors and the Delta free variables.

### What was actually run, and over what

[Ticket 117](117-suite-gate-needs-a-landing-point.md) has **not** ruled — it is
open and claimed by another session — so this reports the subsets rather than
claiming a certified full suite, which is arm (b) of that ticket practised
before it is decided.

    python3 -m pytest test/dd2 test/sfho test/mixed test/tov test/baseline \
                      test/test_imports.py -q -p no:randomly
    -> 799 passed, 15 skipped, 0 failed in 881s (14:41)

    python3 -m pytest test/dd2/test_dd2_su6_breaking.py \
                      test/sfho/test_sfho_su6_breaking.py test/test_imports.py -q
    -> 229 passed in 6.3s

    eos.dd2.verify.run_full_check()   -> PASS, 12 checks
    eos.sfho.verify.run_full_check()  -> PASS, 10 checks

Interpreter: python.org 3.14.2, the repo run in place (not pip-installed).
**`test/baseline` is inside the first run and passed**, which is the "no
baseline moved" half of the gate; the bit-identity half was checked directly
against HEAD rather than inferred from it.

**Two honesty notes about that first run, since a count that is not qualified
is the thing this map keeps catching.** (1) It covered the tree as of its
start: no `eos/*.py` was written during it — the newest, `eos/dd2/species.py`,
predates the start by ~22 s — but the two new test files had an unused import
deleted mid-run, which is why the second command re-runs exactly those.
(2) Two OTHER pytest runs from a concurrent session (`pytest test -q` and a
seven-directory subset) were live throughout, so the wall-clock is contended
and says nothing; and that session was also editing `eos/mixed/*` and
`test/*` before my run began. Nothing it touched is in this ticket's diff, but
"799 passed" is a measurement of a shared tree and should be read as one.

**Not run**: `test/zl`, `test/vmit`, `test/alphabag`, `test/did`, `test/njl`,
`test/ccdm`, `test/enjl`, `test/general`, `test/gmode` and the remaining
top-level `test/*.py`. The diff touches `eos/dd2`, `eos/sfho`, `CLAUDE.md`,
`docs/`, and one notebook; the only cross-model surface in it is
`test/test_imports.py`, which ran.
