# The S_i / C_i undetermined-potential check, now that general/verify/ exists

Type: task
Status: resolved
Blocked by: -
Parent: ../map.md

## Question

Graduated from the map's Not-yet-specified section, which held it as fog for
two reasons that have both now cleared:

- **It had no home.** `eos/general/verify/` did not exist. It does now —
  [ticket 64](issues/64-general-verify-suite-missing.md) built it.
- **It had one witness.** [Ticket 56](issues/56-baseline-empty-sector-gate.md)
  measured it once, in `sfho`, after the fact. It now has a second, taken
  independently and predictively.

### What ticket 62 measured

[Ticket 62](issues/62-regenerate-baselines-py314.md) applied the screen as its
first pass over 53763 baseline keys across two Python stacks, and it separated
the whole set correctly:

| model | shift | carried by | verdict |
|---|---|---|---|
| `ccdm` | 5.572e-05 | `mu_3`, `mu_C`, `x`, `mu_modes`, `mu_star` | undetermined |
| `ccdm` | 2.786e-05 = **half** | `mu_8` | its projection coefficient |
| `njl` | 1.291e-05 | `mu_3`, `mu_C`, `x`, `mu_modes`, `mu_star` | undetermined |
| `njl` | 6.453e-06 = **half** | `mu_8` | its projection coefficient |
| `enjl` | 7.3e-04 MeV | `mu_S` at Y_S = 0, every other `x` entry identical | undetermined |
| `enjl` | O(1) fractions | masses, densities, P, eps, mu_S together | **physics** |

Two independently written models produced the same exact-ratio structure in the
CFL pattern, where §3 says the locking leaves no free charge fraction — so
`mu_3 = mu_C` is undetermined and `mu_8` carries exactly half of it. Every
determined quantity in both was bit-identical. And `enjl` is the **negative
control**: it fails the screen, and it turned out to be a real branch flip
([ticket 72](issues/72-enjl-branch-selection.md)).

So the screen has now been run forward, on data nobody had looked at, and it
was right in both directions.

### What is still to decide

The map recorded the unsharp part as "the form", and that is still the
question:

1. **Single-point identity or two-run differential?** `mu_i` equalling its own
   projection through `B_i, C_i, S_i` is a per-point invariant and cheap. What
   was actually OBSERVED is a differential: two runs of the same case shifting
   in the ratio of a charge. These are different tests and only the second
   catches what ticket 56 and ticket 62 caught. Possibly both.
2. **Where does the differential get its second run?** It needs two states to
   compare. Candidates: the two solver backends (§9's reference/fast pair,
   which `verify/` already compares), a perturbed initial guess, or the stored
   `.npz` — the last being circular, since the point is to judge a baseline.
3. **What does it do when it fires?** A shift proportional to a charge is not a
   failure — it is an undetermined potential, which is legitimate. The useful
   output is a CLASSIFICATION, so a red `test_baseline` can be read without a
   session of hand analysis. Whether that belongs in `verify/` (which asserts)
   or as a tool the baseline test calls on failure to annotate its own message
   is a real design fork.
4. **Does it generalise past `mu_C`, `mu_S`?** The algebra is the same for
   `mu_B`, but `mu_B` is undetermined only where n_B is itself unconstrained,
   which no mode does. Worth stating the scope rather than implying it.

### Why it is worth building

Every instance so far cost a session of hand analysis to classify, and each was
found only because something else went red. Ticket 62 needed it as a gate on
thirteen golden reference files and had to write the comparison from scratch to
get it. The next stack move, solver change or tolerance question needs the same
tool again.

And it is an argument for §2 that the map already noted and this ticket
confirms: the ratio is readable ONLY because species potentials are derived by
projection. A model carrying ad-hoc species potentials would show the identical
failure as unstructured drift across a dozen quantities, with nothing to read.

## Ruling

Agreed with the user: **both forms, and they are different tests with different
homes.**

**The per-point identity** — `mu_i` equalling its own projection through
`B_i, C_i, S_i` — is cheap and belongs in `eos/general/verify/`, the home
[ticket 64](64-general-verify-suite-missing.md) built. It catches a wrong basis
map on every run.

**But it cannot catch what tickets 56 and 62 caught.** An UNDETERMINED potential
satisfies the identity at every point and still moves between runs. The
differential needs two states to compare, so it belongs with the baseline
comparison, where a second run exists by construction.

Recording only the identity would claim coverage the screen's actual track
record does not support — and that record is now strong: run forward over
**53,763 baseline keys**, it separated `ccdm` and `njl`'s CFL undetermined
potentials (with `mu_8` carrying exactly half, twice, in two independently
written models) from `enjl`'s real branch flip, which it correctly failed.
Right in both directions, on data nobody had looked at.

The remaining sub-question — where the differential gets its second run (the
two solver backends, two stacks, or two seeds) — is execution detail for the
suite that implements it, not a decision this ticket owes.

Open for execution.

## Amended by ticket 72

[Ticket 72](72-enjl-branch-selection.md) corrects the table above and adds a
requirement.

**The `enjl` row is not a negative control.** Its two entries are one finding.
`mu_S` at Y_S = 0 is undetermined for exactly the structural reason
`mu_3 = mu_C` is undetermined in the CFL pattern — a conserved charge no
populated species carries — and that undetermined potential is what CAUSED the
O(1) branch flip recorded beside it as "physics". Carried as an unknown with no
row, it is a null column in the Jacobian; the least-squares termination fires
early on the rank-deficient problem and leaves the residual of the whole solve
three decades high (1.7e-11 against the model's usual 2.7e-15), close enough to
`RESIDUAL_TOL = 1e-10` for round-off to decide which side a point lands on —
and `enjl.solver.solve` answers a missed gate with a root on the other chiral
branch. So the screen fired on `enjl`, and ticket 62 read its output as noise.

The fingerprint therefore has **three** witnesses, not two, and the third is
the one that did real damage.

**The added requirement: an undetermined potential is a CONDITIONING hazard,
not only a reporting one.** A differential that only reads ratios between two
runs would have classified `enjl` correctly and still missed that the mode was
about to select a chiral branch by round-off. So the check wants a
single-point, single-run limb as well:

- a mode carries an unknown whose residual row is identically zero;
- equivalently, a Jacobian column below the numerical rank threshold;
- and the observable consequence is the whole solve's residual sitting decades
  above what the model's other modes reach.

`eos.enjl.verify.check_residual_margin` is the narrow, per-model form of that
last item — every mode must clear `RESIDUAL_TOL` by two decades rather than
merely pass it — and is the worked example this ticket's general version can be
written against. Note it is a *symptom* test: it fires on the consequence, not
on the null column. Whether the general check should look for the column
directly is this ticket's to decide.

One more site for the survey. `test/baseline/generate_baseline.py`'s `row()`
already documents the `mu_S`-at-Y_S = 0 case, and handles it by EXCLUDING
`mu_S` and the S-carrying species potentials from the recorded baseline. That
hides the symptom at the recording layer and leaves the ill-conditioning in the
solver. It is still the right protection for the models that have not had
ticket 72's fix (`did` at Y_S = 0 is named there), so it was not changed — but
whether the general answer is "do not record it" or "do not leave it
undetermined" is exactly this ticket's question, and `row()` is where the
repository already answered it once, the other way.

## Resolution

**Landed as `29afea0`, 0 added failures.** Three limbs, not two: the ruling's
identity and differential, plus the single-run conditioning limb the ticket-72
amendment added.

    eos.general.basis.projection_residual     one state, no second run
    eos.general.basis.undetermined_potential  two runs, the exact charge ratio
    eos.general.solve.undetermined_unknowns   one run, the null column itself

The library half is in `eos/general/`; the differential's application to the
baseline is `classify_drift` in the gitignored `test/baseline/test_baseline.py`,
beside the comparison it annotates. Splitting it that way is deliberate: the
TOOL is what ticket 62 had to write from scratch, so it lives in the published
tree, while the test that calls it lives where §11 puts tests.

### Where the differential gets its second run: the stored baseline

Chosen on availability, and the alternatives were weighed rather than
dismissed:

- **the stored `.npz`** — the only second run that exists for every model and
  every key without solving anything twice, and it is already being read at
  exactly the moment the question is asked.
- **two backends** (§9's reference/fast pair) — exists for two models. Partial
  coverage of a screen whose whole point is to be applicable everywhere.
- **two stacks** — what ticket 62 used, and it needs a second interpreter, so
  no in-process test can be it.
- **two seeds** — answers a genuinely different question (is the solve
  seed-stable) and costs a re-run of every grid.

The ticket's circularity worry does not bite, because the tool never judges
whether a baseline is RIGHT. It runs only on an already-red comparison and
says which of two readings the red is.

### Two readings, because one model shape has no quantum number to divide by

The first version covered only `mu_i` / `mu_eff_i` and was therefore **blind to
`ccdm` and `njl`** — the two models the screen's whole forward record comes
from, which carry `mu_3`, `mu_8`, `mu_C` flat per point rather than a species
dict. Caught by firing it at a real red `test_baseline[ccdm]` and getting
silence. Both readings now ship:

1. **Species family, exact charge ratio.** Xi moved twice what Lambda moved
   because S_Xi = 2 S_Lambda, and no nucleon moved: one undetermined mu_S,
   DERIVED. Ticket 56's `did` case.
2. **Only potentials moved, everything else at the point bit-identical.** No
   quantum number to divide by, so the ratios are printed as evidence rather
   than tested — the weaker reading, labelled as such. Ticket 62's `ccdm` and
   `njl` case.

Anything else moving beside the potentials is reported as a physics change,
which is what `enjl`'s branch flip was. A species family that moved in NO
charge's proportion is the strong negative and takes that reading, not the
weak one.

### Sub-question 3: it annotates, it does not assert

The classification is added to `test_baseline`'s own failure message, on an
already-red comparison. It does not gate: a shift proportional to a charge is
legitimate, so asserting on it would fail a green tree, and asserting its
ABSENCE would re-bless whatever moved. `verify/` asserts on the screens
themselves — that each fails on a broken input.

### Sub-question 4: the scope, stated

The algebra is identical for `mu_B`, and `undetermined_potential` accepts it,
but `mu_B` is undetermined only where n_B is unconstrained and no mode of §3
leaves it so. The classifier therefore reads S then C, and says so.

### Every limb was proved to fail

Ticket 63's lesson, applied to each. Neutering
`undetermined_potential`'s zero-coefficient guard turns the suite red with
"a neutron that moved under a mu_S drift passed"; neutering
`undetermined_unknowns` turns it red with "a mu_S column of exact zeros was
not caught". Fired on real data as well as synthetic: an exact S_i drift
injected into `sfho`, `did` and `dd2` baselines is named as one undetermined
mu_S of the right size, the `ccdm`/`njl` CFL pattern reproduces ticket 62's
`mu_8=+0.500` fingerprint, and a mass moved beside them flips both to the
physics reading.

### Found and NOT fixed

**`did`'s `fixed_YC_YS` at T = 0 carries `mu_S` as a null column** — measured
with `undetermined_unknowns` on the model's own finite-difference Jacobian, at
Y_S = 0 AND at Y_S = 0.05; at T = 20 MeV the Fermi tails populate the strange
species and the column is live. That is exactly the defect
[ticket 72](72-enjl-branch-selection.md) fixed in `enjl`, in the model this
ticket's amendment names as still carrying it, now measured rather than
inferred. Fixing it would move `did`'s numbers and belongs with that ticket.

The amendment's last question — whether the general check should look for the
column directly or fire on the symptom — is answered **directly**.
`eos.enjl.verify.check_residual_margin` remains the per-model symptom test and
is worth keeping: the column test needs a Jacobian, and the margin test needs
only a solve.

### Gate

python.org **3.14.2** (`python3`), **1407** collected over `test/baseline
test/general test/{dd2,sfho,zl,did,vmit,alphabag,njl,ccdm,enjl,mixed}
test/test_imports.py test/test_nonconvergence_return.py
test/test_parameter_routes.py`, isolated **pair** from `git archive 77f8962`
plus a snapshot of the gitignored `test/`:

    control (77f8962)      1397 passed, 1 failed, 5 skipped   17:53
    mine    (+ ticket 75)  1402 passed, 0 failed, 5 skipped   15:26

**0 added failures**; the +5 is four new `classify_drift` tests plus the
control's one failure passing. That failure is `test_baseline[ccdm]` and is
**intermittent**: it moved only `field_residual`, `gap_residual`, `n_3` and
`n_8` — the round-off keys the module docstring warns recompiled Numba kernels
shift — and it passes twice in a row in isolation on the control tree. All
**eleven** `verify/` suites PASS.
