# The S_i / C_i undetermined-potential check, now that general/verify/ exists

Type: task
Status: open
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
