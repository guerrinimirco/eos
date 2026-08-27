# Should `alphabag.gluons` default False like the six?

Type: grilling
Status: resolved
Blocked by: -
Parent: ../map.md

## Question

Raised by [ticket 65](65-species-flag-defaults.md) and deliberately left to a
ruling, because `gluons` is **not** one of §4's six names and that ticket was
scoped to the six.

After 65, `eos/alphabag/species.py` reads:

    photons: bool = False
    gluons: bool = True
    thermal_neutrinos: bool = False

All three are mu = 0 thermal boson gases that carry no conserved charge and
contribute to eps, P and s alone, all three vanish at T = 0 — and one of them
is now on by default while its two neighbours are off. `SpeciesFlags()` in
`alphabag` therefore gives a thermal gluon gas and no photons.

**The case for False.** §4's rule is about sectors, not about which list a
flag's name appears on: "no sector is enabled or disabled implicitly ... if a
sector is off, its flag is False". A default of True switches the sector on
for every caller who did not name it, which is the same defect ticket 65
closed for the six.

**The case for True.** `gluons` is alphabag's own physics, not shared
vocabulary, and §4's six are the only names the repository promises behave
identically across models. A bag model's gluon gas is arguably part of what
the model *is* — closer to `enjl`'s fixed `hyperons=True` than to a
convenience — in which case the honest form may be `enjl`'s: fix it and raise
on any move, rather than leave it a silent default.

**What it would cost.** `alphabag` moved 0 baseline keys under ticket 65
because the generator calls its raw solvers (see
[ticket 81](81-second-default-solver-kwargs.md)), so this is measurable
cheaply through `eos_point` / `eos_table` before anything is regenerated.

`njl` and `ccdm`'s `csc`, and `dd2`/`sfho`/`did`'s `phi_field`, are the same
class of question and should be answered by the same ruling: `csc: bool =
False` is already off, and `phi_field=False` **raises** in `sfho` and `did`,
which is the fixed-by-the-model shape rather than a default.

## Ruling

Agreed with the user, after a grilling that ran the tree to an empty frontier:
**yes, and the rule is bigger than the flag.**

**The rule.** §4's "if a sector is off, its flag is False" binds a model's OWN
flags too, and the test is not which list a name appears on:

> **A flag with two legal values is a DEFAULT and is False, whatever its name.
> A flag with only one legal value RAISES on the other and is a STATEMENT
> about the model. There is no third category.**

What ruled out the alternatives:

- **Which of ticket 65's two justifications was load-bearing.** 65 argued both
  "`SpeciesFlags()` means one thing everywhere" and "no call inherits a sector
  it did not name". Only the second reaches an own flag, since no other model
  has `gluons` for it to agree with. 65's own deciding sentence was "the only
  rule that cannot silently ADD physics" — the second. So it decides here.
- **`gluons` cannot take `enjl`'s fixed-and-raise shape.** Both values are
  legal physics: a bag model with no thermal gluon gas is the standard MIT
  configuration, and `alphabag/verify/run_full_check.py:131,341` itself calls
  `include_gluons=False` to subtract the sector before checking Euler. A raise
  would be a lie. That left True-or-False and nothing else.
- **The third category was where both defects lived.** `alphabag.gluons` and
  `dd2.phi_field` were the repository's only two flags that defaulted True and
  quietly accepted False. Naming that category and abolishing it is what makes
  the rule testable rather than a case-by-case judgement.

**`dd2.phi_field` flips rather than starting to raise.** It was the closer
call. Against flipping: DD2Y is fitted by Fortin (2017) Table 1 WITH the SU(6)
phi coupling, so a hyperonic run without the field leaves a fitted coupling
unused — which is exactly the claim `sfho` and `did` make in their raise
messages. For flipping: `dd2` also ships `from_hyperon_potentials()`, a
documented non-DD2Y route, so hyperons-without-SU(6)-phi is not categorically
wrong here the way it is in SFHo; and a CONDITIONAL raise (legal nucleonically,
refused with hyperons) would be a shape no other model in the repository has.
A raise can be added later on evidence; it should not be added for tidiness.

**Out of scope, filed rather than folded in:**

- the CFL gluon term -> [ticket 92](92-cfl-gluon-term.md).
- the `include_gluons` solver kwarg -> already answered by
  [ticket 81](81-second-default-solver-kwargs.md) §2, which deletes it into the
  flags; measurement and a correction to its premise recorded on
  [ticket 90](90-solver-signature-and-units-sweep.md).
- the two `notebooks/hadronic_eos` sites -> below.

## Resolution

**Two flags flipped. `alphabag.gluons: True -> False`, `dd2.phi_field:
True -> False`.** Nothing else: an audit of every own flag in all ten models
against the rule found exactly these two violations.

| flag | before | after | why |
|---|---|---|---|
| `alphabag.gluons` | `True` | **`False`** | two legal values -> a default |
| `dd2.phi_field` | `True` | **`False`** | two legal values -> a default |
| `dd2.neutrinos`, `dd2.thermal_vectors`, `njl.csc`, `ccdm.csc` | `False` | unchanged | already conform |
| `dd2.sigma_star` | `False`, raises on True | unchanged | statement |
| `sfho.phi_field`, `did.phi_field` | `True`, raises on False | unchanged | statement — those models always solve the field |
| `abpr.gluons` | `False`, raises on True | unchanged | statement |

### The blast radius, measured before the first edit

**Zero baseline keys. Zero golden references. Zero test failures.** Each
established from the source rather than assumed:

- **`dd2.phi_field` is structurally inert without hyperons.** All four
  consumers read the conjunction — `table.py:209`, `solver.py:411`,
  `solver.py:1019`, `thermodynamics.py:359` all compute
  `flags.phi_field and flags.hyperons`. No `SpeciesFlags()` call and no
  nucleonic call can see the flip.
- **`test/baseline/` does not move.** `case_dd2` names `phi_field=True`
  explicitly on both hyperonic rows (`generate_baseline.py:294-295`).
  `case_did` and `case_sfho` inherit it, but those two models RAISE on
  `False`, so no ruling can move their default. `case_mixed` is
  `hyperons=False` at T = 0. `case_alphabag` calls the raw solvers and never
  constructs a `SpeciesFlags`, so the `gluons` flag never reaches it — ticket
  65's coverage gap, still open, and now recorded on ticket 90.
- **§12's golden references are all nucleonic** — the DD2 SNM point at
  n_B = 0.16 and the published NMP/TOV values. `phi_field` cannot reach them.
- **Three call sites that looked like they inherited, do not.**
  `test/tov/test_solver_fast_robustness.py:89,191` name `phi_field=True` on
  the wrapped line; `test/mixed/test_species_flags.py:51` maps only §4's six,
  so `phi_field` never crosses into the mixture; `test_alphabag_api.py:112`
  passes a bare `SpeciesFlags()` but asserts an unknown-vector LENGTH, which
  no thermal boson gas can move.

### The gate: every delta is exactly the sector, and nothing else moved

Measured through the PUBLIC surface, because `test/baseline/` cannot see this
change (above). alphaBag at n_B = 0.8, `beta_eq_neutrinoless`, and the paired
branch at Delta0 = 100:

| case | before P | after P | delta P | -P_gluon(T) |
|---|---|---|---|---|
| `beta.T0` | 147.2605542143 | 147.2605542143 | 0 | 0 |
| `beta.T10` | 148.2672443915 | 148.2657785530 | -1.465838e-03 | -1.465838e-03 |
| `beta.T30` | 156.2984560478 | 156.1797231315 | -1.187329e-01 | -1.187329e-01 |
| `cfl.T0` | 169.6188429205 | 169.6188429205 | 0 | 0 |
| `cfl.T30` | 173.0304201729 | 172.9116872566 | -1.187329e-01 | -1.187329e-01 |

Every delta equals minus the gluon pressure to machine precision, and T = 0
does not move at all. dd2 at n_B = 0.6, T = 0, DD2Y:

| case | before P | after P | |
|---|---|---|---|
| `SpeciesFlags()` (nucleons) | 219.9261345872 | 219.9261345872 | identical |
| `hyperons=True, muons=True` | 135.4308408121 | **105.3535680598** | moved onto the phi-off value |
| `... phi_field=True` | 135.4308408121 | 135.4308408121 | identical |
| `... phi_field=False` | 105.3535680598 | 105.3535680598 | identical |

**The one number worth naming: 135.43 -> 105.36 MeV/fm^3, a 22% drop.** That
is what a hyperonic call which named only `hyperons=` was inheriting. The
ruling was taken when this was described as "the 148 `hyperons=`-only calls"
without its size; the size is large, and the change is therefore worth the
explicit `phi_field=True` it now demands rather than in spite of it. Nothing
in the repository was making that call — see below — but the magnitude is the
argument for the rule, not against it.

### The suite gate: an isolated control/change PAIR, because the tree was not mine

**The first full-suite run had to be thrown away.** It was compiling against
`eos/dd2/thermodynamics.py` while a concurrent session was mid-edit on
[ticket 67](67-dd2-t0-adoption.md) — deleting dd2's local T = 0 closed forms
into `general.fermi_integrals`. A suite result that mixes two tickets cannot
attribute a failure to either, so it was killed rather than reported.

Gated instead in a PAIR of isolated copies built from clean `HEAD`
(`git archive`), one of them carrying ONLY this ticket's eight files. Both
copies got the same `test/` — including the WIDENED drift check — so the
control is expected to fail exactly that one node and nothing else, which
makes it a stronger control than a matched pair would have been.

| | control (clean HEAD) | + ticket 82 |
|---|---|---|
| `test_every_species_flag_defaults_off_or_raises` | **FAILED** (by design) | passes |
| `test_thermal_meson_feedback[neutral]`, `[fixed]` | passed | **FAILED** -> fixed, below |
| `test_baseline[ccdm]` | FAILED | FAILED — neither arm's, see below |
| | 2 failed, 1732 passed | 3 failed, 1731 passed |

**Final, after the finding below, change arm run ALONE:**

    1734 passed, 20 skipped, 48 warnings in 1493.24s (24:53)

**Zero failures.** CPython 3.14.2 / numpy 2.3.5 / scipy 1.17.0.

    control  output/_audit/pytest_ticket82_control_2failed_1732passed_py314.txt
    change   output/_audit/pytest_ticket82_change_0failed_1734passed_py314.txt

**`test_baseline[ccdm]` was the apparatus lying, not a failure.** It went red
in BOTH arms of the pair, which already made it unattributable to this change;
run in isolation on the CONTROL it passes (`1 passed in 25.41s`), and the solo
change-arm re-run passes it too. The two suites had been running concurrently.
Recorded because a red `test_baseline` is exactly the kind of thing that gets
copied forward as a known failure when it is nothing of the sort.

### The one real failure this change caused, and what it was

`test/dd2/test_thermal_meson_feedback.py::test_analytic_jacobian_matches_finite_difference`,
both parametrisations. **Not a solver defect — ticket 65's finding #1 again, in
a file it did not reach.**

The module's helper read

    def _flags(ps=False, tv=False, hyperons=True):
        return SpeciesFlags(hyperons=hyperons, muons=True,
                            thermal_mesons=ps, thermal_vectors=tv)

— `hyperons=True` named, `phi_field` INHERITED. The residual carries a phi row
only when `phi_field and hyperons`, and the Jacobian test hand-builds its
unknown vector with `phi0` in it at line 153. So the vector's LENGTH was being
decided by a default the helper never mentioned: with `phi_field` off the
residual drops 6 rows to 5 and the test raises
`IndexError: index 5 is out of bounds for axis 1 with size 5`.

Fixed by naming the flag — `phi_field=True` in `_flags`, with a comment saying
why — which is the test's own intent restored, not a tolerance loosened. Proof
that it is behaviour-preserving rather than a patch over the change: the file
is **11 passed in both arms**, control and change alike, because on the control
the explicit value equals the inherited one.

### The drift check, and the exemption it retired

`test_the_six_species_flags_all_default_to_off` is now
**`test_every_species_flag_defaults_off_or_raises`** (node id changed): it
iterates every field of every model's `SpeciesFlags`, not only §4's six, and
allows a `True` default only if constructing the flag with the opposite value
raises `NotImplementedError`.

**It has no exemption list.** 65's `exempt` dict was self-invalidating by
design, and it duly invalidated itself the moment the check was widened: under
the two-category rule `enjl` is not an exemption at all, it is simply the
STATEMENT category — it fixes every flag and raises on any move. The carve-out
existed only because the narrower check could not express "raises on the other
value". One rule, no carve-outs, less code than before.

Verified non-vacuous in both directions: restoring `alphabag.gluons = True`
turns it red, restoring `dd2.phi_field = True` turns it red, and the shipped
state is green. `test/test_imports.py`: 200 passed.

### Prose

- **`CLAUDE.md` §4** gains the rule itself, in the paragraph that already says
  "if a sector is off, its flag is False" — a model's own flags follow it, the
  two categories are named, and the third is explicitly abolished.
  [Ticket 85](85-claudemd-sentences-owed.md) is told, not asked: it is blocked
  behind 29, 70 and 84, and this is a self-contained insertion.
- **`README.md`** — the "all six default to False" paragraph gains the own-flag
  rule and the worked consequence, `SpeciesFlags(hyperons=True, muons=True,
  phi_field=True)` for a hyperonic DD2Y run.
- **`eos/__init__.py`** — the `#:` block above `SPECIES_FLAGS`, same content.
- **`eos/alphabag/species.py`**, **`alphabag.md`**, **`alphabag.tex`** — the
  `gluons` paragraph says it defaults False and why, and the flag's field
  docstring now records that the paired phase's gluons are Meissner-massive
  (ticket 92).
- **`eos/dd2/species.py`** — the module docstring states the rule and the
  `phi_field and hyperons` conjunction that makes the flag inert nucleonically.
- **`eos/alphabag/api.py:83`** — a stale docstring, "the default has photons,
  gluons and the untracked thermal neutrino flavours on", two thirds falsified
  by ticket 65 and missed by it. Corrected.

### Found in passing: ticket 65 left a wrong number in README

**README example 1's captured output was already wrong on `main`.** It shows
`True 32.27157556177297` for `SpeciesFlags(hyperons=True)` at n_B = 0.32,
T = 10. That value is reproduced only by `hyperons=True, muons=True,
photons=True, phi_field=True` — the PRE-65 defaults. Ticket 65 flipped `muons`
and `photons` in dd2 and its resolution claimed "every captured output
reproduces bit-identically"; that held for four of the five examples and this
one escaped, because 65 made the other four explicit and left this one
inheriting.

Fixed here rather than filed: the flags are spelled `hyperons=True,
phi_field=True` and the output recaptured as `True 32.431587607228806`,
verified by running the snippet. Example 4 was already explicit and its
`P = 135.431 MeV/fm^3` reproduces bit-identically with `phi_field=True` added.

### Notebooks

Ruled: preserve where preserving is also correct physics, let the rest take
the new default.

- **`notebooks/quark_eos.py` needs no edit.** The premise it was ruled on
  turned out to be wrong in the model's favour: both CFL table builds are at
  **T = 0** (`quark_eos.py:745`, `:1136`), where the gluon gas is identically
  zero, so no shipped figure moves. The own-flags audit cell prints defaults
  live and self-updates to `gluons: False`. What does move is printed table
  cells at `PROBE_T = 10` and the hot point, by exactly `P_gluon(T)` —
  **`quark_eos.ipynb`'s captured outputs are stale there and want a re-run**,
  which is a notebook execution rather than an edit and is not done here.
- **`notebooks/hadronic_eos.py:456,894`** inherit `phi_field` with
  `hyperons=True` and want an explicit `phi_field=True` to hold their figures.
  **Deliberately not touched**: both `hadronic_eos.py` and `hadronic_eos.ipynb`
  were modified in the working tree by a concurrent session throughout this
  ticket, and a paired jupytext `.py`/`.ipynb` mid-edit by another writer is
  where that goes wrong. Owed as a follow-up.

### The working tree, and a concurrent session

Ticket 81 was **resolved by another session while this ticket was being
worked**, spawning tickets 89, 90 and 91, and two further tickets were filed
under the number 88. This ticket's own new ticket was therefore renumbered
**88 -> 92**. Nothing here was written without an explicit pathspec, and the
four files the other session holds — `notebooks/enjl_eos.{py,ipynb}`,
`notebooks/hadronic_eos.{py,ipynb}` — were not touched.

### Landed

Committed on `main` with explicit pathspecs: `eos/alphabag/{species,api}.py`,
`alphabag.{md,tex}`, `eos/dd2/species.py`, `eos/__init__.py`, `CLAUDE.md`,
`README.md`, and the notes to tickets 81 and 90.

Gate, python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0, on `a4fdfbd`:
**1734 passed, 20 skipped, 0 failed** (20:58). Against
`output/_audit/pytest_after_ticket74_py314.txt` (1 failed, 1680 passed,
15 skipped) that is **0 added failures** — one FEWER, the `enjl` node closed
by `f8ccc33`. Both README examples were re-run against the shipped text and
reproduce exactly: example 1 `True 32.431587607228806`, example 4
`P = 135.431 MeV/fm^3`.

**Two things did not land as this ticket left them, both recorded rather than
quietly fixed.**

1. **This ticket's map entry is in someone else's commit.** A concurrent
   session resolved [ticket 89](89-dd2-honours-species-flags.md) mid-landing
   and staged the whole of `map.md`, sweeping this ticket's 35-line entry into
   `a4fdfbd` ("docs(scratch): resolve ticket 89"). The entry is correct and
   present; only its attribution is wrong. Not rewritten — nothing was pushed,
   but rewriting shared history with another session live in the checkout buys
   a cosmetic fix at a real risk.

2. **The `notebooks/hadronic_eos` sites are still owed**, and the reason has
   changed. This ticket deferred them because a concurrent session held the
   files; that session's edits are still uncommitted in the working tree and
   are NOT this ticket's — they carry no `phi_field` change at all, but a
   widened grid (12 -> 300 points), a thermal axis to 100 MeV, four species
   flags turned on and three comment blocks deleted. So `hadronic_eos.py:456`
   and `:894` still inherit `phi_field`, and the explicit `phi_field=True`
   those two figures need is unwritten. It cannot be added without also
   committing that unrelated edit, which is why it is still owed.
