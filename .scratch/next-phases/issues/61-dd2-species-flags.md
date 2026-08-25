# dd2 cannot take two of §4's six species flags

Type: task
Status: open
Blocked by: 44
Parent: ../map.md

## Question

Found by [ticket 04](04-notebook-skeleton.md) while building the shared knobs
cell, and covered by **no row** of ticket 08's audit or
[ticket 11](11-conformance-triage.md)'s triage — §3-ii covers
`thermal_neutrinos` + trapped *behaviour*, not the missing *names*.

`SpeciesFlags(**{six §4 flags})` raises **`TypeError` on dd2** and succeeds on
the other nine models. dd2 is missing `thermal_mesons` and `thermal_neutrinos`:

    dd2  extra = neutrinos, phi_field, sigma_star,
                 include_pseudoscalars, include_thermal_vectors

Two different problems wearing one finding:

1. **`thermal_mesons` is present but split** into `include_pseudoscalars` and
   `include_thermal_vectors`. §4 names one flag — "pi, K (and optionally the
   vector nonet)" — so the split is arguably a refinement, but §4's name does
   not exist and §13's "the same job carries the same name in every model" is
   the rule that binds. Decide: one `thermal_mesons` flag with the vector nonet
   behind a second, clearly-secondary name, or `thermal_mesons` as the alias the
   other nine models spell.
2. **`thermal_neutrinos` is genuinely absent.** dd2's `neutrinos` is NOT a
   misspelling of it — the field's own comment reads "only trapped / fixed-Y_Le
   modes", i.e. the matter-composition neutrinos. So the tau gas is unwired.
   §4 is explicit that setting a flag a model does not implement **raises**, and
   that a NotImplementedError is never a silent no-op — so a raise is an
   acceptable resolution here and wiring the sector is not required. What is not
   acceptable is the flag not existing, because then the caller gets a
   `TypeError` that reads as their own bug.

Also worth measuring while in the file: the **defaults** diverge across models —
`photons` True in six and False in four, `muons` True in four and False in six,
`alphabag` shipping `thermal_neutrinos=True`. §4's "no sector is enabled or
disabled implicitly" is only honoured if a caller passes all six every time.
Whether the defaults should be unified is a separate ruling; report, do not
change, unless the answer is obvious.

**Tickets [12](12-hadronic-skeleton.md), [15](15-quark-notebook.md),
[18](18-enjl-notebook.md) and [58](58-hybrid-skeleton.md) are blocked by this.**
The alternative — a per-model translation table in the knobs cell — would be
copied into four notebooks.

Blocked by [44](44-rename-dd2.md) so it does not collide with the dd2 renames in
the same files, and run tickets 42/43's AST check here: both proved that a
rename onto a name this repo already uses for a local adapter fails silently.

Resolved when `SpeciesFlags(**six)` constructs for all ten models, or raises
with a §4-compliant message, and `test/baseline/` is unmoved.

**Reporting.** Name the interpreter **in the filename** of whatever lands in
`output/_audit/` — `_py39` or `_py314` — not only in the prose. Per
[ticket 47](47-dd2-nmp-inversion.md) the stack is the whole difference between
0 and 14 failures, and the directory already holds files from both with nothing
in the name to tell them apart, so a listing alone can hand the next session the
wrong before-image. Report the **collected count** alongside the failure count
too: ticket 20's +2 tests in `test/test_imports.py` moved it to 1665.

**Neighbour, not duplicate:** [ticket 60](60-dd2-hyperon-flag-raise.md) is a bare
`KeyError` when `hyperons=True` meets a nucleonic parameter set — the same file
family, a different defect (a refusal with the wrong exception, against a flag
that does not exist at all). Whoever runs second should read the other first.

**Two documentation sites change when this lands**, both written by
[ticket 20](20-phase5-api-readme.md) and both stating the gap as it is TODAY,
so neither is wrong now and both go stale the moment dd2's flags move:

- `README.md`, the "Species flags" section — the paragraph beginning "One
  model does not yet carry two of the six", which names
  `include_pseudoscalars`, `include_thermal_vectors`, and `neutrinos` as the
  matter-composition field rather than the tau gas.
- `eos/__init__.py`, the `#:` comment above `SPECIES_FLAGS` (commit
  `7c5b7a9`), which says the same thing in the same words and ends "Passing
  the six shared names to dd2 raises TypeError, which reads as the caller's
  bug."

The second is deliberately a commit of its own so it can be reverted or
rewritten without touching the lazy-import work in `fe68f20`. If this ticket
resolves by giving dd2 the two names, both paragraphs are deleted rather than
edited; if it resolves by a raise that names the gap (which §4 permits for
`thermal_neutrinos`, since the tau gas is unwired rather than misnamed), both
are rewritten to say THAT, and `README.md`'s claim that the six names are
carried by every model becomes true without qualification.

Also: `test/test_imports.py::test_the_top_level_carries_the_mode_and_species_vocabulary`
checks §4's six names against seven models' dataclass fields and **exempts
`dd2` by name**, with this ticket cited in its docstring. Drop the exemption
when this lands — that is the check that keeps the answer from drifting back.
(`test/` is gitignored, so it is in a working copy only; see the map's
untracked-guards bullet.)


## Verified, and the exemption is not dd2-specific

Checked rather than taken on relay. `test/test_imports.py:252` reads

    for model in ("sfho", "zl", "did", "vmit", "alphabag", "njl", "ccdm"):

— an **inclusion list of seven**, not an exemption of one. Three models are
absent: `dd2` (which genuinely fails), and **`enjl` and `abpr`, both of which
PASS the six-flag check today** and are being skipped for nothing. Measured with
`dataclasses.fields`: neither is missing any of §4's six.

So dropping dd2's exemption when this closes is necessary and not sufficient.
**The list must be inverted**: iterate every model and exempt by explicitly named
exception, because as written a model added tomorrow is exempt by default and
the check meant to prevent drift is what permits it. That is the same defect the
docstring's `dd2` sentence describes, one level up.

Three prose sites carry the claim and all three change when this lands
(eos-88, `bcb9d56`): `README.md`'s species-flags paragraph, `eos/__init__.py`'s
`SPECIES_FLAGS` `#:` comment (`7c5b7a9`), and this test's docstring. The edit
differs by resolution — giving dd2 the two names DELETES those paragraphs, a
raise that names the gap REWRITES them.

**The exemption is a TWO-WAY gate, and this paragraph is the durable copy of
that fact.** The check exempting `dd2` does not merely skip it: it asserts
that dd2 IS still missing the names, so the moment this ticket gives dd2 the
two flags the test goes red rather than green, and its message names what has
to move with it. Closing this ticket therefore requires **three edits in one
change** — delete the `dd2` entry from the exemption dict, delete `README.md`'s
"One model does not yet carry two of the six" paragraph, and delete the
corresponding `#:` block above `SPECIES_FLAGS` in `eos/__init__.py`. Do them
together or the suite stays red, which is the point.

Recorded here rather than left to the failing assertion because **`test/` is
gitignored**: a fresh clone, a `git worktree`, or another machine has no such
test, so the assertion cannot reach whoever works this ticket there. The
ticket body is the only artifact that survives the checkout. (Where `test/`
does exist, the enforcement is real and will catch a partial close.)
