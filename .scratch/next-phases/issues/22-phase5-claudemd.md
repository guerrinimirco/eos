# Phase 5 item 5 — apply the CLAUDE.md diff

Type: task
Status: resolved
Assignee: mirco (session)
Blocked by: 02, 09, 11
Parent: ../map.md

## Question

`docs/REFACTOR_PROMPTS.md` Phase 5 item 5: re-read `CLAUDE.md` against the repo as
it now is and correct anything the refactor settled differently. It was written
in Phase 0 as a target.

This ticket **applies** what three earlier tickets ruled — it does not decide
anything itself:

- ticket 02 — §11's "one usage notebook per model" line
- ticket 09 — §11's mandated `.tex`, and the matching line in the Acceptance
  criteria block of `docs/REFACTOR_PROMPTS.md` if the `.tex` goes
- ticket 11 — every (b)-class row from the conformance triage

[Ticket 08](08-conformance-table.md) produced 11 (b)-class rows for this diff.
The largest: **`ccdm` appears nowhere in CLAUDE.md**, and `njl` only ever inside
the string "enjl" — §1's model list omits `did`/`njl`/`ccdm`, §11's omits
`njl`/`ccdm`, and §5's adapter list omits the shipped `njl_phase` and
`ccdm_phase`. The `verify/` carve-out from the model-to-model import rule is real
(`test/test_imports.py:88-97` plus a DEFERRED entry) but unstated in §1.

One of the 11 lands outside CLAUDE.md: **the §10 acceptance criterion**
`grep -rn "rcParams" eos/ nucleation/` *"hits exactly one file"* now hits three,
but two of those are prose saying the file does **not** set rcParams. The rule
passes in substance — all ~30 assignments live in `general/figure_style.py` — so
it is the grep-as-gate that needs rewording, in the Acceptance criteria block of
`docs/REFACTOR_PROMPTS.md` rather than here.

One correction found independently by [ticket 06](06-document-audit.md) and owed
nothing to the tickets above: **`njl` and `ccdm` are absent from CLAUDE.md's own
§1 and §11 model lists**, though both ship as full models with documents, tests
and `verify/` suites. That is a plain omission in the specification, not a
settled-differently row, so it lands here regardless of how the triage rules.

Show the diff before removing anything. The (c)-class entries from ticket 11 land
in `docs/DEFERRED.md` under this ticket too, so the two documents move together.

## Carried in from ticket 11 — the twelve (b)-class rows

[Ticket 11](11-conformance-triage.md) ruled these to be document changes, not code
changes: CLAUDE.md describes a target the refactor settled differently. Each is
one edit to a named section; evidence and file:line in
[conformance-table.md](../research/conformance-table.md).

- **§1 gains the `verify/` carve-out** (finding 3). Three suites reach sideways —
  `abpr/verify:47-48` into `alphabag`, `enjl/verify:817,834-835` into `mixed`.
  The refactor settled this deliberately: `test/test_imports.py:88-97` documents
  it ("it checks END-TO-END invariants … doing that requires reaching sideways by
  construction. It is also not on the path a sampler imports, which is what the
  layering rule protects") and `docs/DEFERRED.md:155-157` confirms it. §1 does
  not contain the carve-out and should. Note `abpr`'s edge is unledgered even
  though `enjl`'s is; writing the carve-out into §1 covers both.
- **§1, §5 and §11's model lists are stale, and `ccdm` appears nowhere in the
  document** (finding 39) — `grep -n "ccdm" CLAUDE.md` returns nothing. §1
  line 22 omits `did`, `njl`, `ccdm`; §11 line 449 omits `njl`, `ccdm`; §5's
  adapter list omits the shipped `njl_phase` (`mixed/adapters.py:1051`) and
  `ccdm_phase` (`:1189`); §5's nuclear-sector list omits `did`, which has
  `eos/did/nmp.py`. Both `njl` and `ccdm` are complete models with `verify/`
  suites, `.tex`/`.md`, five test files each and shipped adapters. **The
  cheapest and highest-value document fix in the audit**, and upstream of
  `test_imports.py:76 MODEL_PACKAGES` inheriting the same stale list
  ([ticket 50](50-mechanical-fixes.md) item 1).
- **§2/§7 permit a model with a mixed species list to keep a local
  quantum-number table** (finding 14). `enjl/species.py:38,51,57` and
  `enjl/thermodynamics.py:473-476` write the numbers and the density sums
  longhand. All signs correct and `enjl/verify:67` cross-checks against
  `eos.general.basis`, which is the mitigation. ENJL is the only model with
  baryons *and* quarks in one species list. (The milder `sfho` half — the
  species-potential map written inline at `sfho/thermodynamics.py:171,481-482`
  and `sfho/backends/jacobian.py:197`, where `dd2` and `did` import
  `general.basis.species_potential` — was ruled **(a)** and rides
  [ticket 45](45-rename-sfho.md).)
- **§5 permits `mode` to carry a default where a model has exactly one mode**
  (finding 15b), which is `abpr`. The other three defaults are dropped by
  [ticket 54](54-signature-corrections.md).
- **§5 says a freeze target may appear as a named argument** (finding 16a).
  `dd2/api.py:142` takes `Y_p=None`, a species fraction and the only such name in
  a signature — it is what dd2's `composition` freeze holds. Renaming it to a §5
  condition name would be a lie; it is not a condition.
- **§8 names the P-monotonicity delivery gate as belonging to whoever builds a
  table** (finding 29). Today §8 scopes it to tables "DELIVERED to a structure
  solver" and does not say which suites owe the check, so eight models are
  correct by the letter while `dd2` and `did` build core tables and check
  nothing. [Ticket 51](51-verify-invariants.md) applies the (a) half.
- **§5 rules that `general/` earns a `verify/`** (finding 31) — it is the single
  home of the Fermi/Bose integrals (§7), the basis maps (§2) and the meson gas,
  the things every model's correctness rests on, and the JEL-vs-fallback parity
  gap is already ledgered as untested (`DEFERRED.md:48-62`). §5's `verify/` list
  is written for models; extend it. Building the suite is
  [ticket 21](21-phase5-structure.md)'s.
- **§1's `zlvmit` exemption is widened explicitly to documents and tests, or the
  gap is ledgered** (finding 34). §1 line 44 exempts it from "the uniform API"
  only; `eos/zlvmit` has no `.tex`, no `.md`, no `verify/` and zero `test_*.py`
  (`test/zlvmit/` holds 61 golden `.dat` files), while `test/baseline/` does carry
  a `zlvmit.npz`. The map already places `zlvmit` out of scope, so this is a
  document question, not a work item.
- **§5 scopes the `thermodynamics_<sector>.py` rule to models** (finding 36).
  `find eos -name "thermodynamics_*.py"` returns exactly one file,
  `eos/general/thermodynamics_leptons.py`, and §5 forbids "a package holding
  exactly one suffixed file". `general/` is not a model, and the suffix does not
  restate the package name. Either scope the rule or rename the file to
  `leptons.py`; ticket 11 ruled scope.
- **§10's acceptance criterion becomes `grep -rn "rcParams\s*\[" eos/` or an AST
  check** (finding 41). The plain grep hits three files, but two
  (`zlvmit/plot_results.py:184`, `zlvmit/table_reader.py:703`) are prose comments
  stating the file does *not* set rcParams. All ~30 real assignments are in
  `eos/general/figure_style.py`. The rule is satisfied; the published test of it
  cannot tell an assignment from a sentence about assignments.
- **§3 declares the fifth mode name `cfl`** (finding §3-i), used by
  `alphabag/solver.py:61` and `abpr/solver.py:48` and declared in neither §3 nor
  the ledger.
- **§4 states that `thermal_neutrinos` is meaningful with the trapped mode**
  (finding §3-ii). It raises in `sfho:576` and `did:213` and succeeds in
  `njl:275`, `ccdm:307`, `enjl:224-236`. §4's own wording settles it —
  `thermal_neutrinos` means "flavors **NOT tracked in the matter composition**",
  and under the trapped mode the e and mu families are tracked, so the flag means
  the tau family and the combination is meaningful. The three that succeed are
  right; [ticket 54](54-signature-corrections.md) drops the two raises.

Two more, already ruled elsewhere and listed so the diff is complete: §11's
"one usage notebook per model" is amended to the grouped notebooks
([ticket 02](02-notebook-grouping.md)), and §11's mandated `.tex` **stands
unchanged** ([ticket 09](09-tex-or-md.md)).

## Answer

**All twelve (b)-class rows applied, plus the two carried in from tickets 02 and
09. `CLAUDE.md` +97/−17 over 102 changed lines; `docs/REFACTOR_PROMPTS.md` +12.
No `eos/`, `test/` or `docs/DEFERRED.md` file was touched.**

### What landed where

| § | change |
|---|---|
| §1 | model list gains `did`, `njl`, `ccdm` — it had three of the ten |
| §1 | the **`verify/` carve-out** is written down, and scoped: a suite may import another model or the engine, nothing else in the package may, and **the astro half of the rule gets no such carve-out** |
| §1 | `zlvmit`'s exemption widened explicitly to §11's document and §12's tests, naming `test/baseline/zlvmit.npz` as the whole of what pins it |
| §2 | the narrow exception for a species list spanning baryons **and** quarks, with the `verify/` cross-check named as what buys it |
| §3 | **`cfl` declared as a fifth mode**, with why it is not available to every model — a locked phase *has* no free fraction, so it is a statement about which phase the model describes, not a choice of equilibrium condition |
| §4 | **`thermal_neutrinos` is meaningful with `beta_eq_neutrino_trapped`** and a model must not raise on it |
| §5 | `mode` is required, with the single-mode exception (`abpr`) |
| §5 | a **freeze target** may be a named argument of `eos_response`; `leptons` is neither condition nor freeze target and is always explicit |
| §5 | adapter list gains `njl_phase` and `ccdm_phase` |
| §5 | nuclear-sector list gains `did` and `zl` |
| §5 | the `thermodynamics_<sector>.py` rule scoped to **model** packages |
| §5 | **`general/` carries a `verify/`**, with what it checks |
| §8 | **the delivery gate belongs to whoever builds the table** |
| §11 | model list gains `njl`, `ccdm`; `notebooks/` reworded to grouped |
| PROMPTS | the §10 rcParams acceptance criterion |

### Three things that came out differently than the ticket anticipated

**1. The §3 `cfl` row could not be a one-line table entry.** Adding `cfl` beside
the four modes implies every model owes it, which is the opposite of true — §3
opens "Every model exposes the same modes". So the row carries a paragraph
saying why this one is not universal: it is not a choice of equilibrium
condition but a statement about which phase the model describes. That paragraph
then does double duty as the justification for §5's single-mode `mode` default,
so the two rows are cross-linked rather than independent.

**2. The §4 `thermal_neutrinos` ruling is derivable from §4's existing wording,
which is why it is stated as a clarification and not a change.** The flag is
defined by what it does NOT cover — "flavors NOT tracked in the matter
composition" — so under trapping, where the e and mu families are tracked, it
means the tau family. The document did not need a new rule, it needed the
consequence spelled out, because five models read the same sentence two ways.

**3. The rcParams criterion needed two attempts, and the first one was wrong in
the exact way the finding describes.** The audit proposed `grep -rn
"rcParams\s*\["`, but `figure_style.py` also sets style through
`rcParams.update(base)`, so that pattern misses 1 of the 22 real assignments.
Widening to `rcParams\s*[\[.]` then **re-broke the original bug**: it matches a
prose sentence ending in "rcParams." at `eos/zlvmit/plot_results.py:184`. The
shipped criterion is

    grep -rnE --include='*.py' 'rcParams\s*(\[|\.update)' eos/ nucleation/

— subscript or `.update(`, the two ways rcParams is actually set, with
`--include='*.py'` keeping prose out. **Verified from
`~/Desktop/Research/Python_codes` across both repositories: 22 hits, all in
`eos/eos/general/figure_style.py`, one file.** Without `--include` it also hits
this effort's own audit report, which quotes the pattern.

### The DEFERRED.md half did NOT move with this ticket

This ticket's Question says the (c)-class entries "land in `docs/DEFERRED.md`
under this ticket too, so the two documents move together". They did not, for a
reason found at the start of this session: **`docs/DEFERRED.md` is a file the
rename tickets edit** — [ticket 43](43-rename-vmit.md)'s resolution rewrote its
vmit entry, [ticket 10](10-rename-approvals.md) corrected another line of it, and
it was last written at 09:07 today by that work. A concurrent session
(`eos-48`) was live in this checkout applying [ticket 44](44-rename-dd2.md) while
this ticket ran. Two sessions writing one file loses writes regardless of
section, so the (c) entries stay at [ticket 55](55-deferred-ledger.md), to run
once the renames are committed. `CLAUDE.md` was taken instead precisely because
it is disjoint: neither 43 nor 44 mentions it, and it had not been written since
Aug 24.

### Suite

**Not run, deliberately.** `CLAUDE.md` and `docs/REFACTOR_PROMPTS.md` are
documentation and no test reads either — every hit of `grep -rln "CLAUDE.md"
test/` is a docstring or comment citing a section number, and nothing opens or
parses the file (`grep -rn "open(.*CLAUDE\|read_text().*CLAUDE\|Path(.*CLAUDE"
test/` returns nothing). So this diff **cannot** move a number or a test.

Running the suite anyway would have produced a misleading result rather than a
reassuring one: the working tree carries `eos-48`'s in-flight dd2/mixed rename,
so any failure count measured now belongs to that work, not to this ticket, and
recording it here would attribute it wrongly.
