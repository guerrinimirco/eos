# Phase 5 item 3 — write docs/STRUCTURE.md

Type: task
Status: resolved
Blocked by: 20, 14, 17, 19
Parent: ../map.md

## Question

`docs/REFACTOR_PROMPTS.md` Phase 5 item 3. `docs/STRUCTURE.md` does not exist
today, and `CLAUDE.md` §10 and §11 both reference it — §10 requires that a worked
figure example live there.

Aimed at a physicist who has never seen the repo: the module map; the mode and
species tables; the charge conventions (`Y_C` non-leptonic, `S = +1` per s
quark); the units (n in fm^-3, T and mu in MeV, eps and P in MeV/fm^3); the
reference/fast contract (§9); how to add a new model; and one worked end-to-end
example longer than the README ones. **Link each model to its document** — to the
`.tex` or the `.md` per ticket 09's ruling.

Two additions on top of the Phase 5 text:

- **Execute every code block and paste the real output.** Same standard as the
  README.
- **Link the three notebooks from Stages 1–3 as the worked examples** — which is
  why this is blocked on all three benchmark tickets.

Acceptance asks that a physicist find the function computing a given quantity in
under a minute from this document. Judge the draft against that.

## Carried in from ticket 11

[Ticket 11](11-conformance-triage.md) ruled finding 31: **`general/` earns a
`verify/` suite.** It is the single home of the Fermi/Bose integrals (§7), the
conserved-charge basis maps (§2) and the thermal meson gas — what every model's
correctness rests on — and the JEL-vs-pure-Python fallback parity gap is already
ledgered as untested (`docs/DEFERRED.md:48-62`, "no parity test currently pins
the two together"). Seven files in `test/general/` cover it, but `test/` is
gitignored, so a fresh clone can check none of it.

§5's `verify/` list is extended by [ticket 22](22-phase5-claudemd.md); building
the suite is this ticket's, alongside `docs/STRUCTURE.md`. The JEL parity check is
the obvious first entry — §7 makes JEL the validated implementation that is never
removed and requires every alternative to be validated against it.

## Resolution

**Shipped: [`docs/STRUCTURE.md`](../../../docs/STRUCTURE.md), 1132 lines, plus
`docs/figures/structure_dd2_vmit.png`.** Committed `f479845` with explicit
pathspecs. Nothing else was touched — the whole diff is one new document and
one new figure.

### The thirteen sections

Units table; the two conventions (§1.1 `Y_C` non-leptonic, §1.2 `S = +1` per s
quark, each with an executed block that shows it rather than asserting it); the
annotated module map and the layering rule with both named carve-outs; **§3, a
quantity -> module -> function index** in four tables (reading a solved state,
`general/`, inside a model, engine + astro); the mode table with the
raise-not-skip probe; the species-flag table with the same; the shape every
model has inside and the `thermodynamics.py` / `solver.py` boundary; the
reference/fast contract with both paths timed and compared; the `verify/`
suites; the document table; the notebook table; the worked example; the worked
figure; how to add a new model.

**§3 is the answer to Acceptance's own test.** "Where is `s` computed" is one
table lookup from the contents, not a grep: the TOC is at the top, §3 is the
third entry, and each row names the module and the function.

### Both non-negotiables met

**Every code block was executed.** Verification was mechanical, the same method
ticket 20 used on the README: a script extracts every ` ```python ` block and
`exec`s them **in one namespace, in order**, so §12's figure genuinely continues
from §11.3's `tab` the way the text says. **14 python blocks, all 14 executed**,
and the pasted output is that run's. Two shell blocks
(`python3 -m eos.{vmit,general}.verify.run_full_check`) were run separately and
pasted verbatim.

One block is NOT ` ```python `: the four-line `try/except ImportError` quoted
from `eos/dd2/solver.py:67` in §7. It is a source citation, not an example, so
it carries a plain fence and the text says "quoted from the file, not run here".
The first draft had it as `python` and the extractor caught it — which is the
point of running the extractor rather than eyeballing.

**The §10 figure example produces a figure through `figure_style.py` and
nothing else.** `paper_grid` + `panel_label` + `apply_style` + `save_figure` +
`STANDARD_COLORS`; no other module is imported for styling and no rcParams are
set by hand. Two panels: pressure coloured by phase segment, and `chi` running
0 -> 1 between the two located boundaries. `constraints.overlay` is *mentioned*
in the surrounding prose with a pointer to the README's example of it, not
re-demonstrated.

### Ticket 09's ruling, applied

Read before choosing: **keep both, and they carry the same information, each
written natively for its format.** So §9 links **both** files for all thirteen
documents — twelve models plus `tov` and `gmode` — and says explicitly that
neither is a pointer to the other and neither is a subset. Linking only one
would have re-created the "the closed forms are in the .tex" shape that ruling
retired. `zlvmit` is listed as carrying no document, which is correct: it is
legacy and exempt.

All four notebooks are linked in §10, `.ipynb` and `.py` each, with what each
one walks through.

### Checks run

- **All 48 relative links resolve** (script-checked against the filesystem).
- **All 13 TOC anchors match a heading**, under GitHub's slug rule.
- **The block outputs were re-verified against the tree AFTER a second session
  went live** and started modifying `eos/dd2/{api,solver,parameters,table}.py`
  among 30 other files. Every physics number is byte-identical across the two
  runs; the only movement is the millisecond timings in §7 and §13, which the
  document's header says up front are the only numbers that move run to run.
- **The README's pure-DD2 star reproduces on this stack** — `M_max = 2.419`,
  `R(M_max) = 11.99`, `R(1.4) = 13.19`, the py3.9 numbers to the digit — which
  is what §11.4's comparison against the hybrid rests on.
- Interpreter **python.org 3.14.2** (numpy 2.3.5, scipy 1.17.0, matplotlib
  3.10.9), never prefixed with `timeout`. **The full pytest suite was not run**,
  per the ticket; the diff is one new document and cannot move a test.

### The carried-in half was already done

Ticket 11's finding 31 — "`general/` earns a `verify/` suite" — was **resolved
by [ticket 64](64-general-verify-suite-missing.md)**, which shipped
`eos/general/verify/` with five checks and proved every one able to fail. Its
entry point runs and passes on the tree as it stands; its output is pasted into
§8 of the document as the second of the two suite examples, precisely because
§8 needs to show that `general/` has one. Nothing was left for this ticket to
build there.

### One defect found, reported not fixed

**`eos/mixed/api.py:eos_response` does not forward `phases=` / `muons=` to its
own central point** (`api.py:357`, the `centre = eos_point(par, mode, species,
...)` call). So the general two-`Phase` calling form — the one
`hybrid_table` and `eos_point` both accept — comes back
`converged=False` with `reason: "either (par, flags[, vmit_params]) or phases=
must be given"`, and nan in every quantity. Measured:

    phases=(dd2_phase, vmit_phase), n_B=0.75, eta=0   -> converged=False, chi=nan
    front door (par, flags, vmit_params=)             -> converged=True,  chi=0.373300

Non-convergence is a return value, so this is not a crash — it is worse in one
respect: a sampler using the general form scores every point as unsolvable and
never learns why. The front door is unaffected, which is why §11.5 of the
document calls that form and §11.1 says so explicitly rather than showing a call
that returns nan. **Not fixed here** — it is engine internals and this ticket's
whole diff is a document. Goes in the Stage 7 report.

Status: resolved.
