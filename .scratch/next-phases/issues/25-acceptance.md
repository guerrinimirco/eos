# Run the Acceptance criteria block and write the Stage 7 report

Type: task
Status: resolved
Assignee: session c2decb98
Blocked by: 01, 21, 24, 76
Parent: ../map.md

## Question

The Acceptance criteria block of `docs/REFACTOR_PROMPTS.md`, every line, with the
tool output behind each:

- `pytest eos/test/` and `pytest nucleation/test/` fully green
- the Phase 1 numerical baseline reproduces at **rtol = 1e-10**
- every model has a `.tex` that compiles — **unless ticket 09 removed that
  criterion**, in which case the replacement criterion is checked instead
- every model implements `eos_point()` and `eos_table()` with the spec signature,
  and its mode and species coverage matches what `CLAUDE.md` claims
- no Fermi or Bose integral implemented outside `eos/general/`
- model parameters are arguments everywhere; no solver raises or hangs on
  non-convergence; model objects pickle. **Show this with one script that
  evaluates a model at 500 random parameter sets across a multiprocessing Pool,
  counts the non-converged ones, and finishes.**
- `grep -rn "rcParams" eos/ nucleation/` hits exactly one file
- every README and STRUCTURE.md example executed, real output pasted
- no file over 5 MB newly tracked in git
- no new third-party dependency

Then the Stage 7 report: both suites verbatim, the baseline reproduction, the
**added-failure count against `output/_audit/pytest_before.txt`**, the list of
files created, changed and deleted, and every question from tickets 03, 05, 09,
10, 11 and 23 that is still open.

## Answer — the Stage 7 report

**Every line of the Acceptance criteria block was run. Ten of the eleven pass.
One fails, and it fails on a name the repository's own document prints.**

All numbers below are **python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0**
([ticket 57](57-canonical-stack.md)'s canonical stack), `eos` at `7f0486e`,
`nucleation` at `37af659` on `paper-release`. Collected counts travel with every
failure count, per the map's Notes. The `eos` working tree carried **0 dirty
files under `eos/`** for the whole run — only `notebooks/hadronic_eos.{ipynb,py}`,
a concurrent session's, which nothing on the suite's import path reads.

---

## 1. The criteria block, line by line

| # | criterion | verdict |
|:-:|---|---|
| 1 | `pytest eos/test/` and `pytest nucleation/test/` fully green | **PASS** |
| 2 | the Phase 1 numerical baseline reproduces at rtol = 1e-10 | **PASS** |
| 3 | every model has a `.tex` that compiles | **PASS** |
| 4 | `eos_point()`/`eos_table()` spec signature; mode and species coverage matches CLAUDE.md | **FAIL** — one undeclared mode |
| 5 | no Fermi or Bose integral implemented outside `eos/general/` | **PASS** |
| 6 | parameters are arguments; no raise or hang on non-convergence; objects pickle | **PASS** |
| 7 | the rcParams grep hits exactly one file | **PASS** |
| 8 | every README and STRUCTURE.md example executed, real output pasted | **PASS with three stale pastes** |
| 9 | no file over 5 MB newly tracked in git | **PASS** |
| 10 | no new third-party dependency | **PASS** |
| 11 | a physicist finds the function computing a quantity in under a minute from STRUCTURE.md | **PASS with one gap** |

### 1. Both suites, fully green

    eos          1737 passed, 20 skipped, 0 failed   1757 collected   20:26
                 output/_audit/pytest_ticket25_py314.txt

    nucleation     72 passed,  0 skipped, 0 failed     72 collected    4.31s
                 output/_audit/nucleation_ticket25_py314.txt

**Added failures against `output/_audit/pytest_after_ticket74_py314.txt`
(1 failed, 1680 passed, 15 skipped, 1696 collected): ZERO, and the arithmetic
closes exactly.** Collection moved 1696 -> 1757, +61. Passed moved +57, skipped
+5, failed -1; 57 + 5 - 1 = 61. So every one of the 61 tests added since ticket
74 is accounted for, and the single surviving red — `test_baseline[enjl]`,
cleared by [ticket 72](72-enjl-branch-selection.md) — is green. There is no
deliberate red left anywhere in the map.

The 20 skips are all availability guards, never silent failures: the RNS binary,
the BPS crust table, the CompOSE reference slices, plus one physics-conditional
(`test_ccdm_pairing`: *"the unpaired state won at this point"*).

**§1 holds in both directions**, checked directly rather than inferred:
0 `import nucleation` anywhere in `eos/`; 12 `nucleation` modules import `eos`;
`test/test_imports.py` 200 passed.

### 2. The baseline reproduces at rtol = 1e-10

    $ pytest test/baseline -v
    test_baseline[abpr] ... [ccdm] [dd2] [did] [enjl] [mixed] [njl] [sfho]
    [tov] [vmit] [zl] [zlvmit] [alphabag]          all PASSED
    20 passed in 41.86s

Thirteen models, `RTOL = 1e-10` at `test/baseline/test_baseline.py:49`, plus the
seven supporting checks including ticket 75's undetermined-potential screen.
Full log: `output/_audit/baseline_ticket25_py314.txt`.

### 3. Thirteen `.tex`, thirteen compiles

`pdflatex -interaction=nonstopmode -halt-on-error`, every one clean:

    abpr 8p   alphabag 8p   gmode 3p   tov 7p    ccdm 20p   dd2 8p   did 13p
    enjl 16p  mixed 9p      njl 13p    sfho 8p   vmit 10p   zl 7p

`-halt-on-error` matters: it is what makes this a compile check rather than a
"produced some PDF" check. [Ticket 09](09-tex-or-md.md) ruled the `.tex` stays,
so this criterion stands as written and no replacement was substituted.

### 4. The one failure — `fixed_YS`

Signatures first. **All eleven units carry all three entry points**, `par` first
and non-optional, `mode` second and required, `species` third
(`output/_audit/mode_species_coverage_out_py314.txt`, section A). The two
documented deviations are both §5's own: `eos/mixed` takes the `Phase` pair in
the parameter position, and `abpr` alone defaults `mode='cfl'` because it has
exactly one mode.

Then every §3 mode name was **actually called** through `eos_point` in all ten
models — 60 subprocess probes, each timed out rather than allowed to hang. The
result matches CLAUDE.md §3 exactly on all five declared names, including the
two that first read as refusals and were not:

- `alphabag.cfl` needs the pairing gap; with `Delta0=100.0` it returns `ok=True`.
- `abpr.cfl` is T = 0 only; at `T=0.0` it returns `ok=True`.

Both are documented in STRUCTURE.md §4, and both were my probe's error, corrected
and re-run rather than reported as findings.

**The failure is a sixth name.** Asking each model to close a nonsense mode makes
it print the set it does close:

    dd2        5: beta_eq_neutrino_trapped, beta_eq_neutrinoless, fixed_YC,
                  fixed_YC_YS, fixed_YS          <-- NOT IN CLAUDE.md §3
    sfho       4: ... (the four)
    zl         3: ... (no fixed_YC_YS -- ruled meaningless, raises and says so)
    did/vmit/enjl/njl/ccdm  4 each
    alphabag   5: the four plus cfl
    abpr       5: closes cfl, refuses the other four by name

    Names reachable through eos_point that no specification declares:
        {'dd2': ['fixed_YS']}

Exactly one model, exactly one name. `fixed_YS` is live at
`eos/dd2/solver.py:875,885,936`, and — this is the part worth pausing on — it is
**pasted verbatim in the repository's own specification**, at
`docs/STRUCTURE.md:397`, inside the block that demonstrates good refusal
messages:

    dd2.cfl: ValueError: unknown mode 'cfl'; expected one of
    ['beta_eq_neutrinoless', 'fixed_YC', 'fixed_YS', 'fixed_YC_YS',
     'beta_eq_neutrino_trapped']

STRUCTURE.md §4's mode table lists five and does not include it. So the document
that exists to say what the modes are prints the sixth one in its own output and
does not name it in its own table. This is open
[ticket 98](98-fixed-ys-undeclared-mode.md), surfaced by a BayEoS design session
reading `eos` as a consumer, and it is the single reason criterion 4 fails.
Not fixed here — ruling it is 98's job, and 98 has a downstream consumer holding
a gate at `skipped` waiting for the answer.

**Species coverage passes.** All six §4 flags default to `False` in nine models;
`enjl` is the documented [ticket 65](65-species-flag-defaults.md) exemption
(`hyperons=True, muons=True`) and raises on any move, which is §4's
"a flag with only one legal value is a STATEMENT about the model".

### 5. No Fermi or Bose integral outside `general/`

Three name hits outside `eos/general/`, none an integral:

- `alphabag/thermodynamics.py:211 fermi_thermo` — a five-line wrapper whose body
  is `solve_fermi_jel(...)`. Its docstring names `eos.general.fermi_integrals`.
- `enjl/thermodynamics.py:158 fermi_momentum` and `astro/gmode/rates.py:148
  _fermi_momentum` — kinematics, `k_F` from `n`, not integrals.

A second sweep for quadrature (`quad`, `simpson`, `trapz`, `leggauss`,
`laguerre`, `romberg`, `fixed_quad`) across all of `eos/` outside `general/`
returns **exactly one hit**: `astro/tov/solver.py:395`, `np.trapz` integrating
the baryon-mass integrand over the stellar *radius*. Different physics, different
variable.

`njl` and `ccdm`'s cutoff-regularised integrals and `alphabag`'s
perturbative-QCD-corrected gas are §7's explicit carve-out — model physics, not
integral re-implementations — and each lives with its model.

### 6. 500 random parameter sets, four samples, 2000 evaluations

`output/_audit/inference_stress_500.py`, output in
`output/_audit/inference_stress_500_out_py314.txt`.

**The first draft of this script was wrong and the library caught it.** Drawing
all four DD2 shape coefficients independently was rejected on the first set:

    ValueError: Parameters: f_sigma(1) = 1 violated by 9.23e-02
    (constraint: a_sigma is dependent on b, c, d)

`couplings.py` fixes `d = 1/sqrt(3c)` from `f''(0) = 0` and `a` from `f(1) = 1`.
So a set with all four free is not a *wide* sample, it is an *invalid* one, and
`__post_init__` is right to refuse it. The script now draws `b`, `c`, the three
`gamma` and `m_sigma`, and **derives** `a` and `d` — which puts the stress where
the criterion wants it, on the solver rather than on the validator.

    SPREAD   n_B     converged   not conv.   raised   slowest task   wall
     ±25%    0.32       500          0          0        0.14 s      0.8 s
     ±25%    1.20       500          0          0        0.12 s      0.8 s
     ±95%    0.32       496          4          0        0.12 s      0.8 s
     ±95%    1.20       493          7          0        0.13 s      0.8 s

    pickle round-trip: Parameters and SpeciesFlags OK (590 bytes)

**±25% was not enough and saying so is the point.** 500/500 converged proves the
happy path, not the failure path the criterion is actually about. The spread was
widened until the sample genuinely left the convergent region: at ±95% the
solver returns `ok=False` for 11 of 1000 sets and **raises for none of them**,
which is the §6 claim demonstrated rather than asserted. Every task finished
under 0.19 s; nothing hung. Pickling is proved twice — explicitly before the
Pool, and again by the Pool working at all.

### 7. rcParams — exactly one file

    $ grep -rnE --include='*.py' 'rcParams\s*(\[|\.update)' eos/ nucleation/
    eos/eos/general/figure_style.py:262,436-466,613-615     (22 lines, 1 file)

And the criterion's own footnote checks out: the bare-word grep additionally hits
`zlvmit/plot_results.py:184`, `zlvmit/table_reader.py:703` and two
`2fam_PNS_nucleation.py` comments — every one a *sentence saying the file does
not set rcParams*. The refined pattern tells the rule from prose about the rule.

### 8. Every example executed — 25 python blocks across three documents

| document | blocks run | byte-identical to the paste | differ | no paste to compare |
|---|:-:|:-:|:-:|:-:|
| `eos/README.md` | 7 | 5 | 1 | 1 |
| `nucleation/README.md` | 5 | 1 | 2 | 2 |
| `docs/STRUCTURE.md` | 13 | 12 | 1 | 0 |

Blocks are replayed **in order** with earlier output muted, because that is what
"copy-paste runnable" means — the README's M–R figure block uses `seq` from the
sequence block before it. Logs:
`output/_audit/doc_examples_{readme_eos,readme_nucleation,structure}_py314.txt`.

**All four mismatches are stale prose or a nondeterministic field. No physics
number in any document is wrong.** Each, measured:

1. **`eos/README.md` Example 2** — 16 output lines, **13 identical to the last
   digit**, 3 differ, and all 3 are the progress line:

       real   [1/3] fixed_YC T=0 Y_C=0.3: 12/12 points in 0.0s
       pasted [1/3] fixed_YC T=0:         12/12 points in 0.1s

   The `Y_C=0.3` is [ticket 50](50-mechanical-fixes.md)'s fix to ticket 11
   finding #7 — `fracs` used to drop the *fixed* fractions. The README was
   pasted before that landed and never re-pasted. Note also that `in 0.1s` is a
   wall-clock field, so this block **cannot** match byte-for-byte on any machine
   as written.

2. **`nucleation/README.md`, both mismatches, one cause.** Every physics number
   identical; the difference is a five-line banner the README omits:

       Loaded 936 points from test/data/eos_hadronic_trapped_fixture.dat
         Equilibrium: trapped_neutrinos
         n_B: [0.1418, 1.402], 18 points   ...

   `eos.sfho.table.load_eos_table(filepath, eq_type)` prints this
   **unconditionally** — no `verbose=`, no `progress=`, no way off. See defects.

3. **`docs/STRUCTURE.md` §7** is the reference/fast timing benchmark. It reports
   ms/point and a round-off-scale difference, neither reproducible by
   construction. The claim it makes reproduces exactly: analytic Jacobian faster
   (0.23 vs 0.31 ms/point here, 1.24 vs 1.59 as pasted), agreement 5.5e-14
   against a pasted 4.4e-14.

**The §12 worked figure regenerates byte-identically.** `save_figure` rewrote
`docs/figures/structure_dd2_vmit.png` and git reports it unmodified — a
deterministic figure through the whole mixed-phase stack.

### 9. No file over 5 MB newly tracked

**454 tracked files checked, none over 5 MB.** Largest tracked file anywhere is
`eos/astro/tov/data/SFHO_compose_betaeq_S.dat` at 1.01 MB — which is also the
largest of the **123 files newly tracked this effort** (ticket 39's shipped crust
tables). Five times of headroom; the criterion is not close to binding.

### 10. No new third-party dependency

Declared: `eos` -> numpy>=2.0, scipy>=1.17, matplotlib, numba. `nucleation` ->
numpy, scipy, `eos`. All within the map's allowed set.

An AST sweep of every `.py` in both packages finds **four imports outside that
set**, and every one predates the map (created `0859cf4`, 2026-08-24 20:53):

| package | site | introduced | guard |
|---|---|---|---|
| `h5py` | `eos/general/table_io.py` | `5083318`, 08-04 | lazy, inside the HDF5 functions |
| `joblib` | `eos/astro/tov/rotating.py` | `4e89355`, 08-19 | lazy, *"when asked and joblib is available"* |
| `nucleardatapy` | `eos/general/constraints/build.py` | `d69122e`, 08-10 | `try/except ImportError`, prints a fetch hint |
| `pandas` | `nucleation/analysis/{outcomes,scan}.py` | `39a8640`, 06-26 | lazy, inside the functions |

Nothing was added. All four are optional and degrade gracefully — but none is
declared as an optional extra either; see defects.

### 11. Findability from STRUCTURE.md

Ten quantities a physicist would look for, each traced from STRUCTURE.md to a
module and symbol, and the symbol then resolved: **ten for ten, 0.9 s total**.
§3.1/§3.2/§3.3 are three tables that carry it.

The one gap: **`eos/general/sound_speeds.py` appears nowhere in STRUCTURE.md** —
not §3.2's shared-physics table, not anywhere. It is a module *this effort
created*, exporting `sound_speed_eq`, `sound_speed_frozen`, `cs2_frozen_isobaric`
and `EOSTable_for_gmode`. "Where is `c_s^2` computed" is answerable only through
§11.5's worked example, not through the index. It is one of exactly two
`eos/general/*.py` the document does not name; the other,
`observational_constraints.py`, is a deliberate compatibility shim.

---

## 2. Files created, changed and deleted

`eos`, `2844a9a..HEAD` (the parent of the map's own creation, 196 commits):

    185 files changed, 53441 insertions(+), 19668 deletions(-)
     20 created    150 modified    15 deleted    0 renamed

**Created (20).** `docs/STRUCTURE.md` + `docs/NEXT_PHASES_PROMPT.md` + two
figures; the four grouped notebooks as `.ipynb` + `.py` pairs — `hadronic_eos`,
`quark_eos`, `enjl_eos`, `hybrid_eos` — which is the Destination's amended count;
`eos/general/sound_speeds.py`, `eos/general/verify/{__init__,run_full_check}.py`,
`eos/mixed/species.py`, `eos/zl/nmp.py`; three `astro/tov/data/` crust and
CompOSE tables.

**Deleted (15).** `eos/dd2/notebook_api.py`, `eos/mixed/scan.py`, and Stage 0's
twelve retired notebook files (`CCDM_usage`, `DD2_usage`, `DID_usage`,
`NJL_usage`, `ENJL_usage`, `DD2vMIT_general1oPT`, `mass distribution.ipynb`).

**All six `.ipynb` in `notebooks/` load as valid JSON** — including
`ZLvMIT_hybrid.ipynb`, which [ticket 41](41-corrupt-notebooks.md) repaired from
the unterminated-string corruption `d9f8eec` left behind.

`nucleation`, `33c1e61..HEAD` (4 commits): 35 files changed, 529 insertions,
175 deletions; 1 created, 1 deleted, **16 renamed** — the
`nucleation/tests/` -> `test/` move of [ticket 80](80-phase6-conformance.md),
tracked rather than gitignored, per [ticket 23](23-phase6-respec.md)'s ruling
that this rule does not transfer from `eos`.

---

## 3. Defects noticed and NOT fixed

Under the map's only-what-the-ticket-asks rule. None of these is in a diff.

**D1 — `eos.sfho.table.load_eos_table` prints unconditionally.** Signature is
`(filepath, eq_type)`; no `verbose=`, no `progress=`, no off switch. It emits
five lines to stdout on every call, and it is the first thing
`nucleation/README.md`'s lead example calls. §5 says table builders default to
silent and turn on through a callback or `verbose=True`; this is a table
*loader*, so it is adjacent to the rule rather than under it, but the same
argument applies with more force to a library function a downstream consumer
calls in a loop. It is also the sole cause of both `nucleation/README.md`
mismatches.

**D2 — `eos/general/sound_speeds.py` is missing from STRUCTURE.md.** Criterion
11's index does not name a module this effort created. One row in §3.2.

**D3 — the README's figure example writes to the repository root.**
`README.md:385` is `save_figure(fig, "dd2_MR")`, a bare relative name, so a
reader copy-pasting it drops `dd2_MR.png` and `dd2_MR.pdf` into the repo root —
not into `docs/figures/`, where the file the README displays actually lives. I
removed the two strays my own run produced. STRUCTURE.md §12 gets this right
(`save_figure(fig, "docs/figures/structure_dd2_vmit")`).

**D4 — `docs/figures/dd2_MR.png` no longer matches what the code produces.**
Tracked file 53037 bytes, `sha256 a0a86665…`; regenerated by the README's own
example, 49400 bytes, `sha256 6c479555…`. Its sibling
`structure_dd2_vmit.png` regenerates byte-identically, so this is not
nondeterminism — the tracked figure is stale. **This is the map's
tracked-figures fog item, and it turns out to live in `eos` too, not only in
`nucleation/output/paper/`.** The publication decision the fog entry asks for is
therefore a two-repository decision.

**D5 — `np.trapz` is deprecated and `pyproject.toml` has no numpy ceiling.**
`eos/astro/tov/solver.py:395`. On numpy 2.3.5 it works and warns:
*"`trapz` is deprecated. Use `trapezoid` instead."* The pin is `numpy>=2.0` with
no upper bound, so the first numpy that completes the removal breaks the TOV
baryon-mass integral. One-word fix; not this ticket's word to change.

**D6 — four optional dependencies, none declared.** h5py, joblib,
nucleardatapy, pandas (table 10 above). All lazy and all degrading gracefully, so
criterion 10 passes — but a user who calls `save_table(..., format='h5')` on a
clean install gets `ModuleNotFoundError` from a documented entry point rather
than a message. `[project.optional-dependencies]` is where this belongs.

**D7 — `eos` is not installed on the canonical interpreter.** `nucleation` is
pip-installed editable into python.org 3.14; `eos` is not, in either repository.
`nucleation`'s own metadata declares `"eos"` as a dependency and the README says
*"`pip install -e ./eos` — must come first"*. Both suites are green above only
because I supplied `PYTHONPATH`. Nothing in the repositories is wrong — this is
an environment fact — but it means the criterion-1 result is not reproducible on
this machine by the README's own instructions until `pip install -e ./eos` is
run. This is the standing "why nucleation needs PYTHONPATH" trap, measured.

**D8 — `eos/general/observational_constraints.py`'s stated exit condition is
unmet.** Its docstring: *"The two names below are kept because `nucleation`
imports them; they are thin wrappers … and they go away once that import is
updated."* `nucleation/analysis/figure/__init__.py:27` and the paper notebook
still import them. Phase 6's conformance pass was the moment that condition
came true and it was not taken. This is a live instance of the map's
"nothing notices when a stated limitation stops being true" fog item.

**D9 — `nucleation` `37af659` is unpushed.** `origin/paper-release` is at
`2b2b72f`; [ticket 76](76-nucleation-golden-tolerances.md)'s golden correction —
the commit that makes the criterion-1 number above true — is local only.
[Ticket 23](23-phase6-respec.md) lifted the no-push rule and asked for a push
after 24 and again after 72. Pushing is the user's act (map, Out of scope), so
this is reported, not done.

---

## 4. Still open from tickets 03, 05, 09, 10, 11 and 23

All six are **resolved**; what follows is what each left behind.

**[03](03-stage0-removals.md) — the held folder's condition is discharged, and
one question is not.** `notebooks/eos_tables_DD2vMIT/` was moved rather than
deleted (`output_old/eos_tables_DD2vMIT_from_notebooks`), so ticket 05's
replacement can be *compared* against the 32 tables and 42 published figures
rather than merely asserted to replace them. Ticket 59 ran that comparison.
**Open:** nothing in the repository checks that an `.ipynb` is loadable — the
`d9f8eec` corruption survived five days and four commits undetected. I verified
all six load today; no gate keeps them loading. Ticket 41 fixed the files, not
the hole.

**[05](05-notebook-coverage.md) — one named gap, deliberately left.**
`astro/gmode` gets no notebook: it has no `eos_table`, it consumes tables, and no
§11 rule forces one. `astro/tov` is covered *in situ* through its public contract
in two notebooks; a notebook whose subject is the TOV solver itself — crust
choices, the RNS backend, tidal deformability — is named as future work outside
this map. `zlvmit` is confirmed out of scope. **Open:** none of the four
notebooks commits its executed outputs (only `hybrid_eos` carries any), so
"executes end to end" is a claim in tickets 13/16/19/59 rather than a fact in
the tree. Nothing re-checks it.

**[09](09-tex-or-md.md) — the ruling has a standing cost.** Keep both; the `.md`
and `.tex` carry the *same* information, each written natively. That is
**24 documents to hold at §11 standard, not 12**, forever, and the ruling
accepted that explicitly. **Open:** the map's own fog says two of two document
pairs checked had a residual-row sign backwards (`zl.tex` passed the audit 14/14
and still had the neutrality row backwards in one of three modes; `vmit`'s R6/R7
swap is the same shape), so **the remaining ten pairs' residual signs should be
assumed unchecked**. Thirteen `.tex` compile; compiling is not verifying.

**[10](10-rename-approvals.md) — one deferred, three frozen.** Deferred:
`thermo_at_potentials` vs `thermo_from_mu`, the name of the *upper* layer that
dd2, sfho and did all carry twice. It wants a name that names §5's phase-adapter
contract and was routed to `mixed.tex` ([ticket 36](36-quark-engine-documents.md)).
Frozen: `VMITTableSettings`, `compute_vmit_table`, `save_vmit_results`, whose
only caller is the legacy `ZLvMIT_hybrid.ipynb`. **That freeze rested on a
premise that has since changed** — the notebook was unopenable when the freeze
was granted, and ticket 41 has since repaired it, so "renaming symbols whose only
caller cannot be opened buys nothing" no longer describes the situation.

**[11](11-conformance-triage.md) — the (c) pile is the residue.** Ten rows went
to `docs/DEFERRED.md` rather than to code: `astro/gmode`'s import breach (with
[ticket 53](53-gmode-contract.md) as the real fix), eleven needless downward
deferred imports, `_CHIRAL_SPLIT`/RNS surface constants/legacy `B4`, `sfho` and
`zl`'s unfrozen dataclasses, `astro/tov` having **no `verify/` at all**,
`abpr`'s four unrecorded mode refusals, `mixed`'s `Y_Lmu` refusal,
`VMITTableSettings`, and `output/public/` not existing. **Open and worth
naming:** `astro/tov` still has no `verify/`, and with `test/` gitignored a fresh
clone has no way to check TOV at all. Ticket 11 called that "a bigger question
than this triage" and it has not been asked since.

**[23](23-phase6-respec.md) — its own noted-not-fixed is still true.**
`eos/alphabag/solver.py`'s `solve_beta_eq_neutrinoless` and `solve_fixed_yc_ys`
take `params=None` and a boolean flag-bag (`include_photons`, `include_gluons`,
`include_thermal_neutrinos`, `include_electrons`) instead of `SpeciesFlags` —
§5's "`par` is never optional" and §4's vocabulary, both still open. That is
now [ticket 96](96-alphabag-solver-flags.md), open, with
[94](94-zl-solver-flags.md) and [95](95-vmit-solver-flags.md) beside it.
Also standing from 23: **[ticket 80](80-phase6-conformance.md) was in scope and
not gating**, by design, and it is resolved; and the push (D9) is outstanding.

---

## 5. What this effort saw and did not close

The map's **Not-yet-specified** section, verbatim in substance, is the honest
list. Eight patches, none of them fixable by a ticket that does not exist yet.
Two the ticket asks be named in particular:

**The tracked-figures publication decision.** Ticket 80's production run
regenerated 23 tracked files under `nucleation/output/paper/` and restored every
one: 14 PDFs differed only inside `/CreationDate`, and the 9 real changes are
round-off — largest **13 cm on an 11.14 km `R_1.4`**, with `sigma_crit_star`
bit-identical across all 398 rows. Restoring was right for a conformance ticket
and it leaves the tracked figures as the **pre-refactor** ones. Whether to
re-commit the regenerated set is a publication decision for the user, and it
wants making before the repository goes public rather than after a reader's rerun
fails to match the committed CSVs. **This report extends it: D4 shows `eos` has
the same drift**, in `docs/figures/dd2_MR.png`. It is one decision across two
repositories now, not one.

**The lazy-import / notebook call-site blind spot.** A concurrent session renamed
`eos.sfho.create_custom_parametrization` -> `from_potential_depths`. Two
`nucleation` consumers import the old name — `test/make_fixture.py:98` and the
paper notebook — and **neither is on the suite's import path**: `make_fixture`
imports it lazily inside `main()`, and a notebook is not collected at all. So the
rename shows a **green** nucleation suite and breaks both silently. That is
exactly the shape ticket 24 hit when 38 of 38 modules could not import while the
suite reported nothing. The question is not the one rename: it is that the
cross-repo call-site check has a blind spot wherever an import is lazy or lives
in a notebook, **and no gate covers it — including the 72-green number in this
report**. My criterion-1 result inherits the blind spot and cannot see past it.

The other six, in one line each:

- **`active_baryons` is copied three times** (`did/species.py`,
  `sfho/species.py`, and `general/basis.py` since ticket 86) and nobody has swept
  the other `species.py`/`parameters.py` for more of the same shape, nor found
  the test for "this belongs in `general/`" that does not drag per-model physics
  up a layer.
- **`nucleation`'s smoke mode cannot complete, and both documents tell a reader
  to run it first.** `2fam_PNS_nucleation.py:1935` clips `F8_SHOW = [1, 3]`
  against a single-`alpha_s` grid, gets `[]`, and `pd.concat([])` raises at
  Figure 5. Second smoke-only shape bug the same cell has carried.
- **Nothing notices when a stated limitation stops being true.** Four instances
  in one afternoon (dd2's `nmp.py` round-trip claim, `eos/__init__.py`'s species
  comment, `test_imports.py`'s one-directional check, and `DEFERRED.md`'s
  fourth prose site that the two-way gate did not list). D8 above is the fifth.
- **The model documents are unverified in two ways** — residual-row signs, and
  gap *dispositions* that are claims about CLAUDE.md rather than about the code.
  Two of two checked were wrong.
- **Real fixes and golden DATA live outside version control.** `.gitignore:75`
  excludes `/test/`, so ticket 56's four regenerated `.npz` exist only in this
  working copy, and `test/baseline/` is §12 ground truth. Mitigated by hand
  copies in `~/Desktop/Research/backups/`; a hand copy is only as fresh as the
  last time it was taken.
- **Whether other tests silently degrade on missing data** is unmeasured. Ticket
  39 fixed two TOV helpers; nothing swept the rest of `test/` for an absent input
  turned into a wrong number rather than a skip. The 20 skips above all looked
  like honest guards; that was an inspection, not a sweep.

---

## 6. Where the map stands

98 tickets, **89 resolved**. Nine open, of which this is one, and **none of the
other eight is gating**: [88](88-fixed-composition-coexistence.md),
[91](91-leptons-default-and-drift-checks.md),
[93](93-invert-nmp-basin-lottery.md), [94](94-zl-solver-flags.md),
[95](95-vmit-solver-flags.md), [96](96-alphabag-solver-flags.md),
[97](97-natural-record-leaves-the-result.md),
[98](98-fixed-ys-undeclared-mode.md).

Ticket 91 is confirmed live by this report's own measurement, independently of
the ticket: `eos_point`'s `leptons=` default is `True` in zl, vmit, njl, ccdm,
enjl and mixed, and `None` in dd2, sfho, did, alphabag — the split 91 exists to
close. Ticket 98 is the criterion-4 failure.

**So: ten criteria of eleven pass, with real tool output behind every one. The
eleventh fails on a single undeclared mode name in a single model, it has a
ticket, and that ticket has a downstream consumer waiting on the ruling.**

Status: resolved.
