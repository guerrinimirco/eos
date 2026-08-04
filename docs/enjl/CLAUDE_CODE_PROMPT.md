# Claude Code kickoff prompt — `eos/enjl/`

Paste the block below into Claude Code with the repo root as cwd.

---

## The prompt

> We are building out `eos/enjl/`: the extended NJL model of Cheng-Jun Xia, in
> which baryons are treated as clusters of three quarks so that baryonic
> matter, quark matter and their transitions come out of one functional. The
> work has two phases in this session — reproduce the published T = 0 results,
> then extend to finite temperature using the Fermi-integral machinery already
> in `eos/general/`. A third phase (mixed phases of the first-order
> transitions, using our `eos/mixed/` framework) is explicitly deferred; do not
> start it.
>
> Read these four documents in order before writing any code:
>
> 1. `CLAUDE.md` — repository conventions. Invariants, not suggestions. The ones
>    that will bite on this task are §2 (S = +1 per s-quark; C excludes
>    leptons), §3 (`eos/general/fermi_integrals.py` returns **fm-based** units
>    and every call site must convert — do not leak natural units across a
>    module boundary), §4 (the `_ref`/`_fast` split, and that `_ref` is right
>    when they disagree), §6 (the JEL integrals stay a selectable option), §7
>    (the thermodynamic identities, in particular that the rearrangement term
>    enters µ and P but **never** ε), and §10 (ask rather than pick a
>    convention silently).
> 2. `docs/enjl/REFERENCE_TABLES.md` — what the five reference tables in
>    `test/enjl/reference/` actually contain. **Read this one properly.** It was
>    written from a numerical audit of the tables, and four of their columns do
>    not mean what their names suggest. If you skip it you will spend an
>    afternoon debugging a correct solver against a misread column: the
>    scalar-density columns, the baryon chemical potential, the electron
>    chemical potential, and 203 rows of one file that are interpolation rather
>    than solver output.
> 3. `docs/enjl/SPECIFICATION.md` — the physics specification, the state of the
>    existing code, and the milestone plan E0–E6 (T = 0) and T1–T4 (finite T)
>    with a validation gate on each.
> 4. `docs/enjl/verify_reference_tables.py` — run it. It is an independent
>    rebuild of the mean field from the published parameters that touches
>    nothing in `eos/`, and its printed numbers are where every tolerance in the
>    specification came from. It is your oracle: if it and `eos/enjl/` disagree,
>    one of them is wrong and the disagreement tells you where to look.
>
> The two papers are in the repository root as `Xia_2024_PRD_extendednjl.pdf`
> (Paper 1, uniform matter — all equation numbers in the specification refer to
> it) and `Xia_2026_preprint_mixedphase.pdf` (Paper 2, Thomas-Fermi mixed
> phases — needed only for the meson masses, and for the deferred third phase).
>
> Then work the milestones in order, starting with **E0**. Each has a gate and
> the gate must be green before the next one starts.
>
> Ground rules:
>
> - **`eos/general/`, `eos/dd2/`, `eos/vmit/`, `eos/tov/` and `eos/mixed/` are
>   validated baseline. Consume them as libraries.** In particular the finite-
>   temperature work is *not* a new Fermi-integral implementation: it is one
>   function in `eos/enjl/thermodynamics.py` that routes to
>   `eos.general.fermi_integrals.solve_fermi_jel` for the medium part and keeps
>   the Λ vacuum term analytic. `eos/dd2/physics/thermo.py::kinetic_thermo` is
>   that exact pattern for the nucleonic sector — read it and follow it. If you
>   find yourself writing a Fermi-Dirac quadrature, stop.
> - **The Λ cut-off is a temperature-independent additive term.** This is the
>   hinge of the whole extension (specification §1.3) and the reference tables
>   already display the split. Getting it right makes finite T nearly free;
>   getting it wrong will look like a broken gap equation.
> - **There is one bug to fix before anything else.**
>   `eos/enjl/eos_beta.py` does not import: it annotates `pt: ENJLEoSPoint` but
>   never imports that name. Add it to the `from eos.enjl.uniform import (...)`
>   list. That module has consequently never been executed, so treat every
>   number it produces as unverified.
> - **Do not "fix" the factor 9 in the ρ coupling.** `RHO_FACTOR = 9.0` in
>   `eos/enjl/parameters.py` looks like it contradicts the paper's printed
>   Eq. (22). It is correct, and it is now confirmed two independent ways —
>   from the symmetry energies, and by reading the coupling straight off the
>   reference tables' isospin splitting, which gives exactly 9.0000×
>   (REFERENCE_TABLES.md §4c).
> - **Do not add a thermal meson gas.** In this model σ, ω and ρ are auxiliary
>   fields eliminated in favour of g²/m². `eos/general/bose_integrals.py` is
>   the right tool for a model with dynamical mesons; this is not one.
> - **Three finite-temperature physics questions are genuinely open** and are
>   listed in specification §3.3: thermal antibaryons, thermal antiquarks, and
>   whether the paper's "quarks restricted to the lowest energy states" remark
>   has finite-T content. The specification states a default and a reason for
>   each. Record the choice in `DD2_OPEN_QUESTIONS.md`, and on the third one —
>   the quarkyonic restriction — ask me rather than deciding, because it changes
>   the model rather than a convention.
> - New code stays in `eos/enjl/` and `test/enjl/`. Follow the existing
>   docstring style: state the physics the module implements, name the equation,
>   cite the paper. Per `CLAUDE.md` §10, docstrings must stand on their own —
>   **no references to `docs/` or to milestone numbers in `eos/` or `test/`**.
>   "Eq. (12) of Xia 2024, PRD 110, 014022" is right; "see spec §1.3" is not.
> - Run `python -m pytest test/ -q` **before you start** to record the baseline,
>   then after each milestone. The suite currently has a dozen pre-existing
>   failures that are environment or unrelated modules — a missing `numba`, a
>   missing `typing` import in `eos/alphabag/compute_tables.py`, an absent crust
>   file, an external `nucleation` package — and `eos.enjl.eos_beta` is in that
>   list only because of the import bug above. Do not go fixing the unrelated
>   ones. `test/enjl` on its own is 10 passed and must stay green.
> - Do not loosen a tolerance to make a test pass. Where the reference data is
>   genuinely less converged than the identity being tested — there is exactly
>   one such region, and REFERENCE_TABLES.md §4b names it — exclude it with a
>   comment saying why.
>
> Start by reading the four documents and running the audit script, then tell me
> your plan for E0 and E1 before implementing.

---

## Notes for the human

**Work on a branch.** `git checkout -b enjl-finite-T` before starting.

**What this session established**, so you know what is already load-bearing and
does not need re-deriving:

- The reference tables satisfy their own gap equation, Eq. (19), Eq. (23) and
  the mean-field Eqs. (14)-(18) at the tolerances tabulated **per file** in
  `REFERENCE_TABLES.md` §4 — per file because they differ by up to five orders
  of magnitude on the same identity, so a single global tolerance is either
  vacuous or wrong. The strongest of them is the mean-field rebuild: µ_i
  reproduced to 0.0047 MeV on the tightest file and 0.20 MeV on the loosest,
  from the published parameters and the tables' own densities, against
  potentials of 1000-2500 MeV. That is the number to build the numerical gate on
  — it leaves no room for a sign error, a wrong factor or a missing
  rearrangement term.
- The factor 9 in Γ_ρ is confirmed independently of the symmetry-energy
  argument that originally motivated it.
- E₀ = 4263.845 MeV/fm³ from `vacuum_energy_density()` matches both the Maple
  worksheet's hard-coded constant and the tables' offset.
- Four verified Maxwell coexistence windows are available as E4 test data.
- The Λ cut-off contributes a strictly T-independent additive term, and
  `solve_fermi_jel` at T ≤ 10⁻⁵ MeV reproduces the exact T = 0 closed forms to
  10⁻¹³ relative or better across the (M, g, k_F) range this model needs
  (4×10⁻¹⁰ for P, which is formed as a cancelling difference). Together these
  mean the finite-T path has a T = 0 limit that is a free, sharp regression test
  (gates T1/T2).

**Four traps in the reference data**, all in `REFERENCE_TABLES.md` but worth
having in front of you when reviewing a diff:

1. `nsq` is the quark scalar density *without* the vacuum term; `Sigmaq` is the
   effective density n̄ˢ_q *with* it. The gap equation takes `Sigmaq`.
2. `munr`, not `mun`, is µ_b. They agree while baryons exist and diverge by
   hundreds of MeV afterwards.
3. `mue` and `mumu` are written as the lepton *mass* when that lepton is
   absent. µ_e is still nonzero there and equals µ_d − µ_u.
4. `Beta_fq0.5_B1.dat` is 180 solved rows plus 203 rows of linear interpolation
   across two Maxwell plateaus. A blank `munr` flags them;
   `reference.solved_rows()` does the masking.

**Baseline test state — measure it yourself before starting.** On the
interpreter used for this session `python -m pytest test/ -q` gave
**12 failed, 359 passed, 27 skipped, 6 errors**, and almost all of that is
environment rather than code:

- **`numba` is not installed** on that interpreter, though `pyproject.toml`
  lists it. `eos/tov/solver_fast.py` imports it at module level, so this takes
  down `test_module_imports[eos.tov.solver_fast]`, all three
  `test/mixed/test_tov_backend_parity.py` tests, and all three
  `test/mixed/test_tables_and_tov.py` errors — seven items. The managed `dd2`
  conda environment does have numba 0.66.0; run the suite there, or
  `pip install numba`, before concluding anything about the baseline.
- **`test/dd2/test_dd2_m4_tov.py` and the two `test/mixed/test_scan.py`
  failures are the crust issue** documented at length in
  `docs/phase2/CLAUDE_CODE_PROMPT.md`: with `~/Desktop/Research/Crust/BPST0.dat`
  unreachable the crust is silently dropped and R(1.4 M☉) comes out 12.33 km
  against the 13.2 ± 0.4 km the test asserts — the same ~0.9 km deficit noted in
  milestone E6 below. Worth knowing about *because* E6 will hit it.
- **`eos/alphabag/compute_tables.py:724` raises `NameError: name 'Any' is not
  defined`** — a missing `typing.Any` import. Pre-existing and unrelated to this
  task, but it takes down three tests, because anything importing that module
  fails: `test_module_imports[eos.alphabag.compute_tables]`,
  `test_module_imports[eos.sfhoalphabag.hybrid_table_generator]`, and
  `test_eos_never_imports_nucleation` (which walks every module).
- **`test/tov/test_rotating.py` needs external RNS data** —
  `~/Desktop/Research/rns-main-official/source/eos/eosC`, absent here. One
  failure and two errors. This is *not* the silent-crust-drop issue described in
  `docs/phase2/CLAUDE_CODE_PROMPT.md`; it raises `FileNotFoundError` outright.
- **`test/dd2/test_notebook_api.py`** imports the external `nucleation`
  package: one error.
- **`eos.enjl.eos_beta` is in that list**, and that one *is* yours — the
  missing-import bug of E0. It is the only entry you should expect to clear.

That accounts for the whole 12-failed/6-error set. Run the suite first, record
your own number, and treat it as the line to hold: `test/enjl` on its own is
10 passed and must stay so. Do not fix the `alphabag`, numba or RNS-data
failures as a side quest — but do not mistake them for something you broke
either.

**Division of labour.** The loop that worked for Phase 2 applies here: Claude
Code implements a milestone, you review the diff, then bring the branch back to
Claude Science for numerical validation — running the audit script against the
new code, checking the thermodynamic identities on a (n_b, T) grid, verifying
the T → 0 limit is exact, and pushing tables through TOV. Claude Code
implements; the analysis session checks the physics.

**The one question worth settling early** is the third open item in
specification §3.3. Paper 1's statement that quarks are "restricted to the
lowest energy states" in the quarkyonic phase is a T = 0 kinematic remark on its
face, and the default reading treats it as having no finite-T content. If it is
instead a genuine constraint on the quark occupation, that changes the finite-T
model rather than a convention, and it is much cheaper to decide before T3 than
after.
