# `eos.enjl.api` documents a name the module does not export

Type: task
Status: resolved
Blocked by: none
Parent: ../map.md

## Question

`eos/enjl/api.py:106` tells the caller that `x0` is "a starting guess in the
order of `eos.enjl.solver.UNKNOWNS`". **There is no `UNKNOWNS`.**
`eos/enjl/solver.py:133` defines `BASE_UNKNOWNS`, the ten of the base vector,
and `:137 unknown_slots(spec)` returns those ten plus one potential per held
charge — so the true ordering is mode-dependent and the docstring names neither
function.

This is the documented contract for a public argument: a caller who follows it
gets an `ImportError`, and a caller who guesses gets an ordering that is right
for `beta_eq_neutrinoless` and wrong for a mode with an extra held charge.

Found while writing [ticket 19](19-enjl-stepwise.md)'s step-1 cell, which
prints the ten unknowns by name rather than importing anything. It is not that
ticket's to fix (notebook-only scope) and
[ticket 54](54-signature-corrections.md), where a signature correction would
have belonged, is resolved.

Done when the docstring names `unknown_slots` (and `BASE_UNKNOWNS` as the base
of it), says the ordering depends on the mode, and no other public docstring in
`eos/enjl` points at a name the module does not carry — the same grep is worth
running over the other models while it is open.


## Answer

**Shipped in commit `4d0ab58`.** Three docstrings corrected, all in `eos/`; no
executable line touched, nothing in `docs/` or `notebooks/`.

**The named fix.** `eos/enjl/api.py` now says the ordering DEPENDS ON THE MODE,
names `unknown_slots(spec)` as the thing that produces it and `BASE_UNKNOWNS` as
its base, lists the ten base slots by name, and tells the caller to build the
vector from `unknown_slots` rather than by counting. Measured, so the docstring
states facts rather than a reading of the source:

    beta_eq_neutrinoless       10 slots   extras: ()
    fixed_YC                   10 slots   extras: ()
    fixed_YC_YS                11 slots   extras: ('mu_S',)
    beta_eq_neutrino_trapped   11 slots   extras: ('mu_nue',)

So the failure mode is worse than a wrong order: a ten-long vector written for
`beta_eq_neutrinoless` is the wrong LENGTH for two of the four modes.

**The grep, run as asked — and it paid.** Every `eos.*` name written in
backticks in any public docstring (module, class or function) under `eos/`,
resolved by actually importing it: **169 files, 17 unresolvable, 13 of them
false positives** — the CompOSE data-file names `eos.nb`, `eos.t`, `eos.thermo`,
`eos.yq` and `eos.thermo.ns` in `general/compose.py` and `astro/tov/crust.py`,
which are filenames that merely look like modules. **Four were real:**

| site | names | truth |
|---|---|---|
| `eos/enjl/api.py:106` | `eos.enjl.solver.UNKNOWNS` | does not exist; the ticket's target |
| `eos/enjl/solver.py:83` | `eos.enjl.verify.check_entropy_limit` | real function, but in `verify/run_full_check.py`; `verify/__init__.py` re-exports only `run_full_check` |
| `eos/vmit/compute_tables.py:5` | `eos.vmit.solver_table` | has never existed; the uniform entry point is `eos.vmit.eos_table` |
| `eos/astro/gmode/background.py:215` | `eos.astro.tov.solver.load_crust_table` | real function, but in `eos.astro.tov.crust` |

The first three are fixed. **The fourth is left alone deliberately**: `astro` is
not one of the nine models this ticket opened, and the map's hard rule is that a
defect noticed rather than asked about goes in the report, never in the diff. It
is a one-line docstring fix for whoever picks it up. Re-running the grep after
the commit gives 14, all of them the CompOSE names plus that one.

The second enjl instance is the ticket's own prediction paying off — it asked
whether any OTHER public docstring in `eos/enjl` points at a name the module does
not carry, and one did, sixty lines away in `solver.py`.

**A second pass found no more.** Bare backticked identifiers (not dotted) in
every model's `api.py` docstrings, checked against that model's package,
`solver`, `table`, `thermodynamics`, `parameters` and `species` namespaces: the
only hits are argument names (`rel_step`, `n_B_grid`, `vmit_params`), mode names
(`fixed_YC`) and result keys (`cs2_isothermal`, `branch_changed`) — prose about
the API, not module attributes. The dotted form is where this class of defect
lives.

**Gate.** Interpreter **python.org 3.14.2** (numpy 2.3.5, scipy 1.17.0), isolated
`git archive HEAD` copy with the three files overlaid: `test/enjl` + `test/vmit`,
**165 collected, 164 passed, 1 failed**. That one failure is not this ticket's
and is not `eos/enjl`'s: `test/vmit/test_uniform_api.py::test_response_equilibrium`
asserts on `resp["cs2_isothermal"]`, which is [ticket 69](69-cs2-eq-naming.md)'s
rename landed on the shared gitignored `test/` tree while HEAD's
`eos/vmit/api.py` still returns `cs2_eq`. Copying that session's working-tree
`eos/vmit/api.py` into the gate turns the file green — **18 passed**. Zero added.
