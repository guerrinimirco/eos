# The eight conformance fixes that move no number

Type: task
Status: resolved
Blocked by: 11
Parent: ../map.md

## Question

[Ticket 11](11-conformance-triage.md) ruled these (a)-class. They are grouped
because none of them can change a converged quantity, so they share one gate and
one commit. Each is located with file:line in
[conformance-table.md](../research/conformance-table.md).

1. **`ccdm` is outside the layering gate** (finding 4). `test/test_imports.py:76
   MODEL_PACKAGES` omits it, so neither `test_a_model_imports_only_general` nor
   `test_no_model_imports_astro` runs against it. Its graph is clean today; this
   is a gap in the gate. One word.

2. **`fracs` drops the *fixed* fractions** (finding 7). §5: "`fracs` carries
   every fraction the line was solved at, **swept or fixed**."
   `eos/dd2/table.py:275` and `eos/sfho/table.py:333` pass `combos[-1][1]`, which
   holds only the swept keys, while the full set is built two lines earlier at
   `dd2/table.py:234`. So `eos_table(..., "fixed_YC_YS", axes={...'Y_C'...},
   fixed={'Y_S': 0.0})` reports `fracs={'Y_C': ...}` and loses `Y_S`. This is the
   one outright key-content violation of the progress contract, and it defeats
   §5's stated purpose that one printer serves every model. Pass `fracs`.

3. **dd2's parameter classmethods are the exact anti-pattern §5 names**
   (finding 9). `eos/dd2/parameters.py:220 from_hyperon_potentials` and
   `:254 from_delta_potential` each carry `from eos.dd2.solver import solve_snm
   # local import breaks the cycle` (`:249`, `:282`). §5: a constructor of this
   shape "is therefore a free function in `nmp.py`, not a classmethod on the
   parameter dataclass — putting it there forces a deferred import, which is the
   cycle announcing itself." `eos/dd2/nmp.py` already holds `invert_nmp` and
   `from_nmp` correctly, so the destination exists.

4. **`dd2/solver.py:880` imports upward from `table.py`** (finding 10).
   `from eos.dd2.table import _mode_kwargs`, while `table.py:21` imports
   `solve_octet` from `solver.py` — a real cycle deferred to hide it.
   `_mode_kwargs` is solver vocabulary; `did` puts the same thing in `solver.py`
   and imports it upward from `api.py:24`, which is the right shape. Two smaller
   siblings in the same package: `dd2/table.py:335 from eos.dd2 import
   Parametrization` (a submodule reaching back through its own `__init__`) and
   `dd2/nmp.py:406,420 from eos.dd2.nmp import esym` (a module importing itself).

5. **`eos/mixed/backends/` is not deletable** (finding 11).
   `eos/mixed/verify/run_full_check.py:44` imports `mixed_jacobian` at module
   scope — the only unconditional module-scope backend import in the repository.
   §5: "**`backends/` is deletable.** Remove it and the model still gives the same
   numbers, only slower." `dd2/verify:97` and `sfho/verify:304,395` defer the same
   import inside functions. One line.

6. **`alphabag` re-derives `quark_charges` five times** (finding 12).
   `eos/alphabag/solver.py:441,521,613,700,745` write
   `n_C = (2/3)n_u - (1/3)n_d - (1/3)n_s` in literal fractions, plus
   `n_B = (n_u+n_d+n_s)/3` at four of them. §2: "Basis changes are declared once
   … No model carries its own copy of these algebraic maps." The model
   contradicts itself — `alphabag/thermodynamics.py:36` already imports
   `quark_charges` from `eos.general.basis`, and
   `alphabag/verify/run_full_check.py:21` claims "no local copy of the map",
   which holds for the reported charges but not for the residual rows.

7. **A second `quark_charges` in `eos/mixed/charges.py:157`** (finding 13),
   alongside the one `mixed/adapters.py:50` imports from `eos.general.basis`.
   Built from the shared `Particle` objects so it cannot drift in sign, but the
   engine now has two functions of that name in scope.

8. **`abpr`'s `eos_table` docstring claims array arithmetic it does not do**
   (finding 23). `eos/abpr/api.py:154-158` says "the grid is evaluated by array
   arithmetic"; `:185` is `[solve_cfl(float(n), par, T=T) for n in nB]`, a bare
   Python loop. The physics justification for having no warm start is sound; the
   array claim is not. ABPR is the one model where genuine array-in/array-out is
   reachable (§6). **Either** vectorise **or** correct the docstring — rule which
   while doing it, and say why.

Items 6 and 7 touch the charge map, so they are checked against `test/baseline/`
for `alphabag` and `mixed` at rtol = 1e-10; the numbers are already correct, so
any movement means the dedup was wrong. The other six cannot move a number.
Report added failures against `output/_audit/pytest_before_with_crust.txt`.


## Answer

**Seven of the eight shipped in commit `5c75584`; the eighth is real, done, and
uncommittable.** None moved a number, which is what the grouping was for.

`test/test_imports.py:76 MODEL_PACKAGES` now carries `ccdm` and `eos.ccdm`
passes both layering tests — but `test/` is gitignored (CLAUDE.md §11), so that
one word lives in the working tree only and appears in no diff. It is recorded
here because a future session reading the commit will not find it.

**What each fix turned out to be, where the ticket's description was incomplete:**

- **`fracs` (item 2).** Fixed by passing the local `fracs` dict, which both
  builders already assemble two lines above. `combos` is deliberately NOT
  changed: it indexes the grid axes and is consumed by `rows_from_result`
  (`dd2/table.py:310`, `sfho/table.py:359,563`), so widening it would move table
  CONTENT, which is exactly the thing this commit is defined by not doing. A
  comment at each site now says why the two differ.

- **The dd2 constructors (item 3).** The move is bigger than the ticket implies,
  because `Parameters.from_hyperon_potentials` is a PUBLIC name with callers:
  `mixed/scan.py:169,172` and five test files
  (`test/dd2/test_dd2_m4.py`, `test_dd2_m5.py`, `test/mixed/test_mixed_responses.py`,
  `test_window_location.py`, and `test/baseline/generate_baseline.py`, which is
  what freezes the dd2 baseline). All updated. Both are re-exported from
  `eos.dd2`, so `from eos.dd2 import from_hyperon_potentials` works.
  `nmp.py` already imported `solve_snm` at module scope, so the two
  `# local import breaks the cycle` lines are simply gone rather than moved.

- **`_mode_kwargs` (item 4).** It cannot move alone: it reads `MODES` and
  `MODE_FRACTIONS`, which also lived in `table.py`. All three moved to
  `solver.py`. `eos.dd2.MODES` and `eos.dd2.table.MODES` both still resolve
  (table.py imports them), so no consumer changed. The two siblings went with
  it: `nmp.py` imported `esym` from itself twice, and `table.py`'s `__main__`
  block reached back through `eos.dd2.__init__`.

- **The `abpr` docstring (item 8) — RULED: correct the docstring, do not
  vectorise.** The physics permits vectorising (every point is independent and
  the density inverse is closed-form), but `solve_cfl` takes one scalar density,
  so genuine array-in/array-out is a change to the SOLVER signature, not to the
  table driver — and a solver change has to be gated on numbers moving, which is
  the one thing this commit's grouping forbids. The docstring now states the
  loop, says array-in/array-out is reachable for this model and no other, and
  says it has not been made. Vectorising remains available as its own ticket.

- **Items 6 and 7 (the charge map) passed their extra gate.** `alphabag` and
  `mixed` both pass `test/baseline/` at rtol = 1e-10, so the dedup was arithmetic
  identity as claimed. `mixed/charges.py` re-exports `eos.general.basis.quark_charges`
  rather than deleting the name, so `eos.mixed.quark_charges` is unchanged and is
  now the same object `mixed/adapters.py:50` already imported. `QUARK_QN` stays:
  it is a quantum-number table built from the shared `Particle` objects, not a
  second copy of the basis map.

**Gate.** Interpreter **python.org 3.14.2** (numpy 2.3.5, scipy 1.17.0), run in
an isolated `git archive HEAD` copy with the changed files overlaid, because two
other sessions were live in the checkout. The full suite was NOT run, for the
same reason.

    test/baseline/                              16 collected,  6 failed,   10 passed
    test_imports + dd2 + sfho + alphabag
      + mixed + abpr + ccdm                   1083 collected,  3 failed, 1080 passed

**Zero failures added.** All nine are pre-existing on this interpreter and named
in `output/_audit/`: the six baselines (`ccdm`, `dd2`, `enjl`, `njl`, `tov`,
`zlvmit`) in `pytest_after_ticket61_baseline_py314.txt`, and the three dd2
NMP-inversion failures (`test_inversion_without_Q_sat_predicts_it`,
`test_inversion_with_Q_sat_still_imposes_it`,
`test_restarts_recover_a_seed_limited_inversion`) in every `_py314` file back to
`pytest_before_ticket62_py314.txt`. No tolerance was loosened.

**One trap for the next session using this gate shape.** Overlaying only the
files a ticket touches onto `git archive HEAD` pairs HEAD's `eos/` with the LIVE
`test/`, which is shared and gitignored. Where a concurrent session has landed
its test-side edit and not its code-side one, that pairing invents a failure
belonging to neither ticket — ticket 69's `cs2_eq` -> `cs2_isothermal` did
exactly this in ticket 71's gate. Attribute before believing.
