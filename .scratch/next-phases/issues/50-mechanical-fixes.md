# The eight conformance fixes that move no number

Type: task
Status: open
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
