# `zlvmit`'s pure-phase warm-start calls have been dead since ticket 90

Type: grilling
Status: resolved (2026-08-29)
Blocked by: -
Parent: ../map.md

## Question

Found by [ticket 94](94-zl-solver-flags.md) while enumerating zl's call sites.

`eos/zlvmit/mixed_phase_eos.py` builds its initial guess by first solving each
pure phase, and falls back to hardcoded values if that fails:

    try:
        if chi < 0.5:
            if eq_mode == "trapped":
                H = solve_pure_H_trapped(n_B_est, T, Y_L, zl_params)
            ...
        else:
            Q = solve_pure_Q_beta(n_B_est, T, vmit_params)
            ...
    except:
        pass

**Every one of those six calls uses the PRE-[ticket 90](90-solver-signature-and-units-sweep.md)
argument order.** Since that ticket put `par` first and required,
`solve_pure_H_beta(n_B_est, T, zl_params)` binds `par = n_B_est` (a float),
and `default_guess` dies on `par.m_p`. The bare `except: pass` swallows it, so
the routine has silently taken its hardcoded fallback ever since — and, past
ticket 94 and 95, the arity is wrong as well. Ticket 90 moved 139 call sites by
an AST rewrite plus 22 by hand; these are in the 22 it did not reach, and
nothing went red because nothing can.

`H.mu_p_H` / `H.n_p_H` / `Q.mu_u_Q` are also not fields of the `EoSPoint`
either model returns today, so the block would need more than an argument
reorder to work.

## The decision

1. **Repair it.** The hybrid gets warm guesses again. **This can move
   `test/baseline/zlvmit.npz`**: a different seed is a different iteration
   path, and §12 makes that file ground truth. Needs a measure-then-regenerate
   gate of its own, and the accessors fixed as well as the order.
2. **Delete the block.** ~130 lines that provably never execute, plus the bare
   `except: pass` hiding them. The hardcoded fallback IS the code that runs, so
   deleting cannot move a number and `zlvmit.npz` proves it. This is the lazy
   answer and probably the right one.
3. **Leave and document.** `zlvmit` is §1-exempt legacy kept for published
   results; a dead branch inside it harms nobody. But a bare `except: pass`
   over a call that is now structurally wrong will mislead the next reader, and
   [ticket 49](49-nonconvergence-return.md)'s rule is that a failure is a
   return value, never a silence.

Whichever is chosen, the gate is **`zlvmit.npz` unmoved at rtol = 1e-10** for
options 2 and 3, and a measured diff for option 1.

**A wider question rides along**: are there other bare `except:` clauses in
`eos/zlvmit` hiding calls ticket 90 broke? The vMIT half of this same block is
one, so enumerate rather than fix the two that are named here.

## Resolution (2026-08-29) — option 2, deleted; the block was never a warm start

**The diagnosis in the Question is right but understated.** The failure is
not `par.m_p` on a float — it is **arity**, so control never enters the
function at all. Every one of the six calls is short exactly one required
positional argument:

| call | passes | signature today |
|---|---|---|
| `solve_pure_H_trapped` | 4 | `(par, n_B, Y_Le, flags, T)` |
| `solve_pure_H_fixed_yc` | 4 | `(par, n_B, Y_C, flags, T)` |
| `solve_pure_H_beta` | 3 | `(par, n_B, flags, T)` |
| `solve_pure_Q_trapped` | 4 | `(par, n_B, Y_Le, T, flags)` |
| `solve_pure_Q_fixed_yc` | 4 | `(par, n_B, Y_C, T, flags)` |
| `solve_pure_Q_beta` | 3 | `(par, n_B, T, flags)` |

`TypeError` at the call site, unconditionally, for every input. The bare
`except: pass` caught it and the hardcoded estimates below ran instead — 100%
of the time, since ticket 90.

**Repair was never a reorder.** Two facts found while ruling:

- `H.mu_p_H`, `H.n_p_H`, `H.mu_eG`, `Q.mu_u_Q`, `Q.n_u_Q`, `Q.mu_eG` exist
  nowhere in the package. Today's `EoSPoint` carries `mu_p`, `n_p`, `mu_e`.
- `mu_eG` is `eos/mixed`'s **global** electron potential (`mixed/solver.py:43`
  — "a global part, a single mu_eG neutralizing the average"). A **pure**-phase
  solve has never had one, in any generation of this code. `H.mu_eG` was
  therefore wrong before ticket 90 touched anything; the block was written
  against a struct that predates the current one and was already reaching for
  a mixed-phase quantity on a pure-phase result.
- `eos/zlvmit` constructs no `SpeciesFlags` anywhere, so repairing also meant
  choosing one on the legacy module's behalf.

So option 1 was not "fix six lines" but "author new seeding code, including a
value for a quantity the source object does not have, inside a §1-exempt
legacy module, and then measure it against a frozen baseline it could move."
§1 keeps `zlvmit` for its published results, and **the published results were
produced by the hardcoded estimates.** Option 2 taken.

### What changed

`eos/zlvmit/mixed_phase_eos.py`, one file:

- The `try:` block in `get_default_guess` — 143 lines, both the hadronic and
  the quark half — and its `except: pass` are gone. The hardcoded estimates
  are now the function's only path, which is what they have been in fact.
  A six-line comment in their place records why the seed was removed.
- The docstring no longer claims "For χ=0 (onset): Use pure H solution to get
  μ's / For χ=1 (offset): Use pure Q solution" — it described only the deleted
  block. It now says what the function does and notes that `chi` and `Y_L`
  select nothing (both are still in the signature; changing it is call-site
  churn this ticket did not ask for).
- "Fallback to hardcoded values" -> "Hardcoded values": there is nothing left
  to fall back from.
- The six `solve_pure_*` aliases (`from eos.zl.solver import ...` entirely,
  and three names off the `eos.vmit.solver` import) are gone with their only
  caller. `zl_thermo_from_mu_n` / `vmit_thermo_from_mu_n` stay, so zlvmit
  still imports both models and no §1 statement moves.

No `except:` clause was narrowed, because none was left to narrow.

### Gate: measured, not asserted

The `zlvmit.npz` comparison is the weaker check here, because the file is a
py3.14 artifact and this session's default interpreter is anaconda 3.9.7,
where `zlvmit` already drifts at ~1e-10 (the drift the map records at line
2069). Both were run:

- **Bitwise regeneration, py3.9.7** — `case_zlvmit()` run with the edit and
  with it stashed, same interpreter, arrays compared with `np.array_equal`:
  **540 of 540 keys identical, zero bit differences.** This is the real proof
  and it is interpreter-free: deleting code that never executed cannot move a
  number, and the measurement says so rather than the argument.
- **`test/baseline/test_baseline.py -k zlvmit`, py3.14** (the stack the
  baselines were frozen on): **passed**. `zlvmit.npz` unmoved, md5
  `bcb6dc67be27b9e254a3bf2228106358`, untouched.
- The same test on py3.9.7 fails — **and fails identically with the edit
  stashed**, the control run first. Pre-existing interpreter drift, not this
  diff.
- `test/test_imports.py` + the full `test/baseline` suite on py3.14:
  **235 passed**. `test/zlvmit/` is `.dat` fixtures, not tests.

### The wider question, enumerated

`eos/zlvmit` has **four** bare `except:` clauses. **Exactly one hid a
ticket-90 breakage** and it is the one deleted here. The other three are
point-skip guards, and none hides a broken call:

| site | what it wraps |
|---|---|
| `mixed_phase_eos.py:2939` (pre-edit) | `RegularGridInterpolator` lookups in the free-energy crossing scan; `continue` on a point outside the grid |
| `plot_results.py:407` | `eos_collection.get('chi', ...)`; appends `np.nan` so the point is not shaded |
| `plot_results.py:437` | `eos_collection.get(sp, ...)`; appends `np.nan` so the curve breaks |

Left alone: they are correct in intent, narrowing them is unasked-for churn in
a §1-exempt module, and the map's hard rule sends a noticed-but-unasked defect
to the report rather than the diff. Same ruling for `vmit_default_guess` and
`vmit_warm_start` (`mixed_phase_eos.py:102-103`), which are **already** dead —
one occurrence each, the import line — and were not made so by ticket 90 or by
this ticket.

### For the report

- Three bare `except:` clauses remain in `eos/zlvmit`, listed above.
- `vmit_default_guess` / `vmit_warm_start` are imported and never called.
- `get_default_guess`'s `chi` and `Y_L` parameters now select nothing.
- The `RuntimeWarning: invalid value encountered in scalar divide` at
  `eos/zl/thermodynamics.py:331` fires during the zlvmit baseline on both
  interpreters, before and after this diff.
