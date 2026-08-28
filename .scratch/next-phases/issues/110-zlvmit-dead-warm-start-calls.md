# `zlvmit`'s pure-phase warm-start calls have been dead since ticket 90

Type: grilling
Status: open
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
