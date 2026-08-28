# SFHo's nuclear-matter derivatives should be analytic too, for
# reproducibility rather than for conditioning

Type: task
Status: open
Blocked by: -   (111 is the worked precedent)
Parent: ../map.md

## Question

Raised on [ticket 103](103-nmp-closures-four-models.md) (2026-08-29).
`sfho.nmp.compute_nmp` takes K_sat as a second finite difference of E/A, Q_sat
as a third, and K_sym as a second difference of `esym`, each of a quantity that
is itself the output of a nonlinear solve.
[Ticket 111](111-dd2-analytic-nmp-derivatives.md) did the same program for dd2
and found something that has nothing to do with conditioning:

> `test/baseline` documents a 0.351 MeV Q_sat shift between anaconda 3.9.7 /
> numpy 1.26.4 / scipy 1.13.1 and python.org 3.14.2 / numpy 2.3.5 / scipy
> 1.17.0 -- 2e-3 relative, all of it stencil roundoff. The four analytic values
> now agree across those same two stacks to 1.2e-13 .. 3.1e-16.

That argument transfers unchanged: SFHo's published Q_sat is a stencil number
and therefore an interpreter-dependent one. SFHo's own h-sweep puts the plateau
spread at 0.13 MeV on 467 (2.8e-4 relative), with the h = 5e-5 end 2.5 MeV away.

**This does NOT make Q_sat imposable in sfho, and the ticket must not claim it
will.** 103 measured why: sfho's only candidate fifth isoscalar knob is `c3`,
whose Jacobian column is 550x weaker than `g_sigma`'s, and admitting it drops
the closure's `sigma_min` from 0.051 to 0.0063. That is a statement about what
saturation constrains -- a high-density omega^4 term is not a saturation
observable -- and no derivative accuracy touches it. The default four-row
closure is already h-exact, so nothing shipped waits on this ticket either.

SFHo should be easier than dd2 was: its couplings are constants, so there is no
density-dependent coupling to differentiate and no rearrangement term. The
program is 111's -- differentiate the gap equation implicitly for the sigma
field's density derivatives, substitute into the closed-form isoscalar E/A --
with `E_sym`'s closed form (Steiner et al., Phys. Rept. 411 (2005) Eq. 20)
giving L_sym and K_sym directly.

**Forward and inverse go analytic together.** 111's constraint holds here for
the same reason: the finite-difference bias cancels on a round trip only while
both sides difference identically.

## Gate

Analytic values inside the (h, h/2) Richardson pair's own scatter; the four
values identical across two interpreters where the stencils differ; the
h-exact keys (n_sat, E_sat, m*/m, E_sym, P_sat) bit-identical; `test/baseline`
`sfho.npz` passing, with every moved published number named old -> new; sfho
`run_full_check` PASS and `test/sfho` green. A `verify/` entry mirroring dd2's
`analytic NMP derivatives`.
