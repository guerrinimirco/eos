# Should the CFL phase carry a free thermal gluon gas at all?

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

Raised by [ticket 82](82-alphabag-gluons-default.md) and left out of its
ruling, which was about a DEFAULT rather than about which sectors the paired
phase has.

`eos/alphabag/solver.py::solve_cfl` adds the full free-gas gluon term whenever
`include_gluons` is set:

    if include_gluons:
        gluon = gluon_thermo(T, alpha)
        P_total += gluon.P; e_total += gluon.e; s_total += gluon.s

`gluon_thermo` is 16 massless bosons at mu = 0 with the factor
1 - 15 alpha_s/(4 pi) — the UNPAIRED phase's gluon gas.

**The model's own document says that is not the paired phase's physics.**
`eos/alphabag/alphabag.md` (and `.tex` at the same place):

> The gluon term is not part of (Pcfl). In the CFL phase the gluons are all
> massive through the Meissner effect and their thermal population is
> suppressed; the sector remains available as a flag at the solver level and
> is added to the totals there, as it is in the unpaired phase, but it is not
> inside the phase's own potential.

So the behaviour is *declared*, not silent — but "available as a flag" is a
weak defence for a term the same paragraph calls suppressed. And the two
locked-phase models disagree about it: `eos/abpr/species.py` REFUSES `gluons`
and gives this exact physics as the reason —

> "a thermal sector, identically zero at T = 0, and in the CFL phase the
> gluons are Meissner-massive besides"

— while `alphabag`'s `cfl` mode adds the unsuppressed gas on request.

**What ticket 82 already changed.** `gluons` now defaults `False`, so no `cfl`
call inherits the term; a caller who gets it has named it. That closes the
silent half and is why this is not urgent.

**Candidate answers.**

1. **Suppress it properly** — give the CFL branch a Meissner-massive gluon
   thermodynamics (a mass gap in the Bose integral) rather than the massless
   gas. Correct physics, most work, and needs a source for the masses.
2. **Refuse it in `cfl`** — `solve_cfl` raises on `include_gluons=True`, the
   way `abpr` refuses the flag outright. Cheapest honest answer, and it makes
   the two locked-phase models agree.
3. **Keep it and say so louder** — the document already declares it; leave the
   physics and treat the flag as the caller's explicit responsibility.

**Cost is measurable and small.** The two places the notebooks build a CFL
table (`notebooks/quark_eos.py:745` and `:1136`) are both at T = 0, where the
gluon gas is identically zero, so no shipped figure depends on the answer.
`test/baseline/case_alphabag`'s `cfl.*` rows are also T = 0. At T = 30 the term
is P = 0.1187 MeV/fm^3 against a CFL pressure of 173.0.

Recommend 2 unless a Meissner-mass source is to hand, on the ground that a
sector the document calls suppressed should not be reachable in the closed
form that assumes it is not.
