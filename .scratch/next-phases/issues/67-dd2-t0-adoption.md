# dd2 cannot adopt the shared T = 0 door without re-freezing three NMP entries

Type: grilling
Status: open
Blocked by: - (74 resolved)
Parent: ../map.md

## Question

[Ticket 52](52-general-t0-integrals.md) opened the door §7 said dd2 had not
found: `eos.general.fermi_integrals.solve_fermi_t0` is public and changes no
number. The other half of finding 24 — **dd2 deleting its own four T = 0
functions and importing that door** — was implemented, MEASURED, and reverted,
because it fails the gate 52 set for it. This ticket is the ruling on what
happens next; the work is already written and the evidence is below.

### What the adoption is

`eos/dd2/thermodynamics.py:66-94` (`number_density_t0`, `scalar_density_t0`,
`eps_kin_t0`, `P_kin_t0`) delete, and `kinetic_thermo` becomes

    if T == 0.0:
        n, P, e, s, ns = solve_fermi_t0(mu_eff, m, g, False)
    else:
        n, P, e, s, ns = solve_fermi_jel(mu_eff, T, m, g)
    return n * hc3, P * hc3, e * hc3, s * hc3, ns * hc3

`include_antiparticles=False` reproduces dd2's threshold convention exactly, and
the massless branch dd2 wrote by hand falls out of the shared forms (the m^4 log
terms carry a factor m). The algebra is identical row for row. The FLOATING
POINT is not, in three ways that cannot be removed without re-implementing the
formulas somewhere: the shared version is `@njit(fastmath=True)` where dd2's is
plain NumPy; it evaluates the polynomials at `mu` where dd2 round-trips through
`EF = sqrt(kF^2 + m^2)`; and it returns fm-based units, so dd2 multiplies back by
`hc3` what the shared code divided by it.

### What that costs, measured

Isolated-copy pair on anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1, one copy
carrying the adoption and one the same tree without it (the live checkout could
not be used: a concurrent session was editing `dd2/api.py` mid-run, which is
what a HEAD control beside the change is for). All 4692 quantities of
`test/baseline`'s dd2 case, recomputed in both:

| | |
|---|---|
| bit-identical | **3434 of 4692** |
| every EoS quantity (n, P, eps, s, mu_i, fields, m_eff) | <= **3.5e-12 abs**, <= **5.9e-15 rel** |
| `nmp.n_sat`, `E_sat`, `E_sym`, `m_eff_ratio` | ~1e-14 rel |
| `nmp.L_sym` (first derivative) | 1.4e-12 rel |
| `nmp.K_sat`, `nmp.K_sym` (second) | **1.9e-8**, **2.2e-8** rel |
| `nmp.Q_sat` (third) | **3.6e-4** rel — 0.061 MeV on 168.65 |

`eos/dd2/verify/run_full_check.py` stays PASS with the golden SNM(0.16) point at
1.40e-05 and CompOSE HS(DD2) at 2.83e-05, both unmoved; backend parity moves
8.88e-16 -> 6.44e-14, still fourteen orders inside its gate.

So the physics is a no-op and **the finite-difference NMP map is not**: each
derivative order multiplies last-bit noise by another factor of ~1/h, and by the
third derivative that is 1e6 to 1e8. Two failures follow, both in that map and
nowhere else:

- `test/baseline/test_baseline.py::test_baseline[dd2]` — the three entries above,
  against `rtol = 1e-10`.
- `test/dd2/test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion` —
  inverting to (K_sat, Q_sat) = (240, 300) lands at isoscalar residual 3.36e-02
  against the 2e-02 gate after 32 restarts, where today it converges. The test
  asserts a solver VERDICT (`restarted.ok`) about a target that sits near the
  edge of what the closure can realise; a perturbation at the last bit flips it.

### The decision

1. **Does dd2 adopt it at all?** The alternative is that finding 24 stands
   deferred: dd2 keeps four functions §7 says it may not have, and the ledger
   says why. Note what is NOT available — no reformulation makes the adoption
   bit-exact, so "try harder" is not one of the options.
2. **If yes, the three NMP entries re-freeze.** That is a §12 act and belongs
   with [ticket 62](62-regenerate-baselines-py314.md), which regenerates every
   `.npz` on the canonical stack and whose stop condition is exactly "anything
   larger than round-off stops the regeneration and is reported". This change is
   round-off AMPLIFIED, which is a third category 62 does not name and should.
3. **And `test_dd2_m8` needs a ruling of its own** — regeneration does not touch
   it. Either the target moves off the knife edge, or the assertion stops being
   about `ok` and starts being about the residual falling by orders of magnitude
   between the single seed and the restarts, which is what the test's own
   docstring says it is guarding.
4. **Worth asking while the numbers are in front of us:** `nmp.Q_sat` is
   reproducible to about four digits under any last-bit perturbation of the gas.
   Freezing it at `rtol = 1e-10` pins noise, and any future refactor that touches
   a dd2 kernel will fail on it the same way. Whether the baseline should carry a
   per-key gate for the FD curvatures, or the NMP map should compute them from a
   Richardson extrapolation instead, is the durable half of this ticket —
   [ticket 47](47-dd2-nmp-inversion.md) already found this floor from the other
   side.

The reverted diff is reproducible from this ticket in ten minutes; nothing is
lost by ruling first.

## Update from [ticket 62](62-regenerate-baselines-py314.md) (not a resolution)

**The vehicle this ticket was waiting for has departed, and two of its premises
moved with it.**

1. **"The three NMP entries re-freeze ... belongs with ticket 62, which
   regenerates every `.npz`" is now stale.** Ticket 62 has landed. `dd2.npz` is
   re-frozen on python.org 3.14.2, and the three entries this ticket names —
   `nmp.K_sat`, `nmp.K_sym`, `nmp.Q_sat` — are **exactly and only** the three
   keys that moved in that regeneration. All 4689 other `dd2` keys are
   bit-identical across the two stacks. A re-freeze for the T = 0 adoption is
   therefore its own §12 act now, not a rider on someone else's.

2. **The scale this ticket quotes has a sibling that is five times larger.**
   This ticket measures the adoption moving `nmp.Q_sat` by 3.6e-04 rel
   (0.061 MeV on 168.65). The stack change alone moved the same quantity by
   2.08e-03 rel (**0.351 MeV**, 168.65 -> 169.00), with `K_sat` at 1.2e-07 and
   `K_sym` at 1.4e-08 — the same three entries, the same ordering by
   derivative order, one order of magnitude apart in cause. Point 4 of this
   ticket ("`nmp.Q_sat` is a third finite difference ... whether the baseline
   should carry it at all") is the question both measurements are asking, and
   it now has two independent witnesses instead of one.

3. **Now blocked by [ticket 74](74-py314-non-baseline-failures.md)**, which
   re-derives `test_api.py`'s `abs=0.2` on Q_sat from a noise floor measured on
   the canonical stack. Deciding whether a 0.061 MeV adoption shift is
   acceptable requires knowing what the honest floor is, and today's `abs=0.2`
   is a number ticket 47 showed asserts below the stencil noise. Answering this
   ticket first would be measuring against a tolerance already known to be
   wrong.

## Unblocked by [ticket 74](74-py314-non-baseline-failures.md)

**The honest floor this ticket was waiting for: Q_sat carries 0.25 MeV of
stencil excursion at the shipped h = 1e-4**, measured by sweeping h in the
forward and inverse maps together per `nmp.py:85` — the [2e-4, 1e-3] plateau
has a spread of 0.088 MeV and the shipped h sits 0.2508 MeV off its mean. Two
independent evaluations of Q_sat therefore differ by up to ~0.5 MeV.

Against that floor, all three witnesses of point 4 fall inside it:

    this ticket's T = 0 adoption      0.061 MeV   a quarter of the floor
    the 3.9 -> 3.14 stack change      0.351 MeV   inside the 2-evaluation spread
    published table vs re-derived     0.207 MeV   h-STABLE, so NOT noise

So **question 4 has its measured answer: no**, `nmp.Q_sat` does not belong in a
frozen `rtol = 1e-10` baseline — a 0.061 MeV adoption shift is not
distinguishable from the stencil, and pinning it at ten digits pins noise.
Note the third row is the odd one out and worth keeping separate: it is flat to
three digits across h in [3e-4, 2e-3], so it is a genuine difference between two
parametrizations (the published table's 6-decimal coefficients against the
re-derived ones), not something a Richardson extrapolation would remove.

**Point 3 is discharged, not merely unblocked.** `test_dd2_m8`'s target moved to
(220, 300), which is seed-limited on BOTH stacks, and the test now asserts the
residual falls by at least x100 alongside the `ok` verdict — which is the
"stops being about `ok` and starts being about the residual falling by orders of
magnitude" this ticket asked for.

**Point 2's premise is confirmed stale**: the re-freeze vehicle has departed and
ticket 74 did not restore it. `h` was deliberately NOT moved, for the same
reason — both maps together plus a `dd2.npz` re-freeze is a §12 act neither
ticket authorises. So this ticket's decision 1 (does dd2 adopt the shared T = 0
door at all?) is now the only open half, and it is a ruling, not a measurement.

**One piece of this ticket's own evidence went stale with the target.** The
measurement above — "inverting to (K_sat, Q_sat) = (240, 300) lands at
isoscalar residual 3.36e-02 against the 2e-02 gate after 32 restarts" — was
taken against a target `test_dd2_m8` no longer uses, and on the 3.9 stack where
(240, 300) was still seed-limited. If the adoption is revisited, that row must
be re-measured at (220, 300) on the canonical stack before it counts as a cost:
the second failure this ticket attributes to the adoption may not survive the
change of target.
