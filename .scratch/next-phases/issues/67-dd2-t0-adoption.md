# dd2 cannot adopt the shared T = 0 door without re-freezing three NMP entries

Type: grilling
Status: resolved
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

## Ruling (session 4035eac1)

**dd2 ADOPTS the shared T = 0 door.** The ticket's title states a cost that the
measurements do not support: re-freezing three NMP entries is forced only by a
gate that asserts below the stencil, and the second failure is not a cost of the
adoption at all.

All numbers below were taken on python.org 3.14.2 in an isolated control /
adoption PAIR built from `git archive HEAD` (the live checkout carried another
session's `general/` edits). The adoption is a reconstruction of the reverted
diff, not the original: four functions deleted, `kinetic_thermo`'s T = 0 branch
becomes `solve_fermi_t0(mu_eff, m, g, False)`. Its forward-map Q_sat shift is
5.1e-4 rel against the 3.6e-4 this ticket recorded on 3.9 -- consistent, and
the difference is the stack.

### Decision 1: adopt. The forward-map cost is below every stencil floor

`h`-sweep of `compute_nmp` over h in [5e-5, 3e-3], control vs adoption. Each
key's floor is its shipped-h (1e-4) offset from the [2e-4, 1e-3] plateau mean:

    key            adoption shift    stencil floor    margin
    n_sat, E_sat, m_eff_ratio, E_sym
                   ~1e-14            EXACTLY 0        h-independent
    L_sym          1.4e-12           3.0e-6           2e6 below
    K_sat          1.9e-8            3.1e-6           166x below
    K_sym          3.1e-9            7.7e-6           2500x below
    Q_sat          5.1e-4            1.5e-3           3x below

All three keys this ticket names shift by less than their own stencil noise.
Point 4 asked the question for Q_sat alone; executed for all three, the answer
is the same for all three. The four remaining keys have exactly zero spread
across a 60x range of h and are honestly freezable at ten digits.

`L_sym` is a FOURTH key on the same 3e-6 floor. It passes rtol = 1e-10 today by
luck of magnitude and is armed for the next refactor that touches a dd2 kernel.

### Decision 2: the gate, not a re-freeze -- and the gate is independent

The baseline stores the FORWARD map, so the honest partition is not
imposed-vs-predicted (that is the inverse map's semantics) but h-exact vs
h-sensitive, which the sweep draws with no judgement call:

    rtol = 1e-10   n_sat, E_sat, m_eff_ratio, E_sym      (zero spread)
    rtol = 1e-5    K_sat, L_sym, K_sym                   (floors 3.0-7.7e-6)
    rtol = 3e-3    Q_sat                                 (floor 1.5e-3)

as a per-key dict beside RTOL/ATOL in `test/baseline/test_baseline.py`. Dropping
the keys instead would let a genuine 10% K_sat regression pass unnoticed; a 1e-5
gate still catches that with four orders to spare. This change stands on HEAD
today, with or without the adoption, and belongs in its own commit ahead of it.

**Consequence: the adoption needs NO re-freeze.** Under these gates every dd2
baseline key passes -- the three NMP entries by the margins above, the other
4689 at <= 5.9e-15 rel. The title's premise resolves to zero .npz acts. (A real
`test_baseline[dd2]` run is the confirmation step before landing; test/ is
gitignored and absent from the isolated pair.)

### Decision 3: `test_dd2_m8` is a lottery ticket, not a cost

Re-measured at the CURRENT target (220, 300) on the canonical stack, as this
ticket required. The failure survives, and worse than recorded:

    control     single 0.611 (miss)   32 restarts -> 6.8e-4 (pass)   drop 895x
    adoption    single 0.434 (miss)   32 restarts -> 5.5e-2 (MISS)   drop 7.9x

Deeper search does not recover it (64 -> 4.6e-2, 128 -> 2.9e-2), so it is not a
seed lottery a larger N_RESTARTS fixes. `fastmath` is exonerated: the
JIT-DISABLED adoption stalls identically, so the cause is the closed forms'
arithmetic grouping and no compiler flag removes it.

But the decisive measurement is on the 5x5 solve at DD2's OWN NMPs, with the
target perturbed by a relative eps and NO code change:

    eps      control                  adoption
    0        converged 6.7e-11        stuck at seed 2.2007e-3
    1e-14    stuck at seed 2.2007e-3  stuck at seed 2.2007e-3
    1e-13    stuck at seed 2.2007e-3  converged 2.3e-8
    1e-12    converged 2.3e-8         stuck at seed 2.2007e-3
    1e-10    2.0e-3 (partial)         converged 2.4e-10

"Stuck at seed" is literal: every recovered coupling returns moved_rel = 0.0,
bit-for-bit the published seed, with ok = True because ISO_GATE = 2e-2 admits
the 2.2007e-3 cross-row violation -- the failure mode `nmp.py:70-78` documents
and warns `ok` cannot detect. HEAD is ALREADY a coin flip: nudge the target in
its fourteenth digit and today's code stops leaving the seed. The adoption does
not cause this; it re-rolls the same dice.

So the test asserts the outcome of a last-bit lottery.

**Re-targeting was attempted and is NOT available.** (K_sat, Q_sat) over
K_sat in {210, 220, 230, 240} and Q_sat in {300, 350}, each at eps in
{0, 1e-13, 1e-11} and under both gas kernels -- 48 configurations. **Not one
of the eight targets holds its verdict across its own six.** The best,
(220, 300) and (230, 300), pass at eps = 0 and 1e-11 on control and fail
everywhere else. The 32-restart landing residual scatters between 3e-5 and
0.15 with no relation to eps at any target. There is no knife-edge-free point
in this grid, and the scatter says the surface has none to find: the lottery is
the whole plane sampled, not one unlucky target.

What IS stable, measured across twelve configurations (eps in
{0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10}, both kernels): the **5x5 default
closure at K_sat = 220** -- ok = True and residual under ISO_GATE in all
twelve, residuals 4.6e-9 to 6.5e-3 against a 2e-2 gate. That is the
assertion `test_dd2_m8` can honestly carry.

Cost of the swap, stated plainly: `N_RESTARTS` then has no test. The sweep
above is why that is correct rather than a loss -- restart coverage today is
coverage of a coin flip. It is rebuilt in the spin-out ticket below as the
statistical property `nmp.py:84-96` already records (7/68/115 of 187 cells at
0/32/64 restarts), which is a count over a grid and therefore stable where any
single cell is not.

### Decision 4: answered above, and it has a sibling

Point 4 ("does nmp.Q_sat belong in a frozen rtol = 1e-10 baseline") -- no, and
neither do K_sat, K_sym or L_sym. Decision 2 is that answer as a gate.

**A new finding, larger than this ticket, spun out:** `invert_nmp`'s outcome at
DD2's own NMPs is decided in the fourteenth digit; `InversionStatus.ok` cannot
distinguish converged from never-moved; `test_inversion_is_deterministic` gives
false comfort (two stuck runs agree perfectly); and the converged branch lands
on couplings 3.9% from DD2's own that reproduce the same six NMPs, so the 5x5
closure is not injective at DD2's own point. None of that is the T = 0 door's
doing. Its own ticket --
[ticket 93](93-invert-nmp-basin-lottery.md) -- cross-referenced to
[ticket 47](47-dd2-nmp-inversion.md), which found this floor from the other
side; the fix candidates are a seed comparison inside `ok`, or a residual gate
that separates stuck from converged.


## Landed (session 4035eac1)

`eos/dd2/thermodynamics.py` — the four T = 0 functions deleted,
`kinetic_thermo` calls `solve_fermi_t0(mu_eff, m, g, False)`. dd2 was the last
model carrying its own T = 0 Fermi integrals; sfho, zl and did already reached
the shared door through `solve_fermi_jel`'s T < 1e-4 dispatch. The massless
species must NOT be routed through `solve_fermi_jel` -- its m < 1e-5 branch is
tested BEFORE its T < 1e-4 branch and returns the net particle-antiparticle
density, negative where mu_eff < 0, which is exactly where dd2's neutrinos sit.
`solve_fermi_t0` handles m = 0 correctly, so the direct call is the shape.

`test/baseline/test_baseline.py` — `MEASURED_RTOL`, the per-key gate, in two
families. The four finite-difference NMP keys at their stencil floors, and the
TOV sequence keys at the integrator's.

**`test_baseline[dd2]` passes with no re-freeze**, which is the answer to this
ticket's title. `test_baseline[tov]` did NOT, and that is a cost this ticket
never anticipated: dd2's EoS quantities are stable to 5.9e-15, but a TOV
sequence integrated over that table moves 1.24e-07, with M_max 2.57e-09.
Attributed in the isolated pair -- control passes, adoption fails with those
exact numbers -- so it is the adoption's, not the concurrent session's.

It is the integrator's own floor, and measured the same way as the stencil:
perturbing the EoS table by a relative 1e-15 moves the sequence 1.22e-07 and
M_max 2.87e-09, and at 1e-12 it is 1.26e-07 and 8.05e-10 -- a PLATEAU, already
saturated at 1e-15 rather than growing with the perturbation, which is what
identifies adaptive-step placement rather than propagated error. So the ruling
is the same as for the NMP keys and for the same reason: gate at the measured
floor rather than re-freeze, because a re-freeze pins the noise and fails again
on the next kernel change. `dd2`, `vmit` and `mixed` sequence/M_max keys all
carry it -- vmit's do not move today, and are listed for the reason `L_sym`
was.

`test/dd2/test_dd2_m8.py` — the (220, 300) restart test replaced by the 5x5 at
K_sat = 220, measured stable across twelve configurations; Q_sat dropped from
the round-trip's asserted keys, being a prediction whose two evaluations differ
by ~0.5 MeV (it read 0.515 and the gate was 0.5).

`test/dd2/test_api.py` — the 6x6 test narrowed to the routing claim it can
prove. `impose_Q_sat=True` is now recorded in `docs/DEFERRED.md` as available
but not certified.

`test/tov/test_solver_fast_robustness.py` — the soft Delta-rich parametrization
frozen as `_SOFT_DELTA_RICH` data. These three tests are about the crust join
and fast/scipy agreement, and were reaching their EoS through an NMP inversion,
which made a chaotic solve a prerequisite for a TOV regression net. Ticket 74
had already re-tuned that sample once for this reason.

Suite: **500 passed, 15 skipped, 0 failed** over test/dd2, test/tov, test/mixed
and test_baseline[dd2]. `eos/dd2/verify` PASS with the golden SNM(0.16) point at
1.40e-05 and CompOSE HS(DD2) at 2.83e-05, both UNMOVED; backend parity
8.88e-16 -> 5.51e-14, fourteen orders inside its gate. `eos/general/verify` PASS.

**What this ticket got wrong along the way, recorded because the next reader
will be tempted by the same two steps.** "The cost is not real" was stated once
the forward-map measurements came in and was wrong: four tests failed, all
through the inversion lottery, and three of them were outside dd2. And
re-targeting `test_dd2_m8` -- this ticket's own point 3, and the obvious repair
-- is not available: no target in the sampled plane holds its verdict. Both
errors came from generalising a measurement past what it covered.

## Landed on `main`

Committed with explicit pathspecs on `3781907`. Gate, python.org 3.14.2 /
numpy 2.3.5 / scipy 1.17.0, every run solo:

    with the adoption        1734 passed, 20 skipped, 0 failed  (19:25)
    without it (same HEAD)   1734 passed, 20 skipped, 0 failed  (20:58)

Byte-identical counts, so **the adoption changes no test outcome**. Against
`output/_audit/pytest_after_ticket74_py314.txt` (1 failed, 1680 passed,
15 skipped) that is 0 added failures — one FEWER, the `enjl` node closed by
`f8ccc33`. `eos/dd2/verify` PASS with the golden SNM(0.16) point at 1.40e-05
and CompOSE HS(DD2) at 2.83e-05, **both unmoved**, backend parity 5.51e-14;
`eos/general/verify` PASS.

**Decisions 2 and 3 never became commits: `test/` is gitignored**
(`.gitignore:75:/test/`, and CLAUDE.md §11 says so deliberately). The
`MEASURED_RTOL` dict and the `test_dd2_m8` swap are landed in the working tree
only. Decision 2's claim that it "stands on HEAD today, with or without the
adoption" was therefore gated as a run rather than a commit, and it holds:
`test/baseline` is **20 passed, 0 failed** on HEAD with the adoption absent.
Whether `test/` should be tracked at all is a §11 question this ticket does not
own; it belongs to the Stage 7 report.

**Two premises moved under this ticket while it was landing**, both from a
concurrent session resolving [ticket 89](89-dd2-honours-species-flags.md):

1. `eos/dd2/solver.py` now honours `flags.photons`, and `test/baseline/dd2.npz`
   was **re-frozen** at 10:57 on 2026-08-27 — 162 of 4692 keys, all `P`, `eps`
   and `s` at 54 points. No NMP key moved, so decision 2's floors are
   unaffected, and the re-freeze happened while this adoption was stashed,
   which makes the new file a clean pre-adoption baseline to measure against.
   Both gates above are on that file.
2. The first full-suite gate for this landing straddled that regeneration and
   was **discarded rather than reported**, the same call ticket 82 made for the
   same reason.

The spin-out is [ticket 93](93-invert-nmp-basin-lottery.md), renamed from 88
during this landing because two tickets shared that number; the map,
`docs/DEFERRED.md` and [ticket 81](81-second-default-solver-kwargs.md) all
point at the new one.
