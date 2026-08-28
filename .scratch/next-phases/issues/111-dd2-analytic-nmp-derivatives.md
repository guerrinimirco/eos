# The DD2 nuclear-matter derivatives should be analytic, and Q_sat cannot be
# imposed until they are

Type: task
Status: resolved (2026-08-28)
Blocked by: -   (105 is resolved; this is the half it deferred)
Parent: ../map.md

## Question

Ruled by the user on [ticket 105](105-dd2-isoscalar-conditioning.md)
(2026-08-28), Q4: **"all the relations that can be written analytically
should."** This is that program for `dd2/nmp.py`, and it is what stands
between the Q_sat-imposing closure and being usable.

`compute_nmp` takes K_sat as a second finite difference of E/A, Q_sat as a
third and K_sym as a second difference of E_sym, each of a quantity that is
itself the output of a nonlinear solve. The isoscalar E/A of symmetric matter
at T = 0 is a closed form in the couplings and the self-consistent sigma
field, so the derivatives can be taken by hand: differentiate the gap equation
implicitly for dsigma/dn, d2sigma/dn2, d3sigma/dn3 and substitute.

### Why it is worth doing, measured on ticket 105

    Q_sat spans 2.48 MeV over h in [5e-5, 5e-4]   (relative floor ~1.5e-3)
    K_sat spans 5.2e-04 MeV over the same range
    Z_sat spans 4.8e+04 on a value of 4547 -- the fourth difference is noise

The Q_sat-imposing closure conditions at 259, so it inherits
259 x 1.5e-3 = **0.39 of relative coupling error**. That is the entire reason
it is not the shipped closure, and 105 measured that no choice of pinned
coefficient and no reparametrisation reduces it materially (the best pin is
259 against 354 / 703 / 4191; the collinearity is a rank statement, not a
coordinate one). **Killing the floor is the only remedy that works**, and with
it gone the amplification is harmless at any pin.

### What it must NOT break

- **Forward and inverse go analytic TOGETHER.** `nmp.py`'s docstring is
  explicit: the finite-difference bias cancels on a round trip only while both
  sides difference identically. Changing one and not the other stops the
  inversion reproducing its own inputs.
- The **default closure does not wait on this** — its rows are P, E_sat, m*/m
  and K_sat, and K_sat's 5.2e-04 MeV spread is already at the h-exact floor.
  So this ticket is not on the critical path of anything shipped.
- `compute_nmp`'s Q_sat at the published couplings will MOVE, by up to the
  stencil noise it currently carries. The user ruled this "a number being
  corrected", not a number moving (105, Q3). `test/dd2/test_dd2_m1.py:73`
  pins Q_sat at `abs=0.5`, which a correction of that size passes; check the
  rest of the pins before assuming it is the only one.

### Also in scope, because it is the same measurement

`ISO_GATE = 2e-2` was set wide to clear two scales, and 105 retired both: the
cross row's 2.2e-3 and Q_sat's stencil in a closure that no longer imposes it.
Measured on a 105-cell (K_sat, m*/m) grid with restarts on, the 101 passing
cells split **95 below 1e-5 and 6 in [1e-3, 2e-2], with nothing in between**,
so six are certified `ok` without being roots. Ticket 93 ruled the gate could
not be tightened, and that ruling rested on the cross row making accurate
solves land at 1.9e-3 — a premise 105 removed. Retune it here, on a scan over
more than two axes.

## Gate

`compute_nmp` and `invert_nmp` agree with their own stencils to the stencil's
accuracy before the change and to ~1e-10 after; the Q_sat-imposing closure
recovers a perturbed target's Q_sat to better than 0.01 MeV; dd2
`run_full_check` PASS with golden SNM(0.16) and CompOSE HS(DD2) unmoved;
`test/dd2` green; every moved published number named with its old and new
value.


## Resolution (2026-08-28)

**Done. Every derivative in `dd2/nmp.py` is analytic, and the Q_sat-imposing
closure works.** The program the ticket set out was executed as written:
differentiate the gap equation implicitly, substitute into the closed-form
isoscalar E/A, replace the finite differences.

### The derivation, and the shortcut that made it a second derivative

Writing `S = Gamma_sigma(n) sigma` (so `m* = m_N - S`),
`G = Gamma_sigma^2/m_sigma^2` and `W = Gamma_omega^2/m_omega^2`, the gap
equation is `S = G(n) n_s(m_N - S, kF(n))` and, with omega_0 eliminated,

    eps = eps_kin(m*, kF) + S^2/(2G) + W n^2/2
    mu  = E_F* + W n + W' n^2/2 - G' n_s^2/2      <- Sigma^R, in these variables

Since `P = mu n - eps` at T = 0, `(E/A)' = P/n^2` and `P' = n mu'`, so at the
P = 0 point

    K_sat = 9 n mu'          Q_sat = 27 n (n mu'' - 3 mu')

**A third derivative of E/A is only a SECOND derivative of mu**, which is why
the gap needed differentiating twice and not three times. What that costs is
`S'`, `S''` from the implicit gap plus the (m*, kF) partials of `n_s` to
second order; the two non-elementary ones,
`int k^4/E^3` and `int k^4/E^5`, close under `k = m* sinh t` as
`int (cosh^2 t - 2 + sech^2 t) dt` and `int tanh^4 t dt`.
`L_sym`/`K_sym` came free from the same `E_F*` derivatives.

`snm_derivatives(par, n_B)` is the one home; `compute_nmp` and
`_isoscalar_quantities` both call it, so forward and inverse went analytic
together as required. It solves symmetric matter ONCE instead of seven times,
so the Q_sat closure also got cheaper.

### Gate, line by line

- **Analytic vs the h-plateau, not vs h = 1e-4.** Held. Against a Richardson
  extrapolation of (h, h/2) at h = 6e-4 the analytic values sit at 3.1e-8
  (K_sat), 6.1e-6 (Q_sat), 2.2e-12 (L_sym) and 3.9e-10 (K_sym) relative —
  every one far inside the stencil's own scatter. The shipped h = 1e-4 was
  0.073 MeV off on Q_sat, which is the whole correction. **A stronger witness
  than the plateau, and unexpected: the analytic values are the SAME NUMBER on
  two interpreters.** `test/baseline` documents a 0.351 MeV Q_sat shift between
  anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1 and python.org 3.14.2 / numpy
  2.3.5 / scipy 1.17.0 — 2e-3 relative, all of it stencil roundoff. The four
  analytic values now agree across those same two stacks to **1.2e-13 (K_sat),
  7.5e-13 (Q_sat), 8.8e-15 (L_sym), 3.1e-16 (K_sym)**. Ten orders.
- **The four h-exact NMPs unchanged at machine precision.** Held, and stronger:
  `n_sat`, `E_sat`, `m*/m`, `E_sym` and `P_sat` are **BIT-IDENTICAL** to what
  `git show HEAD:eos/dd2/nmp.py` returns on the same interpreter. Their code
  paths were not touched.
- **`test/baseline/dd2.npz` does not move.** Held — `test_baseline[dd2]`
  passes against the frozen file. The four corrections are 1.7e-7, 4.4e-4,
  9.1e-8 and 2.4e-7 relative, against gates of 1e-5, 3e-3, 1e-5 and 1e-5.
- **Z_sat reported as unavailable.** Held: `compute_nmp` returns no `Z_sat`
  key and both the module and `dd2.tex` now say why (a third derivative of the
  gap, spent on a quantity no closure imposes and nobody quotes).
- **The Q_sat-imposing closure recovers a perturbed target to better than 0.01
  MeV.** Held by eight orders. At DD2's own point it went from
  `max|residual| = 1.4e-02` with Q_sat imposed to 1.6 MeV, to **1.5e-12 with
  Q_sat imposed to 1.1e-10 MeV**. Perturbed: dQ_sat +20/-30/+100 and dK_sat
  +10/-20 all return Q_sat to |dQ| < 6e-10 MeV.
- **Forward/inverse agree to ~1e-10 after.** Held: the round trip returns the
  six imposed values to 2.7e-11 worst (K_sat), and `status.predictions` agrees
  with a forward map of its own couplings to 5.1e-11 (Q_sat) / 4.5e-13 (K_sym),
  where the old tolerances were 0.5 and 1e-2.
- **`run_full_check` PASS, golden SNM and CompOSE unmoved.** Held: golden
  `1.40e-05`, CompOSE HS(DD2) `2.83e-05` — the same two numbers ticket 105
  gated on.
- **`test/dd2` green.** Held.

### The moved published numbers, old -> new

    K_sat   242.7240553 -> 242.7240147     (rel 1.7e-07)
    Q_sat   168.7135236 -> 168.7868767     (rel 4.4e-04)
    L_sym    55.0336716 ->  55.0336666     (rel 9.1e-08)
    K_sym   -93.2240313 -> -93.2240089     (rel 2.4e-07)

`test/dd2/test_dd2_m1.py` pins Q_sat at `abs=0.5` against a golden 168.6; the
new value is 0.187 away and passes. No other pin was within reach of a move
this size. The one comment quoting a forward-mapped Q_sat of a frozen
coupling set (`test/tov/test_solver_fast_robustness.py`) was re-measured:
299.647016 -> 299.672302, couplings unchanged.

### ISO_GATE, the ticket's second half

**2e-2 -> 1e-8.** Re-measured on the four axes the isoscalar residual actually
has — 240 random targets over n_sat [0.140, 0.170] x E_sat [-17, -15] x
m*/m [0.45, 0.75] x K_sat [180, 320] — the separation is now nine orders:

    233 solves at or below 4.6e-12
      3 in [2.8e-3, 1.7e-2]      <- certified by the old gate, not roots
      4 above 2e-2

with **nothing in between**, and the same split at 0 and at 32 restarts, which
says the three were not seeds that could have been rescued. 1e-8 sits 3.5
orders above the worst genuine root and 5.5 below the lowest non-root, so it
is not a tuned number — anywhere in the gap does the same job. Ticket 93's
ruling that the gate could not be tightened rested on the cross row; 105
removed that premise and this ticket removed the other one.

### Two things the change did that the ticket did not predict

- **The stalls are gone.** `hybr` differences its own Jacobian from the
  residual, so the stencil's noise was making "not making good progress" a
  true report about the SURFACE rather than about the target. 105 measured 12
  of 18 misses as stalls; re-measured now, **0 of 7** on the four-axis grid and
  **0 of 8** on the (K_sat, m*/m) grid, at 0 and 32 restarts alike. Two scans
  are not a proof of absence, so `_stalled` and `STALL_RES` stay — what they
  defend against is a solver returning its input as an answer, which section 6
  forbids reporting silently whether or not it is currently reachable.
- **The verify suite's `restarts extend the basin` check lost its grid.** It
  ran on the Q_sat-imposing closure over (K_sat, Q_sat) precisely because the
  third difference made that the harder surface — 0/9 at zero restarts against
  4/9 at 32. That closure now reaches **30/30 at zero restarts** over a grid
  three times wider. The invariant is real and survives; it was re-measured
  onto the DEFAULT closure over (K_sat 160-320) x (m*/m 0.40-0.90),
  **22/30 -> 27/30**, which is where the basin structure now shows.

### Also landed

A new `verify/` entry, **`analytic NMP derivatives`**: each of the four must
sit within the (h, h/2) stencil pair's own scatter of their Richardson
extrapolation. Self-calibrating — the scatter is what the stencil says its own
accuracy is — so there is no tolerance to tune and none to loosen. **h = 1.5e-3
and that is the one measured choice in it**: the estimator is honest only where
truncation dominates the pair, and at small h the scatter it divides by is
roundoff, which for a third difference grows as h^-3 and makes the ratio jitter
with the interpreter — Q_sat's worst ratio runs 0.509/0.040/0.033/0.016/0.003
on 3.9 and 0.403/**0.633**/0.038/0.022/0.003 on 3.14 over
h = 4e-4 ... 1.5e-3. At 1.5e-3 both stacks pass with **300x** of margin
(0.003); at 6e-4 one of them would pass with 1.6x, which is a check waiting to
fail for a reason that is not physics.

### Gate

dd2 `run_full_check` **PASS on both interpreters** (golden SNM `1.40e-05`,
CompOSE HS(DD2) `2.83e-05`, both unmoved from 105, on each); `test/dd2` **211
passed** on anaconda 3.9.7 and on python.org 3.14.2; `test_baseline[dd2]` green
on both against the unmodified frozen file; and `test/dd2 + test/mixed +
test_imports + test_parameter_routes` **732 passed, 5 skipped, 3 xfailed**.
Every measurement in this resolution was taken on both stacks; where they
differ the ISO_GATE split is identical (233 roots ≤ 4.2e-12, three in
[2.8e-3, 1.7e-2], four above) and so is the basin count (22/30 -> 27/30).

**One cross-stack note, and it is not this ticket's.** On anaconda 3.9.7 /
numpy 1.26.4 / scipy 1.13.1, `test_baseline` fails for `ccdm`, `enjl`, `njl`
and `zlvmit` on lepton-pressure keys (`P_eG`, `P_eL_H`, `P_eL_Q`) at ~1e-9
relative. On python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0 **all thirteen
baselines pass**. `test/baseline/*.npz` was regenerated on 3.14
([ticket 62](62-regenerate-baselines-py314.md)), so a 3.9 run compares against
another stack's artifact — which is the baseline module's own thesis, not a
regression. None of the four imports `eos.dd2` and this ticket touched no
lepton code either way. (An earlier draft of this resolution guessed at
ticket 108 and the concurrent session; running the reference interpreter
showed the guess was unnecessary.)

Landed: `eos/dd2/nmp.py` (the `snm_derivatives` section, `compute_nmp` and
`_isoscalar_quantities` rewired, `h` and `want_Q` gone, ISO_GATE, four
docstring sections), `eos/dd2/couplings.py` (`rational_d3f`),
`eos/dd2/__init__.py`, `eos/dd2/verify/run_full_check.py` (new check +
re-measured basin check), `eos/dd2/dd2.md`, `eos/dd2/dd2.tex` (the full
derivation, section 11), `docs/DEFERRED.md` (the dd2 Q_sat entry resolved,
replaced by the Z_sat non-gap), and in the untracked suite
`test/dd2/test_api.py`, `test/dd2/test_dd2_m8.py`,
`test/baseline/test_baseline.py`, `test/tov/test_solver_fast_robustness.py`.

Status: resolved (2026-08-28).
