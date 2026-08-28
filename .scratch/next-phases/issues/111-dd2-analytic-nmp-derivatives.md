# The DD2 nuclear-matter derivatives should be analytic, and Q_sat cannot be
# imposed until they are

Type: task
Status: open
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
