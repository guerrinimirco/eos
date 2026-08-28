# The dd2 Q_sat closure pins `c_omega`, and the adopted selection rule prefers
# `c_sigma`

Type: task
Status: open
Blocked by: -   (103 adopted the rule; 111 made the closure usable)
Parent: ../map.md

## Question

[Ticket 103](103-nmp-closures-four-models.md) adopted one written procedure for
choosing which couplings a closure pins: build the Jacobian
d(NMP)/d(ln coupling) at the published point and pin whichever subset leaves
the largest smallest singular value `sigma_min`, then confirm with a basin scan
over a grid of targets, which may veto a locally-best choice.

Applied to dd2's Q_sat-imposing closure -- five isoscalar rows, one pin -- the
rule does not return what is shipped. Measured after
[ticket 111](111-dd2-analytic-nmp-derivatives.md) went analytic:

    pin c_sigma    sigma_min 2.969e-01   cond    1235
    pin b_omega    sigma_min 2.738e-01   cond    1377
    pin c_omega    sigma_min 2.370e-01   cond    1589   <- PINNED_WITH_Q_SAT
    pin b_sigma    sigma_min 1.177e-01   cond    3091
    pin Gamma_w    sigma_min 3.230e-09
    pin Gamma_s    sigma_min 3.354e-10

`cond` agrees with `sigma_min` here, so this is not the statistic disagreeing
with itself -- it is that `c_omega` was chosen in
[ticket 105](105-dd2-isoscalar-conditioning.md) on a Jacobian whose Q_sat row
still carried the third-difference stencil, and 111 removed that row's noise
without the ranking being re-run. The DEFAULT closure is unaffected: its
`b_sigma + c_omega` is first under both statistics (sigma_min 0.466, cond 135),
and the two Gamma rows confirm 105's claim that neither vertex coupling can
ever be pinned.

The margin is 25% in `sigma_min`, which is small enough that 103's ruling was
explicit: the basin scan gets a vote before a shipped default moves.

## Gate

A basin scan of both pins over a grid of targets spanning more than two axes,
reported side by side; `PINNED_WITH_Q_SAT` changed only if `c_sigma` wins that
too, and the ranking written into `dd2/nmp.py`'s "Why two coefficients are
pinned" section either way, since that section currently ranks by `cond`
alone. dd2 `run_full_check` PASS with golden SNM(0.16) and CompOSE HS(DD2)
unmoved; `test/dd2` green. No forward-path number moves -- this is the inverse
map's pin.
