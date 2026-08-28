# The dd2 Q_sat closure pins `c_omega`, and the adopted selection rule prefers
# `c_sigma`

Type: task
Status: resolved (2026-08-29)
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

## Resolution (2026-08-29)

**The basin scan VETOES `c_sigma`. The shipped `c_omega` pin stands, and no
code changed — only the docstring section the gate names.** This is the case
[103](103-nmp-closures-four-models.md)'s second half was written for, and it
is the first time it has fired.

### The rule's local half reproduces exactly, and it does prefer c_sigma

The measurement convention was not written down anywhere and had to be
recovered before the ranking could be trusted: **Jacobian rows divided by each
nuclear-matter parameter's own published magnitude (`P` by 1 MeV/fm^3),
columns by the coupling.** That, and only that, reproduces this ticket's
table and 103's default-closure figures to four digits. It is now stated in
`nmp.py` so the next session does not have to re-derive it.

    Q_sat closure, five rows, one pinned
      c_sigma                sigma_min 2.9692e-01   cond 1235.5
      b_omega                          2.7379e-01        1378.0
      c_omega                          2.3706e-01        1589.6   <- shipped
      b_sigma                          1.1771e-01        3092.4
      Gamma_sigma, Gamma_omega         ~1e-10 (numerically zero)

Confirming the ticket's premise: `cond` ranks them in the identical order, so
this is not the two statistics disagreeing with each other, and neither vertex
coupling can be pinned at all.

### The scan overrules it, on both grids, at both restart counts

Same targets, same seeds, both pins, counting targets REACHED (`status.ok`):

                                             0 restarts   32 restarts
    72-cell grid, K_sat x Q_sat x m*/m x n_sat
      pin c_omega                              59/72         64/72
      pin c_sigma                              42/72         59/72
    200 random targets, the same four axes plus E_sat
      pin c_omega                             156/200       172/200
      pin c_sigma                             102/200       134/200

Four axes and five, so the gate's "more than two axes" is met twice over.
`c_omega` wins all four comparisons, by 17 cells and 54 targets at zero
restarts, and 32 restarts closes the gap without reversing it. It also runs in
about half the wall clock, for the same reason: a target reached on the first
solve never pays for restarts.

**Among the targets both pins reach, the two are indistinguishable** — worst
relative error over the five imposed rows 2.6e-11 against 2.7e-11, medians
~1e-12 either way. So the 25% `sigma_min` margin is real and buys nothing: it
describes accuracy inside the basin, and what differs is the basin's SIZE.

**All eight counts are identical on both stacks** (python.org 3.14.2 /
numpy 2.3.5 / scipy 1.17.0 and anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1),
as are the sigma_min and cond tables to five digits. The veto is a property of
the residual surface, not of a solver version — which is worth recording,
because a single-stack veto of a 25% margin would not have been safe to trust.

### What landed

`eos/dd2/nmp.py`, the "Why two coefficients are pinned" section, and nothing
else. It now: states the two-step rule and why the statistic is `sigma_min`
and not `cond`; gives the measurement convention; ranks BOTH closures by
sigma_min with cond beside it; says the local half prefers `c_sigma` and by
how much; says the scan vetoes it and shows the counts; and retires the old
five-row `cond` figures (259 / 354 / 703 / 4191) as the stencil's rather than
the map's. `PINNED_WITH_Q_SAT` is untouched.

The DEFAULT closure's six-choice table was re-measured at the same time and
its absolute numbers moved with [111](111-dd2-analytic-nmp-derivatives.md)
(128/165/185/305/323/354 -> 135/176/195/326/346/366) with the ordering
unchanged, so the section no longer quotes pre-analytic numbers anywhere.

### Gate

- `test/baseline/dd2.npz` **unmoved** — mtime 2026-08-27 10:57:55, md5
  `08fd3966330139dd04aceb64ffc3f72b`, and `test_baseline[dd2]` green.
- dd2 `run_full_check` **PASS**, golden SNM(0.16) `1.40e-05` and CompOSE
  HS(DD2) `2.83e-05`, both unmoved from 111.
- `test/dd2` + `test/baseline` on python.org 3.14.2: **231 passed, 0
  failed** (2:10) — 211 + all thirteen baselines.
- No forward-path number moves, and no solver number moves either: the diff is
  73 inserted and 16 deleted lines, all inside one docstring.

**One measured cross-stack note, not this ticket's.** The first gate run was
taken on anaconda 3.9.7 and produced four `test_baseline` failures — `ccdm`,
`enjl`, `njl`, `zlvmit` — which is exactly the set 111 recorded and exactly
what the map predicts, `test/baseline/*.npz` being 3.14 artifacts. None of the
four imports `eos.dd2`. Reported so a reader of this ticket's logs does not
read them as its own.

### Two things noticed and not changed

1. **`PINNED_WITH_Q_SAT` does not actually drive the closure.** `invert_nmp`
   builds `held` from it but then hardcodes `held["c_omega"]` in the Q_sat
   branch's residual and unknown vector, so changing the constant raises
   KeyError rather than moving the pin. Running this scan therefore required
   temporarily making the pin a variable (verified a bit-for-bit no-op at the
   published point: `1.478e-12` on 3.9, `1.364e-12` on 3.14, against 103's
   quoted 1.5e-12), and that patch was reverted with the verdict. A future
   session re-running this scan has to redo it.
2. **`invert_nmp`'s docstring still describes the pre-111 world.** Its
   `impose_Q_sat=True` branch says the Q_sat row "is a third finite difference
   and the closure amplifies its ~1.5e-3 relative floor by ~259", ending "NOT
   a closure to trust until the derivative is analytic" — which the module
   docstring two hundred lines above already contradicts. A ticket-111
   leftover.

Both are Stage 7 report material under the map's hard rule, not diffs.

Status: resolved (2026-08-29).

### Committed

`eos/dd2/nmp.py` and this file. **`map.md` is deliberately NOT in the commit:**
its working-tree copy carries a concurrent session's uncommitted entries for
[97](97-natural-record-leaves-the-result.md) and
[113](113-did-parameter-provenance.md), and committing it under a ticket-115
message would be a fourth instance of the shared-tree staging trap the map
already records three of. This ticket's Decisions-so-far entry is written into
the working tree and rides with whoever commits `map.md` next.
