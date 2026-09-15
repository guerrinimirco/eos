# The (eta_D, G_V0/G_S) surface, refitted with the T data

Type: task
Status: open
Blocked by: 05, 08
Parent: ../map.md

## Question

CLAUDE.md section 6 is unambiguous: **model parameters are ARGUMENTS**, never
module-level constants, because *"inference varies couplings, nuclear-matter
parameters and B across millions of calls; a parameter that can only be changed
by editing a source file makes inference impossible."*

For `pqm` that means the diquark and vector couplings must be **continuous
axes**, not fit labels. One fit per `(eta_D, G_V0)` pair would make the model
useless for sampling.

The prototype already solved this, and measured it
(`notebooks/csc_bag_map.py:723-724`, `837-932`): a **quadratic surface** in
`(eta_D, G_V0/G_S)` per coefficient per pattern, fitted over
`MAP_ETA_D = (0.75, 1.00, 1.25, 1.45, 1.65)` x
`MAP_G_V0 = (0.25, 0.50, 0.75)` = 15 NJL parameter points.

**Leave-one-out, median error in beta-equilibrium P:**

| regressor | dense grid | earlier sparse grid |
|---|---|---|
| **quadratic** | **0.9%**, 98% phase right | 1.4% |
| Gaussian kernel ridge | 1.9% | 2.9% |
| bilinear | 4.1% | 2.8% |

Quadratic is the pick, and the direction of travel matters: **it got BETTER
with denser data while bilinear got worse**, and the kernel method overfits at
this data volume. That is the measurement that says no neural network is needed
here.

This ticket redoes that surface on the finite-T fits, which changes two things:
the coefficient set is larger (`aT2`, `aT4`, possibly a changed massive-gas
count from ticket 05) and `(Delta_star, sigma, tc_coeff)` now also need
surfaces.

### Three things the refit must keep

**1. The ridge, and the reason for it.** `a4`, `a4l` and `a6` are nearly
collinear over any finite `mu` range, so the least squares pins their
COMBINATION and not each one: individual coefficients jump 20-40% between
neighbouring NJL points while the EoS they predict is good to 2.5%. `RIDGE =
1e-3` (`csc_bag_map.py:373-382`) picks the minimum-norm solution out of the
valley, which is a SMOOTH function of the data -- and a smooth map is the whole
object of the exercise. If ticket 02 replaced `a4l` with alpha_s-weighted
terms, the collinearity may be reduced; measure whether the ridge is still
needed rather than carrying it by inheritance.

**2. The right test is the EoS, not the coefficients.**
`csc_bag_map.py:822-828`: *"the per-coefficient deviations printed above are
the wrong test. The right one is whether an EoS built from PREDICTED
coefficients still reproduces `eos.njl`."* Leave-one-out over parameter points,
scoring the predicted EoS in beta equilibrium and in the fixed-Y modes -- not
the coefficient error.

**3. `eta_V` is not the axis.** Recorded on the map and worth restating here
because it is the one place the brief was wrong: `eta_V` is read only by
`vector_form="constant"` (`eos/njl/parameters.py:73`), which is unmappable --
measured unpaired rms in P of 2.1% / 9.8% / 13.8% at eta_V = 0 / 0.5 / 1,
against 3.8% for `gluon_exchange`. The continuous vector axis is
**`G_V0_over_GS`**. `M_g` is a third potential axis (it sets `M_V = 3 M_g/(2
sqrt 2)` in the saturating `a6` term) and is held at 500 MeV for now; whether it
needs to vary is a question for after this ticket.

### How it lands in `parameters.py`

The `pqm.Parameters` dataclass carries `eta_D`, `G_V0_over_GS`, `M_g`,
`vector_form` and `tc_coeff` as ordinary fields -- the things a sampler varies
-- and the SURFACE coefficients as the published, named set
(`Parameters.default()`, `Parameters.named(...)`). Evaluating the surface at the
current `(eta_D, G_V0_over_GS)` gives the basis coefficients; that evaluation
happens on the parameter object, which is where CLAUDE.md section 5 puts it.

The surface itself is a fitted artefact of a study, so it is DATA shipped with
the model, not code that imports `eos.njl`. That keeps section 1 intact.

### The extrapolation question, which is open

A quadratic surface is fitted over `eta_D` in [0.75, 1.65] and `G_V0/G_S` in
[0.25, 0.75]. `eos/njl/parameters.py:53-70` recommends sampling `eta_D` over
0.75-1.65 with *"the upper half better motivated (QHC19 1.35-1.65, Kunkel
1.45)"*, so the box matches the prior -- but a sampler WILL walk outside it, and
a quadratic extrapolates badly and silently. The model must say what it does
there: clamp, extrapolate with a flag, or return non-convergence. Section 6
says non-convergence is a return value the caller can test, which is the
natural answer, but it needs deciding here rather than at ticket 10.

## Gate

- Surfaces refitted on the finite-T coefficient set, including
  `(Delta_star, sigma, tc_coeff)`.
- **Leave-one-out over the parameter points: <= 1% median in beta-equilibrium
  P, >= 95% phase right**, scored on the PREDICTED EoS and not on coefficient
  error.
- The same leave-one-out reported for the fixed-Y modes, which the earlier pass
  never scored.
- A measured statement of whether the ridge is still required after ticket 02,
  and at what weight.
- **A decided behaviour outside the fitted box**, with the choice written down
  and the box's bounds recorded for `pqm.tex`.
- The quadratic re-confirmed against bilinear and kernel ridge on the new data,
  or the pick changed with the numbers that changed it.
