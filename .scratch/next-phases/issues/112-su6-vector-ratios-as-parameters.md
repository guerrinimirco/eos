# The SU(6) vector ratios become parameters, and the Delta ratios become free
# variables

Type: task
Status: open
Blocked by: -   (103 ruled it; 106 waits on this)
Parent: ../map.md

## Question

Ruled by the user on [ticket 103](103-nmp-closures-four-models.md)
(2026-08-29). Today the hyperon VECTOR coupling ratios are hardcoded at their
SU(6) values (`sfho/parameters.py:37`, dd2's hyperon constructor) and only the
SCALAR ratios are inverted from the single-particle depths. SU(6) is an
assumption, not a measurement, and CLAUDE.md section 6 is unambiguous about
what that costs: "a parameter that can only be changed by editing a source
file makes inference impossible".

**The ruling.** For every RMF model with a hyperon sector:

- `x_sigma_H` stays fixed by the potential depth `U_H`, as now.
- `x_omega_H`, `x_rho_H`, `x_phi_H` become **SU(6) x y**, with `y` an ordinary
  parameter on `Parameters` defaulting to 1.0. **Nine factors**, one per
  (meson, multiplet) pair over M = omega, rho, phi and H = Lambda, Sigma, Xi
  -- named fields, not a dict, so a sampler varies them by name and section 13's
  explicit form is kept.
- The **Delta sector takes no factors**: its couplings are the free variables
  `x_Delta_sigma`, `x_Delta_omega`, `x_Delta_rho` directly.

The shipped sets are the test that nine is the right count and not three:
SFHoY is `y = 1.5` on omega AND phi for Lambda and Sigma, and `y = 1.875` on
both for Xi, with the rho ratios left at SU(6) (`sfho/parameters.py:391-410`);
SFHoY* is all ones. A per-meson factor shared across multiplets cannot express
that; a per-multiplet factor shared across mesons cannot express a set that
breaks omega and phi differently.

One documentation point the ruling names: `y_phi_H` multiplies a NEGATIVE
SU(6) ratio (-sqrt(2)/3 for Lambda and Sigma, -2sqrt(2)/3 for Xi), so a factor
above one makes the coupling more negative. Say so where the field is defined,
or a reader takes "1.5x" for "more repulsion".

## Gate

Every published hyperonic set reproduces its current couplings exactly through
the new fields; `SFHoY` and `SFHoY*` are expressed as factor sets rather than
as absolute numbers; the per-model documents state the nine factors and the
Delta free variables; every model's `verify` and `test/<model>` green with no
baseline moved. [Ticket 106](106-su6-breaking-rescaling.md) is unblocked by
this and is where the rescaling itself is decided.
