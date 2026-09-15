# The conformal limit is imposed at a POINT, and a point is not a limit

Type: prototype
Status: open
Blocked by: 01
Parent: ../map.md

## Question

The fit carries one extra row demanding `P/P_free = 1` at `MU_CONFORMAL =
3.0e5` MeV (`notebooks/csc_bag_map.py:392-394`, `448-454`). It was added for a
measured reason and it works:

> *"MEASURED without this row, on the dense low-density grid: the CFL fit
> returns a4 = 20, a6 = -36, gs150 = +35 and P/P_free(60 GeV) = 20.5, which is
> an excellent fit to the data and a nonsense equation of state one decade
> above it."* (`csc_bag_map.py:384-391`)

With it: coefficients O(1), `P/P_free(inf)` 0.99-1.00 per phase, and no cost in
residual.

**But a row at one finite scale pins a VALUE, not a LIMIT, and the basis
contains a term that beats the limit.** `a4l` multiplies
`sum_f mu_f^4 ln(mu_f/mu_star)` (`csc_bag_map.py:163`), which grows FASTER than
`mu^4`. So `P/P_free -> infinity` as `mu -> infinity` for any `a4l != 0`,
however good the value at 3e5 MeV is. The row makes the sum right at one place
and says nothing about the slope there; a fit is free to satisfy it with a
large `a4l` balanced by a compensating `a4`, and `a4/a4l/a6` collinearity means
it has a whole valley of ways to do exactly that.

The other terms are already honest about the limit:

- `a6` SATURATES by construction -- `sum_f mu_f^6/(mu_f^2 + M_V^2)`
  (`csc_bag_map.py:167`), which is `mu^6/M_V^2` below the vector scale and
  `mu^4` above it, "never mu^6 forever". That fix is recorded with its physics:
  for `P ~ mu^k`, `c_s^2 -> 1/(k-1)`, so a bare `mu^6/mu_star^2` sends
  `c_s^2 -> 1/5` and `P/P_free -> infinity`. `M_V = 530 MeV = 3 M_g/(2 sqrt 2)`
  for the shipped gluon-exchange `M_g = 500 MeV` (`csc_bag_map.py:115-119`).
  Measured at `mu_B = 60 GeV`: `P/P_free = 0.938`, `c_s^2 = 0.316`.
- the massive gases tend to `mu^4`.
- `a0`, `a2`, `a2s` are subleading.

So `a4l` is the one term that does not die, and the docs already say what to do
about it. `docs/csc_bag_mapping.md:412-419`, on the NLO pairing coefficients of
Geissel et al.:

> *"At 10-15 n_sat their COEFFICIENTS are not usable. Their TERM STRUCTURE is:
> it is what says which functions of `mu` belong in the basis. **Use the
> structure, fit the coefficients.**"*

The pQCD structure for the logarithm is not a bare `mu^4 ln mu`. It is
`mu^4` times a series in `alpha_s(mu)`, and `alpha_s` RUNS TO ZERO. That is why
the real pQCD pressure reaches the conformal limit and a polynomial with a
bare log does not.

### What to try

**Replace the bare `a4l` with alpha_s-weighted terms.** Two candidates, both
still LINEAR in their coefficient so the fit stays one bounded `lsq_linear`:

    "a4a"   scale * sum_f mu_f^4 * a(mu_f)
    "a4a2"  scale * sum_f mu_f^4 * a(mu_f)**2

with `a(mu) = alpha_s(mu)/pi`. The running coupling is already in this
repository at `eos/general/pqcd.py:102-118` -- two-loop MSbar at `N_f = 3`,
`LAMBDA_MS = 378.0`, `b0 = (33 - 2*3)/(12 pi)`, `b1 = (153 - 19*3)/(24 pi^2)`.

**Taking it is a legal import and worth being explicit about why.** CLAUDE.md
section 1 lets a model import `general/`, and section 7 makes `general/` the
single home; `alpha_s` is a pure running-coupling function, model-independent
and fitted to nothing. What must NOT be taken is `pqcd.pressure`, which assumes
`mu_u = mu_d = mu_s = mu_B/3` (`pqcd.py:121-127`, `free_pressure` is
`mu_B^4/(108 pi^2)`) -- beta equilibrium, the one line this whole effort is
trying to leave. The module's own docstring says it is "the one first-principles
statement about the same quantity the models in this repository parametrize",
which is exactly the right relationship: `pqm` borrows the STRUCTURE and the
coupling, not the series.

Both terms die as `mu -> infinity` on their own, so the limit stops depending
on a cancellation.

**Then add a second conformal row on the logarithmic derivative:**
`d(P/P_free)/d ln mu -> 0` at the conformal scale. One more linear row (the
basis functions are differentiable in closed form after ticket 01). It is what
distinguishes "the value happens to be 1 here" from "the limit is 1".

### What might go wrong, and what it would mean

- **The residual in 2-15 n_sat gets worse.** `a4l` is doing real work in the
  fitted window -- it is one of the three collinear terms and the combination
  is pinned. If the alpha_s-weighted terms cannot span the same shape there,
  that is a finding: the log in the window is not the pQCD log, and the honest
  answer is to keep a bare log AND accept that the model has a stated upper
  domain rather than a conformal limit.
- **Nothing changes.** Plausible -- the ridge may already be driving `a4l`
  small. Then the ticket is cheap and the limit is genuinely secured rather
  than accidentally satisfied.
- **The second row over-constrains.** If demanding both the value and the slope
  makes the coefficients move a lot, the two rows disagree, which means the
  basis cannot reach the limit smoothly and the term set needs more thought.

## Gate

- `P/P_free` measured at **two scales a decade apart** (e.g. 3e4 and 3e5 and
  3e6 MeV), per pattern, both within **1%** of 1 -- and the sequence
  MONOTONICALLY approaching it, which one scale cannot show.
- `c_s^2` at the conformal scale within **0.02** of 1/3, per pattern.
- The 2-15 n_sat residual (P, n_B, n_C, n_S rms per pattern) **no worse than
  the ticket-01 baseline**, or a written statement of the trade and why it is
  accepted.
- The coefficients O(1) -- no reappearance of the `a4 = 20, a6 = -36` valley.
- A one-line verdict in the map's Decisions-so-far naming which term set the
  model carries from here.
