# Close the derivative gap: n_a and s in closed form, not finite differences

Type: task
Status: closed
Blocked by: --
Parent: ../map.md

## Question

The prototype computes the conserved-charge densities by CENTRAL DIFFERENCE of
its own pressure, `notebooks/csc_bag_map.py:204-216`:

```python
def charges(f, mu_B, mu_C, mu_S, T=0.0, h=1.0):
    """(n_B, n_C, n_S) [fm^-3] as derivatives of the model's own pressure."""
    for axis in range(3):
        hi = [mu_B, mu_C, mu_S]; hi[axis] += h
        lo = [mu_B, mu_C, mu_S]; lo[axis] -= h
        out.append((pressure(..., *hi, ...) - pressure(..., *lo, ...)) / (2.0 * h))
```

with `h = 1.0 MeV`. The same differencing builds the density rows of the fit
itself (`csc_bag_map.py:436-445`).

This was a deliberate and good choice for a prototype -- the docstring says so
at `csc_bag_map.py:416-418`: *"the density rows are central differences OF THE
MODEL's own basis functions, which are elementary, so no chain rule is
maintained by hand and adding a term needs no derivative written for it."*

**But the whole point of fitting Omega rather than P and n separately is that
the identities then hold IDENTICALLY**, and with an `h = 1 MeV` difference they
hold to the truncation error of that difference instead. CLAUDE.md section 8
wants the Euler relation to ~1e-8 relative; a second-order difference of a
`mu^6`-scale function at `h/mu ~ 1e-3` is nowhere near that, and `s = -dOmega/dT`
is not computed at all.

There is no integral to invent. Every basis term is elementary EXCEPT the three
massive strange Fermi gases, and for those the derivatives are quantities the
same call already returns:

- `eos/alphabag/thermodynamics.py:218-230` shows the shape --
  `solve_fermi_jel(mu, T, m, G_QUARK, include_antiparticles=True)` returns
  `(n, P, eps, s)`. The prototype takes index `[1]` (the pressure) at
  `csc_bag_map.py:172` and throws `n` and `s` away. **Index 0 IS `dP/dmu` and
  index 3 IS `dP/dT`.**
- The model must take that from `eos/general/fermi_integrals.py` directly, not
  through `eos.alphabag` -- see ticket 10; but for this ticket, in the
  notebook, either import is fine.

The chain rule from `(mu_u, mu_d, mu_s)` back to `(mu_B, mu_C, mu_S)` is the
constant charge matrix of `eos/general/basis.py` (`quark_potentials`), which
CLAUDE.md section 2 says is declared once and imported. With `S = +1` per s
quark:

    mu_u = mu_B/3 + 2 mu_C/3
    mu_d = mu_B/3 -   mu_C/3
    mu_s = mu_B/3 -   mu_C/3 + mu_S

so `dX/dmu_B = (1/3)(X_u + X_d + X_s)`, `dX/dmu_C = (2 X_u - X_d - X_s)/3`,
`dX/dmu_S = X_s`, where `X_f = dX/dmu_f`. That IS `quark_charges`, which is why
the inverse already used in `flavour_densities` (`csc_bag_map.py:219-229`) is
the same map read the other way.

**One term needs care.** The CFL branch is evaluated at a COMMON potential
`mu = (mu_B + mu_S)/3` (`csc_bag_map.py:144-152`), so there `dP/dmu_C = 0` and
`dP/dmu_B = dP/dmu_S = (1/3) dP/dmu`, which reproduces `n_C = 0` and
`n_S = n_B` exactly -- the locking, as an identity of the derivative rather
than as a filter. Worth asserting: it is a free correctness check on the chain
rule.

**A second thing to settle while in here.** `TERM_BOUNDS` says one thing in its
comment and ships another: `csc_bag_map.py:186` reads *"its upper bound is 2
rather than the perturbative 1"*, while `csc_bag_map.py:193` ships
`"a4": (0.0, np.inf)`. One of them is wrong. The physics behind the bound is
recorded and is not in question -- the RG-consistent NJL medium
(`lambda_UV = 10`) is STIFFER than a free quark gas, so `a4 > 1`, and with the
perturbative bound imposed `a4` pegs there for every phase at every parameter
point. What is in question is whether the ceiling is 2, or absent. Decide it
and make the comment and the code agree.

### Why this is first

Everything downstream reads the derivatives. Ticket 02's conformal rows are
statements about the limit of `P/P_free` and its logarithmic derivative;
ticket 05 fits `s`, which does not exist yet; ticket 08's construction needs
`n_B(mu)` to be exactly the derivative of the `P(mu)` it compares. Doing this
after them means redoing them.

It also costs nothing in accuracy by itself -- the difference and the closed
form agree to the FD error -- which is what makes the gate below checkable.

## Gate

- `charges()` and a new `entropy()` return closed-form `-dOmega/dmu_a` and
  `-dOmega/dT`, with the massive-gas derivatives taken from `solve_fermi_jel`'s
  own `n` and `s` and the chain rule taken from `eos/general/basis`.
- **The identities, asserted rather than measured:** `eps + P - T s - sum_a mu_a n_a`
  and `f - (eps - T s)` and `f - (-P + sum_a mu_a n_a)` all <= **1e-12**
  relative, at every point of the existing T = 0 grid and at a handful of
  T > 0 points, in all three patterns.
- **Closed form against the old central difference**, same coefficients, same
  grid: agreement at the FD truncation level (expect ~1e-4 relative on `n_B`),
  with the difference SHRINKING as `h` shrinks. A disagreement that does not
  shrink with `h` is a chain-rule bug and this gate catches it.
- **The CFL identity**, asserted: on the CFL branch `n_C = 0` and
  `|n_S/n_B - 1| <= 1e-12` come out of the derivative, with no filter applied.
- **Refitting with closed-form density rows changes the coefficients by less
  than the old FD error**, on at least one parameter point per pattern. If it
  changes them by MORE, the FD rows were biasing the fit and that is a finding
  worth its own line in the map.
- The `a4` bound inconsistency resolved, with the reason written where the
  bound is.

## Outcome

Closed 2026-09-14. `.scratch/pqm/checks/01_derivatives.py` is the gate, 17
checks, all passing in ~1.5 s off the `output/csc_map_cache/` samples (python
3.14.2, numpy 2.3.5, scipy 1.17.0).

- `basis_terms` returns `Term(X, dX/dmu_B, dX/dmu_C, dX/dmu_S, dX/dT)` per
  term -- value and slopes side by side, so a new term cannot be added with
  its derivative forgotten. `basis()` is now the values alone. `charges()` is
  closed form, `entropy()` is new, and `fit_phase`'s density rows are the same
  slopes, so the coefficients are fitted to exactly the derivative the model
  later reports (and one basis evaluation per record replaces seven).
- The chain rule is `DMU_DCHARGE`, the Jacobian of `eos.general.basis`'s
  `quark_potentials` read off by evaluating it on the unit potentials -- the
  map is linear, so its columns ARE its Jacobian, and nothing is re-declared.
  `flavour_mu` now delegates to it too.
- The massive gases take n and s from `solve_fermi_jel` directly, so the
  `eos.alphabag` import is gone already (ticket 10's half of the job that
  falls out here).
- Identities: 2.8e-16 / 0 / 1.4e-16 over 336 T = 0 grid points and 5 T > 0
  points in all three patterns. They are structural given
  eps := -P + sum_a mu_a n_a + T s; the content is the FD gate.
- FD agreement: h = 4, 2, 1, 0.5 MeV, every pattern, T = 0 and T = 30 -- the
  difference falls by 64x for 8x in h, i.e. exactly h^2, in all six. At h = 1
  the old rows were good to 1.0e-3 (unpaired), 4.7e-4 (2SC), 3.6e-7 (CFL).
- Refit drift: the predicted P and n_B move by 1e-7 to 2.4e-6 of scale,
  two to three orders BELOW the FD error of the rows they replace. So the FD
  rows were not biasing the fit and no map line is owed for that. Individual
  coefficients move up to 5e-4 relative, which is the a4/a4l/a6 ridge valley
  and is why the gate is on the prediction.

**Two findings, both from the CFL bullet.**

1. **The gap power law ran on mu_B, which broke the locking off mu_S = 0.**
   `Delta = Delta_star (mu_B/mu_star)^sigma` gives dP/dmu_B a term dP/dmu_S
   does not have, so n_S/n_B - 1 came out at 2e-4 instead of zero -- and the
   CFL records are all n_S = n_B to 1e-6, so the old n_S row was asking the
   fit to match a number the ansatz could not produce, with `pair` absorbing
   the mismatch. The gap of a locked phase runs on the locked phase's own
   potential, mu_gap = mu_B + mu_S = 3 mu. On the sampled line the two are
   the same number, and the refit confirms it: rms P 2.41e-4 -> 2.42e-4, rms
   n_B and n_S unmoved, while `pair` collapses from 3.2e-4 to 1.7e-8 -- the
   same "the LO condensation coefficient goes to zero anyway" the map already
   records, now without a wrong row propping it up. n_C = 0 and n_S = n_B are
   both EXACTLY 0.00e+00 after it.
2. **The massive gases were cut off below mu_s = m.** A no-op at T = 0 (JEL
   returns exactly 0 there) and a jump at T > 0: the gs150 gas has
   P = 0.58 MeV/fm^3 at (mu_s, T) = (149, 30) and switched on discontinuously
   two MeV later. Dropped -- it changes no T = 0 number and makes dP/dT the
   entropy of the gas everywhere, which is what ticket 05 will fit.

**The a4 bound: no ceiling, and the comment now says so.** Measured at rkh,
eta_D = 1.45, G_V0/G_S = 0.5: a4 = 2.67 unpaired, 3.13 2SC, 2.81 CFL. A
ceiling of 2 pegs all three exactly as the ceiling of 1 did, so it is the same
wrong prior one notch looser. What keeps a4 finite is the conformal row, which
constrains the SUM of the mu^4 terms; a4, a4l, a6 and the gases are collinear,
so bounding one individually pins nothing.

**Not done here**, deliberately: `aT2`/`aT4` are still unfitted (GRID_T is
(0,)), so at T > 0 the entropy is the massive gases alone -- ticket 05. And
`lepton_thermo` is still the T = 0 gas, so a mode WITH leptons is a T = 0
statement; the quark phase's own s is what `phase_state` now reports.
