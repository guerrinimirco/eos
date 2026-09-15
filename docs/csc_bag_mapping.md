# Mapping NJL colour-superconducting matter onto a bag model

A closed-form equation of state for deconfined quark matter with colour
superconductivity, whose parameters are fitted to `eos.njl` phase by phase, so
that a construction, a TOV sweep or a Bayesian sampler can be run at bag-model
cost while the numbers still come from the NJL model.

The target application is compact stars, mergers, supernovae, mixed phases and
possibly heavy ions, i.e. baryon densities up to roughly 10-15 n_sat and
temperatures from 0 to a few tens of MeV.


## 1. Why

`eos.njl` solves, at every point, a coupled system for three constituent
masses, three gaps, two colour potentials and the mode's own conditions, with a
Bogoliubov-de Gennes spectrum diagonalised at every quadrature node. Measured
on this machine (`rkh`, eta_D = 1.45, T = 0, `backend='fast'`, warm-started
along a density ladder):

| pattern  | one residual | one solved density |
|----------|--------------|--------------------|
| 2SC      | 2.35 ms      | 31-108 ms          |
| CFL      | 7.52 ms      | ~470 ms            |

The CFL cost is structural rather than a solver weakness: with only Delta_3
nonzero the 9x9 quasiparticle problem splits into four 4x4 blocks, and with all
three gaps on it splits into six 4x4 blocks plus one 12x12, whose
diagonalisation is ~27 times the cost of a 4x4 at every node.

A bag-model evaluation of the same phase is a handful of elementary functions:
microseconds, no iteration, and analytically differentiable and invertible.
What follows is how to get from one to the other without giving up the physics
that made the NJL numbers worth having.


## 2. Conventions

The repository's, throughout (CLAUDE.md sections 2 and 5):

- **S = +1 per s quark**, the opposite of the PDG sign.
- **C is the charge of strongly-interacting matter only**; leptons are excluded
  and enter through the separate condition of total electric neutrality.
- The public boundary is fm-based: `n` in fm^-3, `T` and every `mu` in MeV,
  `eps` and `P` in MeV/fm^3. Expressions below are written in natural units
  (MeV^4 for a pressure) and converted once by `(hc)^3 = 7.6838e6 MeV^3 fm^-3`.

The flavour potentials follow from `mu_i = B_i mu_B + C_i mu_C + S_i mu_S` with
the quark quantum numbers of `eos.general.basis`:

    mu_u = mu_B/3 + 2 mu_C/3
    mu_d = mu_B/3 -   mu_C/3                                          (mu_f)
    mu_s = mu_B/3 -   mu_C/3 + mu_S

and the inverse charge sums are `n_B = (n_u + n_d + n_s)/3`,
`n_C = (2 n_u - n_d - n_s)/3`, `n_S = n_s`.

Colour potentials `mu_3` and `mu_8` are NOT in this list. They are not
conserved charges of a mixed system, and `eos.njl` closes colour neutrality
internally; the bag model has no colour potentials at all, so whatever their
cost is, the fit absorbs it into the other parameters. This is stated here
because it is a real approximation, not an omission.


## 3. What the NJL side supplies

No inversion is needed to build the data the fit consumes. The phase-adapter
contract (CLAUDE.md section 5) already runs in the required direction --
potentials in, thermodynamics out:

    phase = njl_phase(par, SpeciesFlags(csc=True), patterns=("CFL",),
                      backend="fast")
    th = phase.thermo(mu_B, mu_C, mu_S, T)      # -> PhaseThermo

`PhaseThermo` carries `n_B`, `n_C`, `n_S`, `P`, `eps`, `s`, the per-species
densities and `mu_i`. A grid over `(mu_B, mu_C, mu_S, T)` is therefore a grid
of direct evaluations, with the pairing pattern DECLARED (`patterns=("CFL",)`)
rather than enumerated, so that one branch is followed and the fit is not
handed a first-order transition it cannot represent.


## 4. The ansatz

One thermodynamic potential per phase, with the phase entering through a single
counting coefficient and the value of the gap. Write the pressure of the
strongly-interacting sector as

    P_phase(mu_u, mu_d, mu_s, T) = sum_f P_f(mu_f, T; m_f, a4)
                                   + dP_pair(mu_u, mu_d, mu_s, T; Delta, phase)
                                   - B                                    (P)

### 4.1 The free term, with the perturbative correction

`P_f` is the ideal Fermi gas of one flavour with degeneracy g = 6 (three
colours times two spins), mass `m_f`, multiplied by the leading perturbative
factor `a4 = 1 - 2 alpha_s/pi`. At T = 0 it is elementary,

    P_f = a4 (g/(48 pi^2)) [ mu_f k_f (2 mu_f^2 - 5 m_f^2)
                             + 3 m_f^4 ln((mu_f + k_f)/m_f) ]        (P_free)

with `k_f = sqrt(mu_f^2 - m_f^2)`, and at T > 0 it is the same gas through the
integrals of `eos/general`. Expanding it in `x = m_f^2/mu_f^2` gives

    P_f = a4 (mu_f^4/(4 pi^2)) [ 1 - 3 x + (9/8 + (3/2) ln 2) x^2
                                 - (3/2) x^2 ln sqrt(x) + O(x^3) ]   (P_exp)

**verified numerically against `eos.alphabag.fermi_thermo` to 5e-8 relative**
over mu = 600-2000 MeV at m = 60, 100, 150 MeV. Two things follow from (P_exp).

First, at a common `mu` with only the s quark massive, the O(m_s^2) term is
`- 3 m_s^2 mu^2/(4 pi^2)`, which is exactly term (ii) of the
Alford-Braby-Paris-Reddy pressure that `eos.abpr` evaluates and that the
CSS-style fits use. The flavour-resolved form is therefore not a different
model, it is the same model before the common-`mu` assumption is made.

Second, **(P_exp) is not what should be implemented.** At 10-15 n_sat the
quark potential is `mu_q ~ 400-800 MeV` while a constituent or effective
strange mass is 100-500 MeV, so `x` is 0.05-1.5 and an expansion in it is not
controlled. (P_free) is closed form and costs the same, so the exact massive
gas is what the model carries, and the expansion appears only where it is
needed to identify a term.

### 4.2 The pairing term

The condensation pressure of a paired phase, to leading order in the gap, is
one factor of `Delta^2 mu^2` per gapped quasiparticle. Writing it per flavour,
as `eos.alphabag.cfl_P_correction` already does for CFL:

    dP_pair = c_phase (mu_u^2 + mu_d^2 + mu_s^2) Delta^2 / pi^2       (dP)

    c_phase =  1     CFL       (all nine modes gapped)
    c_phase =  1/3   2SC       (four modes gapped: u,d in two colours)
    c_phase =  0     unpaired

At a common `mu` this is `3 Delta^2 mu^2/pi^2` for CFL and
`Delta^2 mu^2/pi^2` for 2SC. The ratio 3 : 1 is not a convention. It is the
counting `sum over gapped quasiparticles of Delta_qp^2`, which for CFL is
`8 Delta^2 + (2 Delta)^2 = 12 Delta^2` and for 2SC is `4 Delta^2`; and it is
confirmed independently by the leading term of the coefficient `gamma_1` of
Geissel, Gorda and Braun [arXiv:2504.03834], who obtain `gamma_1^CFL = 4` and
`gamma_1^2SC = 4/3` in the normalisation `p = p_free gamma_1 Delta_bar^2`,
`p_free = mu_B^4/(108 pi^2)`, `Delta_bar = Delta/(mu_B/3)`. The same 3 : 1.

Two consequences worth stating, because they explain results that look odd:

- **A single effective gap serves every phase** if it is defined as
  `Delta_0^eff = sqrt( sum_i Delta_i^2 / 3 )` over the model's three
  condensates. It equals `Delta` for CFL and `Delta_3/sqrt(3)` for 2SC, which
  is why an NJL scan can show `Delta_3(2SC)` LARGER than `Delta_3(CFL)` while
  `Delta_0^eff(CFL)` is the larger of the two: the CFL gap is smaller and paid
  nine times over.
- At next-to-leading order the coefficient acquires an `alpha_s` dependence,
  `gamma_1^CFL = 4 - 4 m_s_bar^2/3 + 40.9 alpha_s` against
  `gamma_1^2SC = 4/3 + 9.17 alpha_s`, so `c_phase` is strictly a function of
  `alpha_s` too. Whether to carry that is a modelling choice discussed in
  section 7.

### 4.3 The gap is not a constant

`Delta` in (dP) is a function of the chemical potential. Following
Geissel et al. Eq. (13), take a power law,

    Delta(mu_B) = Delta_star (mu_B / mu_star)^sigma ,  mu_star = 2600 MeV  (Delta)

with `Delta_star` and `sigma` free. This is not a cosmetic refinement: over a
CFL branch of `eos.njl` at fixed couplings, `Delta_0^eff` moves by 14-42 MeV
across the fit window, so a constant gap misrepresents the branch by 10-20%
before any parameter is fitted.

Fitting (Delta) to the `eos.njl` branches gives (measured, `rkh`, T = 0,
beta equilibrium, n_B = 0.6-1.6 fm^-3):

| eta_D | eta_V | phase | sigma | Delta_star(2.6 GeV) |
|-------|-------|-------|-------|---------------------|
| 1.45  | 1.0   | CFL   | 0.41  | 222 MeV             |
| 1.45  | 1.0   | 2SC   | 0.32  | 181 MeV             |
| 1.45  | 0.0   | CFL   | 0.73  | 357 MeV             |
| 1.25  | 0.5   | CFL   | 0.68  | 232 MeV             |
| 0.75  | 0.0   | CFL   | 2.34  | 425 MeV             |

`sigma > 0` everywhere, i.e. the NJL gap RISES with density. The
weak-coupling value is `sigma ~ -0.23` and the functional-renormalisation-group
value `sigma ~ +0.45`; at the well-motivated corner of the NJL parameter space
(eta_D = 1.45) this model sits on the fRG value, and at weak diquark coupling
it leaves the `sigma` in [-0.5, 0.5] prior of that reference entirely.

For a finite-temperature model the BCS closing of the gap is carried by the
same factor `eos.alphabag.cfl_gap` already uses,
`Delta(T) = Delta(0) sqrt(1 - (T/T_c)^2)` with `T_c = tc_coeff * Delta(0)`.

### 4.4 The parameters

Five per phase:

| symbol      | meaning                                        |
|-------------|------------------------------------------------|
| `alpha_s`   | through `a4 = 1 - 2 alpha_s/pi`, multiplying the free gas |
| `m_s`       | effective strange mass in (P_free), carried exactly |
| `B`         | bag constant, quoted as `B^(1/4)` in MeV       |
| `Delta_star`| gap at the reference potential `mu_star`       |
| `sigma`     | its power-law exponent                          |

`m_u = m_d = 0` is kept, as in `eos.alphabag`. Note that `m_s` here is a FITTED
EFFECTIVE MASS and is a third object, distinct from both the NJL current mass
(140.7 MeV in the RKH set) and the NJL constituent mass `M_s` (a solved
quantity, typically 300-500 MeV): it absorbs whatever the exact massive gas
must do to imitate the chiral dynamics.

The parameters are per phase, i.e. unpaired, 2SC and CFL each carry their own
set. That is a deliberate consequence of the ansatz, not an admission: the NJL
branches differ in their constituent masses and their condensates, and a single
`(alpha_s, m_s, B)` shared across the three would have to reproduce that
through the pairing term alone, which it cannot.

### 4.5 Everything else follows by differentiation

Because (P) is a closed-form potential, no further fitting is needed for any
other quantity:

    n_f  = dP/dmu_f = n_f^free(mu_f, T; m_f, a4)
                      + 2 c_phase mu_f Delta^2/pi^2
                      + 2 c_phase (mu_u^2+mu_d^2+mu_s^2) Delta (dDelta/dmu_f)/pi^2
    n_B  = (n_u + n_d + n_s)/3,  n_C = (2 n_u - n_d - n_s)/3,  n_S = n_s
    s    = dP/dT
    eps  = -P + sum_f mu_f n_f + T s                                   (Euler)
    c_s^2 = dP/deps  along whichever trajectory is being asked about

with `eps` DEFINED by the Euler relation rather than summed independently, so
that thermodynamic consistency holds by construction and CLAUDE.md section 8's
first invariant is satisfied exactly rather than to a tolerance. The third term
in `n_f` is the one a constant-gap model does not have, and it is where a
`sigma > 0` gap contributes to the sound speed.


## 5. Why the fit must be done in the potentials

Along a one-dimensional beta-equilibrium trajectory the three flavour
potentials are slaved to `mu_B`, so every basis function in (P) collapses onto
the same two powers of `mu`:

    a4 term     ~ mu^4
    m_s term    ~ m_s^2 mu^2
    pairing     ~ Delta^2 mu^2
    bag         ~ const

`m_s^2` and `Delta^2` multiply the SAME function of `mu`, so only the
combination `m_s^2 - 4 Delta_0^2` is determined -- which is precisely why the
CSS-style parametrisation is usually written with that combination in it. A
fit along such a trajectory can only ever report the combination, and if the
gap is taken from elsewhere the mass follows, or vice versa, but never both.

Measured consequence, from a 4x3 scan in `(eta_D, eta_V)` on the `rkh` set with
a three-parameter fit in `(mu^4, mu^2, 1)`: the fit reproduces `P(mu)` to
0.03-0.3% for every phase and every parameter point, and the recovered
parameters are outside the bag model's own window at every one of them --
`a4 = 1.5-1.9` (i.e. `alpha_s < 0`) with `B < 0` at `eta_V = 0`, and
`m_s^2 < 0` from `eta_V = 0.5` upward. An excellent fit to a wrong ansatz.

Off the trajectory the degeneracy is lifted, because the basis functions stop
being proportional:

    a4       multiplies  mu_u^4 + mu_d^4 + mu_s^4
    m_s      enters only through  mu_s
    Delta    multiplies  mu_u^2 + mu_d^2 + mu_s^2
    B        multiplies  1

Three independent directions `(mu_B, mu_C, mu_S)` separate these four
functions. This is the substantive reason for the multi-dimensional fit; the
convenience reason -- that the adapter evaluates directly in those variables --
merely makes it cheap.

### 5.1 The CFL phase is one-dimensional whatever the grid does

The paragraph above holds for the unpaired and 2SC phases. It does NOT hold
for CFL, and the reason is the locking itself.

A colour-neutral CFL phase has `n_u = n_d = n_s` (Alford, Rajagopal and Wilczek;
Rajagopal and Wilczek, Phys. Rev. Lett. 86, 3492), hence `n_C = 0` identically
and `n_S = n_B` identically. Then

    dP = n_B dmu_B + n_C dmu_C + n_S dmu_S = n_B d(mu_B + mu_S)          (lock)

so at fixed `T` the CFL pressure is a function of the SINGLE combination
`mu_B + mu_S`, and `mu_C` does not enter at all.

Measured, `eos.njl` through `njl_phase(..., patterns=("CFL",))` on the `rkh`
set at eta_D = 1.45, eta_V = 1, T = 0:

| mu_B | mu_C | mu_S | mu_B + mu_S | P [MeV/fm^3] | n_C |
|------|------|------|-------------|--------------|-----|
| 1900 |   0  |   0  | 1900        | 562.795      | 0   |
| 1900 | -60  |   0  | 1900        | 562.795      | 0   |
| 1800 |   0  | 100  | 1900        | 562.795      | 0   |
| 1840 |   0  |  60  | 1900        | 562.795      | 0   |
| 1900 |   0  |  60  | 1960        | 629.867      | 0   |

Identical to every digit printed. The same table for 2SC and unpaired shows all
three potentials moving `P` independently, as section 5 assumes.

So for CFL the `m_s` / `Delta` degeneracy is not an artefact of a badly chosen
trajectory -- it is a property of the phase, and it cannot be broken by any
grid in `(mu_B, mu_C, mu_S)` at fixed `T`. Three remedies, in the order they
should be tried:

1. **Take `Delta` from the NJL solve.** It is an OUTPUT of the model, carried
   on every point as `Delta_1, Delta_2, Delta_3`; there is no reason to infer
   from `P` a quantity the model reports. Fit (Delta) to `Delta_0^eff` along
   the branch, hold it fixed, and the `mu^2` coefficient then determines `m_s`
   uniquely. This resolves CFL completely and is what section 6 does.
2. **Use the temperature axis.** The gap closes with `T` and the mass term does
   not, so `P(mu, T)` at `T` approaching `T_c` separates them on physical
   grounds. This is a real extra dimension and it lies inside the intended
   application range (mergers, proto-neutron stars) rather than being invented
   for the fit.
3. Do NOT transfer `(alpha_s, m_s)` from the unpaired fit of the same NJL
   parameter set. It is tempting and it is wrong: the constituent masses differ
   sharply between the branches -- measured on the same set at
   `mu_B = 1800 MeV`, `M_s = 463 MeV` unpaired, `447 MeV` in 2SC and `247 MeV`
   in CFL, since chiral restoration is far advanced in the locked phase -- so
   a shared effective `m_s` would be a statement about a mass the phases do not
   share.


## 6. The fit

### 6.1 Data

A grid over `(mu_B, mu_C, mu_S)` at each temperature, one pattern declared per
grid, evaluated through `njl_phase(...).thermo`. Keep only points where the
phase exists in the declared layout (`pattern_realised == pattern`) and the
solve converged.

For the intended applications the grid should cover, at minimum:

- `mu_B` spanning `n_B` from the branch's lower terminus to 10-15 n_sat;
- `mu_C` from the CFL-neutral value (`mu_C = 0`, since a colour-neutral CFL
  phase is electrically neutral with no electrons) out to the unpaired
  beta-equilibrium value and somewhat beyond, so that mixed-phase constructions
  which drive the quark phase off neutrality are inside the fitted region; Consider we want to use this also for YC=0.5 tables.
- `mu_S` around the beta-equilibrium value (which is `mu_S = 0`, since weak
  equilibrium `s <-> d` gives `mu_s = mu_d`), wide enough that `mu_s` moves
  independently of `mu_u` and `mu_d` -- this is the direction that separates
  `m_s` from `Delta`, so a grid narrow in `mu_S` gives back the 1-D degeneracy. Useful cases: YS=0 and µS=0.

For CFL the second and third bullets are inoperative by section 5.1: the grid
collapses to a line in `mu_B + mu_S` and `Delta` is taken from the NJL gaps
instead. The fit code must therefore treat the phases differently, and it is
better that this shows in the code than that a rank-deficient design matrix is
handed to a least-squares routine which will happily return an answer.

### 6.2 Residual

Fit the pressure AND the three charge densities, which the adapter returns at
no extra cost and which are the first derivatives of the same potential:

    r = [ (P_fit - P_njl)/P_scale,
          w (n_B_fit - n_B_njl)/n_scale,
          w (n_C_fit - n_C_njl)/n_scale,
          w (n_S_fit - n_S_njl)/n_scale ]                          (residual)

Fitting the densities alongside the pressure is what pins `a4` and `sigma`: a
pressure fit alone is insensitive to a slope error that a bag constant can
absorb, and `sigma` enters the pressure only through `Delta^2` but the density
through `dDelta/dmu` as well.

### 6.3 Structure of the minimisation

At FIXED `(m_s, sigma)` the pressure (P) is LINEAR in the three remaining
parameters:

    P = a4 * [ sum_f P_f(mu_f, T; m_s, a4=1) ]
      + Delta_star^2 * [ c_phase (sum_f mu_f^2) (mu_B/mu_star)^(2 sigma)/pi^2 ]
      + B * [ -1 ]                                                    (linear)

so the fit is a two-dimensional outer minimisation over `(m_s, sigma)` wrapped
around a linear least squares in `(a4, Delta_star^2, B)` -- separable, or
variable-projection, nonlinear least squares. The outer problem is smooth and
two-dimensional, so a coarse grid followed by Nelder-Mead is sufficient and
there is no need for a gradient. This matters: it is what makes the fit
reliable enough to run inside a parameter sweep rather than by hand.

Constrain the outer problem to the bag model's own window -- `0 < a4 <= 1`,
`B > 0`, `m_s >= 0`, `Delta_star >= 0` -- and REPORT the constraint when it
binds. A fit that only succeeds by leaving the window is telling you the
ansatz is missing a term, and hiding that behind an unconstrained
least-squares is how a wrong model gets published with a small residual.


## 7. What this ansatz cannot represent, and what to do about it

**A constant `alpha_s` and a vector interaction are different mechanisms.**
`alpha_s` softens; the NJL vector coupling `eta_V` stiffens, and its
contribution is `~ G_V n_q^2`, which as a function of `mu` behaves like `mu^6`.
No choice of `(a4, m_s, B)` in (P) can produce that shape, and the negative
`m_s^2` values reported in section 5 are the fit trying to. Two honest
responses:

1. Fit only NJL parameter sets whose vector form is not constant. The
   `gluon_exchange` form saturates and is the one the specification recommends.
2. Add one term. `P += k mu_B^6/(mu_star^2 * ...)`, one extra linear parameter,
   restores the shape at the cost of a sixth number per phase and a model that
   is no longer strictly a bag model. This is a judgement call and should be
   made after seeing the constrained residual, not before.

**Gapless and crystalline phases are absent.** `eos.njl` enumerates
`unpaired, 2SC, CFL` (plus uSC, dSC and the free layout, which over 75 scanned
parameter points won exactly twice and were metastable both times). It does not
carry g2SC, gapless CFL, or crystalline LOFF phases, which are precisely what
occupies the Fermi-surface mismatch window. The bag ansatz cannot represent
them either. The correct treatment is to say so, not to widen the pattern
enumeration.

**The NLO pairing coefficient.** Carrying
`c_phase(alpha_s) = gamma_1(alpha_s)/4` from Geissel et al. is available and
free, but their own convergence statement is that the NLO expansion is "only
well converged down to about 50 n_0" and that the NLO sound speed exceeds the
conformal value even above 200 n_0. At 10-15 n_sat their COEFFICIENTS are not
usable. Their TERM STRUCTURE is: it is what says which functions of `mu`
belong in the basis. Use the structure, fit the coefficients.

**A first-order transition is not a phase.** Each fit is to ONE branch. The
transition between them is reconstructed afterwards by comparing the fitted
pressures at equal potentials, which is the same rule `eos.njl.solver.solve`
applies, and it comes out as the branch crossing it is.


## 8. Where this goes in the repository

The functional form of (P) already exists. `eos.alphabag` carries the exact
massive Fermi gas with the `a4` factor, the bag, and a CFL sector whose pairing
term is `(mu_u^2 + mu_d^2 + mu_s^2) Delta^2/pi^2` -- equation (dP) with
`c_phase = 1` -- with the densities, entropy and Euler-defined energy already
taken as derivatives of it. `eos.abpr` is its analytic T = 0 CFL limit with
closed-form `mu(n_B)`, `mu(P)` and `mu(eps)`.

So this is an extension of an existing model rather than a new one. What is
missing from `eos.alphabag` is:

- the 2SC sector: equation (dP) with `c_phase = 1/3`, pairing u and d in two
  colours, and the charge bookkeeping that goes with a phase that is NOT
  automatically neutral;
- the density-dependent gap (Delta), which is currently a per-call constant;
- the fitting itself, which is not model code: it consumes `eos.njl` and
  produces an `eos.alphabag.Parameters`, so it belongs with the study that
  needs it, not inside either model (CLAUDE.md section 1 -- a model never
  imports another model).

The order of work is: fit first, in a notebook, against the existing
`eos.alphabag` CFL sector and a hand-written 2SC pressure; look at the
constrained residuals; and only then decide whether the 2SC sector and the
running gap are worth adding to the model proper.


## 9. Acceptance criteria

The mapped model is worth having only if it passes all of these; each is
cheap and each has caught a different failure in the exploratory work.

1. **Residual.** `max |P_fit - P_njl| / max|P_njl| < 1%` and the same for
   `n_B`, over the whole fitted grid, with all parameters INSIDE the physical
   window.
2. **Sound speed.** `c_s^2` from the fit within 10% of `c_s^2` from NJL along
   the beta-equilibrium trajectory. This is the derivative test and it is much
   sharper than the pressure test; a fit can match `P` to a part in 1e3 and get
   `c_s^2` wrong by a factor.
3. **Thermodynamic consistency.** The Euler relation to 1e-8 relative, and
   `0 <= c_s^2 <= 1` on any table handed to a structure solver (CLAUDE.md
   section 8). The first is automatic from section 4.5 and should be asserted
   anyway.
4. **Branch ordering preserved.** Where NJL says CFL has the lower free energy,
   the fitted model must agree; a transition density reproduced to a few
   percent. A mapping that inverts the phase ordering is useless however small
   its residual.
5. **The M-R sequence.** `M_max` and `R_1.4` from the fitted EoS within a few
   percent of the NJL ones, through `eos.astro.tov`. This is the application
   test and it is the one that decides.


## 10. On emulators

An interpolating or neural surrogate is a legitimate later step and CLAUDE.md
section 6 explicitly designs for it, but it is the wrong first step here and
for reasons that are about the physics rather than about taste:

- a single closed-form potential gives `P`, `n_f`, `eps`, `s` and `c_s^2` as
  exact derivatives of ONE object, so the Euler and Maxwell relations hold
  identically; a surrogate fitted to those quantities separately does not have
  that property and will violate them by its own fit error;
- the mixed-phase construction inverts `mu(P)` and `mu(n_B)` in an inner loop,
  which the closed form does analytically;
- the parameters of the fitted model are `alpha_s`, `B`, `m_s`, `Delta_star`,
  `sigma` -- quantities with meaning, which is the entire point of doing the
  mapping rather than tabulating NJL;
- extrapolation outside the training box is uncontrolled, and the box here is
  four-dimensional `(mu_B, mu_C, mu_S, T)`.

If a surrogate is eventually needed, fit the single scalar
`Omega(mu_B, mu_C, mu_S, T)` and take every derived quantity from derivatives
of the network, rather than fitting each quantity independently.


## Appendix A. Measured, first pass

Setup: `rkh`, `eta_D = 1.45`, `T = 0`, `backend='fast'`, one pattern declared
per grid through `njl_phase`, `mu_B` in [1500, 2600] MeV on 10 points,
`mu_C` in {-80, -40, 0, 40} MeV, `mu_S` in {-80, 0, 80} MeV (CFL on the
`mu_B` line alone, per section 5.1). `sigma` from the NJL gaps; the linear
subproblem bounded to `0 < a4 <= 1`, `B > 0`, `m_s >= 0`.

**A.1 The vector sector decides whether a bag form can fit at all.** Unpaired
phase, same grid, only the vector setting changed:

| vector sector          | rms in P | B^(1/4) | pegged |
|------------------------|----------|---------|--------|
| constant, eta_V = 0    | 2.1%     | 217 MeV | a4 at 1 |
| constant, eta_V = 0.5  | 9.8%     | 0       | B at 0  |
| constant, eta_V = 1    | 13.8%    | 0       | B at 0  |
| gluon_exchange         | 3.8%     | 185 MeV | none    |

This is section 7's first paragraph, measured. A constant `G_V` contributes
`~ G_V n_q^2`, i.e. `~ mu^6`, and the `mu^4 / mu^2 / const` basis absorbs it
only by leaving the physical window. **The gluon-exchange vector form is the
one to map**, and that is a recommendation about which NJL parameter sets this
whole programme applies to, not a fitting detail.

**A.2 The three phases, gluon-exchange form.**

| phase    | a4    | alpha_s | m_s [MeV] | B^(1/4) [MeV] | Delta\*_fit | Delta\*_njl | sigma | rms P | rms n_B | rms n_C | rms n_S |
|----------|-------|---------|-----------|---------------|-------------|-------------|-------|-------|---------|---------|---------|
| unpaired | 0.656 | 0.541   | 187       | 185           | -           | -           | -     | 0.8%  | 2.1%    | 3.2%    | 6.6%    |
| 2SC      | 0.699 | 0.473   | 289       | ~0            | 232         | 220         | +0.41 | 1.0%  | 2.7%    | 4.1%    | 5.1%    |
| CFL      | 0.518 | 0.758   | 0 (pegged)| 229           | 250 (held)  | 250         | +0.28 | 1.0%  | 2.1%    | 3e-14   | 2.6%    |

Four things to read off it.

- **The pressure maps at the 1% level; the densities do not.** They are 2-7%,
  and since the densities are the first derivatives of the same potential this
  is the sharper test. Criterion 1 of section 9 passes on `P` and fails on
  `n_B`.
- **The 2SC gap is RECOVERED, not imposed.** `Delta_star` there is a free
  linear parameter of the fit and comes back at 232 MeV against the NJL value
  of 220 MeV, a 5% agreement with a quantity the fit was never told. That is
  the strongest evidence so far that the ansatz is the right shape.
- **`n_S` is the worst row everywhere**, and its worst point is always at the
  extreme of the `mu_S` grid, where the strange sea is nearly empty. A fixed
  effective `m_s` cannot follow the NJL constituent mass, which runs from
  ~460 MeV to ~200 MeV across these branches.
- **CFL pegs `m_s` at zero.** With `Delta` held at the NJL power law, the
  leading-order `Delta^2 mu^2` term supplies more pressure than NJL does, and
  the fit would need `m_s^2 < 0` to give it back. Either the LO pairing term
  overestimates the condensation energy at these densities, or the NJL gap is
  not the right object to insert into it. The next experiment is to release
  `Delta_star` for CFL as well and see how far the fitted gap falls below the
  NJL one -- accepting that section 5.1's degeneracy makes that fit weakly
  conditioned, which is why it was not the first thing tried.

**A.3 The map gets WORSE toward the transition, not better.** Restricting the
grid to `mu_B <= 2050 MeV`, i.e. `n_B` up to about 15 n_sat, which is the
stated application range:

| phase    | rms P | rms n_B | Delta\*_fit | Delta\*_njl |
|----------|-------|---------|-------------|-------------|
| unpaired | 2.1%  | 4.8%    | -           | -           |
| 2SC      | 1.8%  | 4.3%    | **0**       | 224         |
| CFL      | 1.5%  | 3.8%    | 258 (held)  | 258         |

Every residual roughly doubles, and the 2SC gap -- recovered to 5% on the wider
grid -- collapses to zero. The reason is chiral: at the low-density end the NJL
constituent masses are still running hard, and a bag model with a fixed `m_s`
and massless `u`, `d` has nothing to run. So the ansatz is at its best deep
inside the quark phase and at its worst near the onset, which is precisely
where a hybrid star puts the quark core. This is the finding that should shape
what comes next; it is not an argument against the mapping, but it says that
`m_s` (and possibly `m_u = m_d`) will have to be given a density dependence, or
the mapped model restricted to a stated density window and matched to something
else below it.


## Appendix B. The power-series parametrisation, and the map

The bag form of section 4 is one choice of basis. Dropping it for a general
power series -- which is what `notebooks/csc_bag_map.ipynb` builds -- makes
every coefficient linear, removes the nonlinear search entirely, and answers
the accuracy question. The pressure of one phase is

    P = (1/(4 pi^2)) sum_t a_t X_t(mu_f, T)

    a0    mu_star^4                        the bag constant, B = -a0/(4 pi^2)
    a2    sum_f mu_f^2 mu_star^2           a mass-like term
    a2s   mu_s^2 mu_star^2                 strangeness breaking
    a4    sum_f mu_f^4                     the free gas
    a4l   sum_f mu_f^4 ln(mu_f/mu_star)    the perturbative logarithm
    a6    sum_f mu_f^6/mu_star^2           the shape a vector coupling makes
    aT2   sum_f mu_f^2 T^2                 the leading thermal term
    aT4   T^4                              and the next
    pair  4 c_phase (sum_f mu_f^2) Delta^2 the condensation energy

with `Delta = Delta_star (mu_B/mu_star)^sigma` taken from the NJL gaps. The
bag model is the sub-basis `(a0, a2s, a4, pair)`.

**B.1 Measured.** `rkh`, gluon-exchange vector form, T = 0, fitting P and the
three charge densities together:

| test | result |
|------|--------|
| beta equilibrium at eta_D = 1.45, n_B = 1.55-3.20 fm^-3 | **max 0.2%, median 0.1%** in P |
| the equilibrium phase, on the sampled potentials | 92-96% agreement with `eos.njl` |
| cost | **0.5-1.0 ms/point against 175 ms** for `eos.njl`: 180-320x |
| the analytic map, leave-one-out over 15 NJL parameter points | median **2.5%**, worst 7.0%, phase right 95-100% |

The beta-equilibrium trajectory is a genuine test: the fit is made on a grid in
the potentials and never sees it.

**B.2 Three things the fit reports that are physics, not fitting.**

- **The leading-order condensation term does not survive.** Pinning
  `pair = 1`, its LO value, costs a factor of a hundred -- 19.2% maximum error
  in beta equilibrium against 0.2% -- and left free the coefficient goes to
  ZERO. At these densities `c_phase (sum_f mu_f^2) Delta^2/pi^2` is not what
  pairing does to the NJL pressure. The pairing physics still reaches the
  model through the per-phase coefficient set and through `Delta_star` and
  `sigma`; it is the textbook coefficient that does not.
- **`a4 > 1`, i.e. the NJL medium is STIFFER than a free quark gas.** With the
  perturbative bound `a4 <= 1` imposed, `a4` pegs there for every phase at
  every parameter point. The RG-consistent regularisation runs the medium
  integral to `lambda_UV = 10` times the vacuum cutoff, and a bag model's
  `a4 = 1 - 2 alpha_s/pi` cannot express that.
- **`a4`, `a4l` and `a6` are nearly collinear** over any finite range of `mu`,
  so the fit pins their combination and not each one. The individual
  coefficients therefore jump between neighbouring NJL parameter points and
  their bilinear surfaces deviate by 20-40% -- while the EoS those surfaces
  predict is good to 2.5%. A small Tikhonov ridge picks the minimum-norm
  solution out of the valley and makes the map smoother at no cost in
  residual.

**B.3 So an emulator is not needed.** A bilinear surface in `(eta_D, G_V0/G_S)`
-- four numbers per coefficient per phase -- already predicts an unseen NJL
parameter point to a median 2.5% with the right equilibrium phase. Section 10's
argument stands: the closed form keeps exact derivatives, analytic inversion
and parameters with names, and a network would have to beat 2.5% to be worth
giving those up.


## References

- M. Alford, M. Braby, M. Paris and S. Reddy, Astrophys. J. 629, 969 (2005) --
  the CFL bag pressure, terms (i)-(iv), and `eos.abpr`.
- M. Alford, A. Schmitt, K. Rajagopal and T. Schaefer, Rev. Mod. Phys. 80,
  1455 (2008) [arXiv:0709.4635] -- colour superconductivity review, pairing
  patterns and quasiparticle counting.
- M. Alford and K. Rajagopal, JHEP 06, 031 (2002) [arXiv:hep-ph/0204001] --
  the chemical-potential matrix, colour neutrality, `Delta > m_s^2/(4 mu)`.
- A. Geissel, T. Gorda and J. Braun, [arXiv:2504.03834] -- NLO CFL pressure,
  `gamma_1` for CFL and 2SC, the power-law gap ansatz, and the convergence
  statement quoted in section 7.
- T. Gorda and S. Saeppi, Phys. Rev. D 105, 114005 (2022) -- cold quark matter
  with perturbative quark masses.
- A. Kurkela, K. Rajagopal and R. Steinhorst, Phys. Rev. Lett. 132, 262701
  (2024) [arXiv:2401.16253] -- astrophysical constraints on the CFL gap.
- M. Buballa, Phys. Rept. 407, 205 (2005) -- the NJL side.
- H. Gholami, I. A. Rather, M. Hofmann, M. Buballa and J. Schaffner-Bielich,
  [arXiv:2411.04064] -- the RG-consistent NJL model `eos.njl` implements.
