# ABPR — the analytic colour-flavour locked parametrization at T = 0

Colour-flavour locked (CFL) quark matter at zero temperature, as the
closed-form pressure of Alford, Braby, Paris and Reddy, ApJ 629, 969 (2005): a
free three-flavour quark gas carrying the leading perturbative QCD correction
in a single factor `a4`, the leading cost of the strange quark mass as an
expansion in `m_s^2/mu^2`, the CFL condensation energy `3 Delta^2 mu^2/pi^2`,
and a bag constant. There is nothing to iterate: `P`, `n_B` and `eps` are
polynomials in the common quark chemical potential `mu`, and the three inverse
maps `mu(n_B)`, `mu(P)` and `mu(eps)` are closed forms as well.

The model is the `T = 0` analytic limit of the CFL phase of `eos/alphabag`, and
[Relation to eos/alphabag](#relation-to-eosalphabag-measured) states the
difference between the two — the `m_s^4` term this expansion drops — with the
measured number.

This file states every equation the code evaluates and every quantity it
returns. `abpr.tex` is the same document typeset, compiled against
`../../docs/eos.bib`; neither defers to the other. Every equation here is
asserted by `verify/run_full_check.py` or by `test/baseline`.


## Conventions

The public boundary is fm-based — `n` in fm^-3, `T` and every `mu` in MeV,
`eps` and `P` in MeV/fm^3, `s` in fm^-3 — with
`(hc)^3 = 7.6838e6 MeV^3 fm^-3` converting the natural-unit expressions once,
where the pressure is assembled.

**Strangeness is S = +1 per s quark**, the opposite of the PDG sign, used
consistently throughout this repository. `C` is the electric charge of
strongly-interacting matter only. Fractions are per baryon.


## The thermodynamic potential

In the CFL phase every quark pairs with a quark of a different colour and
flavour, and the condensate locks the three flavour densities together,

    n_u = n_d = n_s                                              (locking)

whatever the quark masses are (Alford, Rajagopal and Wilczek, NPB 537 (1999)
443). Because (locking) fixes the composition, the phase has a single
independent potential: the common quark chemical potential `mu`, related to the
baryon chemical potential by

    mu_B = 3 mu                                                     (muB)

Alford, Braby, Paris and Reddy write the free energy of this phase, to the
order at which each effect first appears, as four terms:

    P(mu) = 3 a4 mu^4/(4 pi^2 (hc)^3)         (i)   free gas + pQCD
            - 3 m_s^2 mu^2/(4 pi^2 (hc)^3)    (ii)  strange mass, to O(m_s^2)
            + 3 Delta^2 mu^2/(pi^2 (hc)^3)    (iii) CFL condensation energy
            - B/(hc)^3                        (iv)  the bag              (P)

The four terms are, in order:

**(i) The free quark gas with its perturbative correction.** Three massless
flavours at a common `mu`, each a gas of `g = 6` (three colours x two spins)
fermions, contribute `3 mu^4/(4 pi^2 (hc)^3)` — equation (massless) below. The
leading QCD correction to that pressure (Freedman and McLerran, PRD 16 (1977)
1169) multiplies it by `1 - 2 alpha_s/pi`, and the parametrization carries the
whole factor as one number,

    a4 = 1 - 2 alpha_s/pi,   equivalently   alpha_s = (pi/2)(1 - a4)    (a4)

so that `a4 = 1` is the free gas and `a4 < 1` softens it. Equation (a4) is the
*same knob* as the `alpha_s` of `eos/alphabag`, and it is the identity that
lets the two models be driven as a matched pair; the code exposes it as the
property `Parameters.alpha`.

**(ii) The strange quark mass**, to leading order. A free gas of mass `m` at
`T = 0` has less pressure than a massless one, and expanding the closed form
(massive) in `m/mu` gives

    P_m(mu) = mu^4/(4 pi^2 (hc)^3)
              - 3 m^2 mu^2/(4 pi^2 (hc)^3)
              + m^4/(8 pi^2 (hc)^3) [ 9/4 + 3 ln(2 mu/m) ]
              + O(m^6/mu^2)                                   (expansion)

Term (ii) of (P) is the `-3 m^2 mu^2/(4 pi^2 (hc)^3)` of (expansion) applied to
the strange flavour alone, the light flavours being treated as massless. The
`m_s^4` term is **not** carried: it is what separates this model from
`eos/alphabag`, and it is measured below.

**(iii) The condensation energy.** Pairing at a common gap `Delta` lowers the
free energy by `Delta^2 mu^2/pi^2` per flavour, hence `3 Delta^2 mu^2/pi^2` at
three locked flavours. This is the term that makes the phase competitive: it is
positive, it grows like the mass term it offsets, and `Delta > m_s/2` is
exactly the condition under which the combined `mu^2` coefficient of (P) turns
positive.

**(iv) The bag constant** `B`, the constant vacuum pressure that confines
(Chodos et al., PRD 9 (1974) 3471). It enters `P` with a minus and `eps` with a
plus, and neither `s` nor any `n_q`, so it cancels out of `eps + P` and the
Euler relation carries no bag term. It is carried as its fourth root `B^(1/4)`
in MeV, the form it is quoted in, so that a parameter set cannot hold a `B` and
a `B^(1/4)` that disagree.

Terms (ii) and (iii) share the same power of `mu`, and the code groups them,

    P(mu) = A mu^4 + C mu^2 - B/(hc)^3
    A = 3 a4/(4 pi^2 (hc)^3)
    C = 3 (Delta^2 - m_s^2/4)/(pi^2 (hc)^3)                         (PAC)


### The single-flavour thermodynamics the expansion comes from

Equation (expansion) is quoted above rather than cited, because a paper-style
description must be self-contained. At `T = 0` a gas of `g = 6` fermions of
mass `m` at chemical potential `mu > m` has Fermi momentum
`k_F = sqrt(mu^2 - m^2)` and

    n(mu)   = k_F^3/(pi^2 (hc)^3)
    P(mu)   = 1/(8 pi^2 (hc)^3) [ mu k_F (2 mu^2 - 5 m^2)
                                  + 3 m^4 ln((mu + k_F)/m) ]
    eps(mu) = 3/(8 pi^2 (hc)^3) [ mu k_F (2 mu^2 - m^2)
                                  - m^4 ln((mu + k_F)/m) ]
    s       = 0                                                 (massive)

which satisfy `eps + P = mu n` identically. At `m = 0` these reduce to

    n = mu^3/(pi^2 (hc)^3),  P = mu^4/(4 pi^2 (hc)^3),
    eps = 3P = 3 mu^4/(4 pi^2 (hc)^3)                          (massless)

and expanding (massive) in `m/mu` gives (expansion). These expressions live in
`eos.general.fermi_integrals` and are used by `eos/alphabag`; `eos/abpr` does
not call them, because (P) *is* their expansion — that is what the model is.

The finite-temperature Fermi integrals are not written here because this model
has no `T > 0` branch to write them for; that is the `cfl` mode of
`eos/alphabag`, whose document states them.


## Parameters

Four numbers, all of them arguments (`eos.abpr.Parameters`, a frozen
dataclass), never module-level constants:

| symbol | code | shipped value | meaning |
|---|---|---|---|
| `m_s`     | `m_s`    | 150 MeV | strange current quark mass |
| `Delta_0` | `Delta0` | 80 MeV  | CFL pairing gap, temperature independent here |
| `a4`      | `a4`     | 0.7     | pQCD factor, (a4); `alpha_s = 0.4712` |
| `B^(1/4)` | `B4`     | 135 MeV | bag constant; `B/(hc)^3 = 43.23 MeV/fm^3` |

The gap is written `Delta` in the equations above and `Delta_0` in this table
for the same reason `eos/alphabag` does: `Delta_0` is the zero-temperature gap
that is a parameter, `Delta` is its value at the temperature in hand, and here
the two are equal because there is no `Delta(T)`.

The derived quantities `alpha_s` and `B = (B^(1/4))^4` are properties of the
dataclass, so they cannot drift from the numbers they are computed from.
`Parameters.B` returns `B` in MeV^4 — the same unit, for the same attribute, as
`eos.alphabag.Parameters.B` and `eos.vmit.Parameters.B` — and the one division
by `(hc)^3` happens where the pressure is assembled.

There is no published single ABPR parameter set: the four numbers span a range
(`a4` in [0.6, 1], `Delta_0` up to about 200 MeV, `B^(1/4) ~ 130-180` MeV,
`m_s ~ 90-250` MeV) that a hybrid-star study scans, and which point in it is
right depends on the hadronic model the quark phase is paired with. The values
above are `Parameters.default()`, the set this repository's numerical baseline
is frozen at, and a set with some of them changed is
`Parameters(a4=..., B4=...)` or `dataclasses.replace` of one already in hand.

**Where the parametrization is valid.** The expansion behind term (ii) needs
`m_s << mu`, and the pairing term needs `Delta << mu`; both are corrections of
relative order 1e-1 at the densities of a compact-star core and neither is a
controlled expansion at the surface. The gap is taken constant — there is no
`Delta(T)` here, because there is no `T` — and it is imposed, never solved for.


**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist. *By name:*
`Parameters.default()` is the working set above, and
`Parameters.named('abpr_default')` takes it by name. ABPR ships exactly one
set, so the map has a single entry; it exists so that a caller sweeping
parameter sets need not know which models happen to have more than one. *A new
set:* every field carries a default, so `Parameters(a4=..., B4=...)` names only
what changes; the dataclass is frozen, so `dataclasses.replace` is how a set
already in hand is modified. *From nuclear-matter parameters:* no route, and
none is missing -- ABPR has no nuclear sector, so there is no `nmp.py` and
nothing to invert. The inverse maps of the section below are a different
object entirely: they invert this model's own STATE VARIABLES -- `mu` from
`n_B`, `mu` from `P` -- not its parameters.

## Everything else, as derivatives of P

Since the composition is fixed by (locking), `P(mu)` is the whole model and
every other quantity is a derivative of it. With `mu_B = 3 mu`,

    n_B(mu) = dP/dmu_B = (1/3) dP/dmu
            = a4 mu^3/(pi^2 (hc)^3)
              + 2 (Delta^2 - m_s^2/4) mu/(pi^2 (hc)^3)
            = a mu^3 + c mu                                          (nB)

    s(mu)   = dP/dT = 0                                               (s)

    eps(mu) = -P + mu_B n_B = -P + 3 mu n_B
            = 3A mu^4 + C mu^2 + B/(hc)^3                            (eps)

    f(mu)   = eps - T s = eps = -P + mu_B n_B                          (f)

with `a = 4A/3` and `c = 2C/3` in the notation of (PAC). Two of these are worth
reading twice. Equation (s) is not an omission: the model is defined at `T = 0`,
where the entropy of any of its sectors vanishes, and `s = 0` is a value the
code returns, not a quantity it declines to compute. Equation (eps) is the
Euler relation itself, so

    eps + P = T s + sum_q mu_q n_q = 3 mu n_B = mu_B n_B           (euler)

holds by construction; `verify/` checks it all the same, and it holds to 3e-16
relative. Note the middle equality: the three flavour potentials are equal
here, so `sum_q mu_q n_q = 3 mu n_B` with no `mu_S` term. The bag has cancelled
out of (euler) — its `+B/(hc)^3` in `eps` against its `-B/(hc)^3` in `P` — which
is the arithmetic statement of the fact that a constant vacuum pressure carries
no charge and no entropy.

There is no scalar density in this model and no identity `n_s = (eps - 3P)/m*`
to state: the masses are parameters and there is no gap equation, so nothing
plays the part `m*` plays in a mean-field model. The `n_s` a point returns is
the strange-quark number density, equal to `n_B` by (locking).

The speed of sound follows from (PAC) and (eps) without any numerical
differentiation,

    c_s^2 = (dP/dmu)/(deps/dmu)
          = (4A mu^2 + 2C)/(12A mu^2 + 2C)
          = (2A mu^2 + C)/(6A mu^2 + C)  ->  1/3  as mu -> infinity  (cs2)

the conformal limit being reached from above when `C > 0` (a gap large enough
that `Delta > m_s/2`) and from below when `C < 0`. At the shipped set `c_s^2`
falls monotonically from 0.3397 at the surface to 0.3339 at `mu = 900` MeV:
this parametrization is always close to conformal, which is the property that
makes it soft.


## Conserved charges

With `n_u = n_d = n_s = n_B` and the quantum numbers of `eos.general.basis` —
and `S = +1` per `s` quark — the conserved-charge densities are

    n_B = (n_u + n_d + n_s)/3
    n_C = (2 n_u - n_d - n_s)/3 = 0
    n_S = n_s = n_B                                             (charges)

so

    Y_C = n_C/n_B = 0,      Y_S = n_S/n_B = +1                (fractions)

identically, at every density and for every parameter set. This is the single
most consequential fact about the model and the mode table below is a
consequence of it: *the CFL phase is electrically neutral by construction*,
with no leptons of any kind, and its strangeness is maximal. `Y_C` here is the
non-leptonic charge fraction of this repository's convention, and since it
vanishes, total electric neutrality — the separate condition `n_C = n_e + n_mu`
— is satisfied with `n_e = n_mu = 0`. The two coincide only because both are
zero.

The conserved-charge potentials follow from `mu_u = mu_d = mu_s = mu` through
the inverse of the map in `eos.general.basis`:

    mu_B = mu_u + mu_d + mu_s - mu_S = 3 mu
    mu_C = mu_u - mu_d = 0
    mu_S = mu_s - mu_d = 0                                   (potentials)

That `mu_S = 0` is a *choice of this parametrization*, not a property of the
CFL phase: locking equal densities at unequal masses would need unequal
potentials, and `eos/alphabag` solves for exactly that, finding
`mu_S ~ 23-48` MeV over `n_B = 0.3-3 fm^-3` at the matched parameters below.
Here that difference has instead been absorbed into term (ii) of (P), which is
why the energy per baryon at the surface is `mu_B` and not `mu_B + mu_S`. Both
bookkeepings are internally consistent — each satisfies (euler) in its own
variables — and they differ by the amount measured below.


## The closure, and the four modes

The composition is not solved for; it is locked by (locking). There is
therefore exactly one request the model answers, and following `eos/alphabag`
it carries the phase's own name rather than an equilibrium's:

| mode | independent variables | what closes the system |
|---|---|---|
| `cfl` | `(n_B, T = 0)` | flavour locking `n_u = n_d = n_s`, (locking) |

The gap does not appear as a per-call condition, as it does in `eos/alphabag`,
because in the ABPR parametrization `Delta` is fitted alongside `a4`, `m_s` and
`B` and belongs with them in the parameter set; carrying it in both places
would be two homes for one number.

The "solve" behind `solve_cfl(par, n_B)` is the inversion of (nB), given in closed
form below.

**What the other four modes mean here, and why each raises.** The repository's
four modes fix the independent variables of an equilibrium. None of them is
reachable in this phase, and the reason is the physics of (locking) rather than
an unwritten feature. In each case the code raises `NotImplementedError` naming
the reason:

| mode | why it has no state here |
|---|---|
| `beta_eq_neutrinoless` | Beta equilibrium fixes the charge potential through `mu_C + mu_e = 0`. Locking has already fixed the composition and left `mu_C = 0`, (potentials), with no electrons to equilibrate against, so the condition has no free variable to determine. Unpaired quark matter in beta equilibrium is `eos.alphabag` or `eos.vmit`. |
| `beta_eq_neutrino_trapped` | The same, plus: a trapped-neutrino mode fixes the lepton fraction `Y_Le`, and the phase carries no leptons at all, so `Y_Le` has nothing to fix. |
| `fixed_YC` | `Y_C = 0` identically, (fractions). A request for any other `Y_C` asks for a state the phase does not have, and a request for `Y_C = 0` is the `cfl` mode itself. |
| `fixed_YC_YS` | The same for `Y_C`, and `Y_S = +1` identically as well. Both fractions are outputs of the closure, not inputs to it. In particular the symmetric-nuclear-matter slice `Y_C = 0.5`, `Y_S = 0` that this mode exists for is not a state of deconfined locked matter. |

Two further restrictions are stated the same way. **Temperature:** (P) is a
`T = 0` expression and `T > 0` raises; the finite-`T` CFL phase, with its BCS
gap `Delta(T)` and its thermal sectors, is the `cfl` mode of `eos/alphabag`.
Wherever this repository accepts an entropy axis in place of a temperature,
`s/n_B = 0` is the only value this model reaches, by (s). **Species:** every
optional sector of the repository's uniform flag list — hyperons, deltas and
thermal mesons (hadronic sectors, meaningless in a deconfined phase), muons (no
leptons here), photons, gluons and thermal neutrinos (thermal sectors,
identically zero at `T = 0`) — is off, and switching one on raises rather than
being silently ignored.


## The inverse maps

A table is asked for at a given `n_B`, a stellar-structure integration at a
given `P`, and a strange-star surface at `P = 0`. All three inversions are
closed forms, so this model iterates nowhere and has no convergence question to
report: the status a solved point carries is the residual of the closed form,
which is at round-off.

**`mu` from `n_B`.** Equation (nB) is a cubic in `mu` with no quadratic and no
constant term,

    a mu^3 + c mu - n_B = 0                                       (cubic)

already in depressed form, so Cardano applies directly. With `p = c/a`,
`q = -n_B/a` and discriminant `D = (q/2)^2 + (p/3)^3`,

    mu = u - p/(3u),   u = cbrt(-q/2 + sqrt(D))       (D >= 0)  (cardano)

and for `D < 0` the three real roots are

    mu_k = 2 sqrt(-p/3) cos( (1/3) arccos( 3q/(p m) ) - 2 pi k/3 ),
    m = 2 sqrt(-p/3),   k = 0, 1, 2

of which the largest is taken. The second form in (cardano) — `-p/(3u)` rather
than the textbook `cbrt(-q/2 - sqrt(D))` — avoids the cancellation that costs
digits when `|p|` is small, and takes the residual of (cubic) at
`n_B = 3 fm^-3` from 1.4e-11 to 8.9e-16 fm^-3.

The root taken is always the physical one: for `n_B > 0` the sign sequence of
(cubic) has exactly one change whatever the sign of `c`, so by Descartes' rule
there is exactly one positive root. When `Delta < m_s/2` the coefficient `c` is
negative and `n_B(mu)` is negative below `mu = sqrt(-c/a)`; the positive root is
the one above that crossing, and it is the one (cardano) returns.

**`mu` from `P` and from `eps`.** Both are quadratics in `mu^2`, by (PAC) and
(eps):

    mu(P)   = sqrt( ( -C + sqrt( C^2 + 4A (P + B/(hc)^3) ) ) / (2A) )
    mu(eps) = sqrt( ( -C + sqrt( C^2 + 12A (eps - B/(hc)^3) ) ) / (6A) )
                                                            (quadratics)

The `+` branch is the physical one in both: it is the larger root in `mu^2`,
and when `C < 0` — the only case in which the smaller root is also positive —
it is the branch on which `n_B` and `dP/dmu` are positive.

**Measured against iteration.** These closed forms replaced three
`scipy.optimize.root` calls. Over the parameter sets and targets of
`test/baseline` they agree with the iterative answers to 2.5e-13 relative at
worst, which is inside the 1e-10 the baseline is frozen at; the largest
disagreement anywhere tested, at `n_B = 3 fm^-3`, is 1.6e-12.


## The P = 0 surface

A self-bound phase has a surface: the density at which its pressure vanishes,
where a bare strange star ends with no crust (Witten, PRD 30 (1984) 272). It is
(quadratics) at `P = 0`, and by (euler) the energy per baryon there is

    (eps/n_B)|_{P=0} = mu_B = 3 mu_0,     P(mu_0) = 0                (EA)

At the shipped set `mu_0 = 277.195` MeV, so the surface sits at
`n_B = 0.2023 fm^-3` with `eps = 168.21 MeV/fm^3` and

    E/A = 831.58 MeV

below the 930 MeV of Fe-56: the shipped set describes *absolutely stable*
strange quark matter in the sense of the Bodmer-Witten hypothesis. The pairing
term is what buys that. Switching it off (`Delta = 0`) with everything else
unchanged moves the surface to `mu_0 = 310.98` MeV, `n_B = 0.2315 fm^-3` and
`E/A = 932.94` MeV, which is above iron and therefore not absolutely stable.


## Relation to eos/alphabag, measured

`eos/abpr` is the `T = 0` analytic limit of the CFL phase of `eos/alphabag`,
and studies in this repository drive the two as a matched pair. The parameters
map exactly:

    m_s -> m_s,   Delta -> Delta0,   B^(1/4) -> B^(1/4),
    alpha_s = (pi/2)(1 - a4),        m_u = m_d = 0              (matching)

They are nonetheless *not the same expression*, and the difference is exactly
one term. `eos/alphabag` carries the strange quark mass exactly, through the
Fermi integrals of (massive); `eos/abpr` carries it as the `O(m_s^2)` term (ii)
of (P). So the `m_s^4` term of (expansion), which `eos/alphabag` has and this
model does not, is the whole of the gap between them:

    dP = P_ABPR(mu) - P_alphaBag_CFL(mu)
       ~ -m_s^4/(8 pi^2 (hc)^3) [ 9/4 + 3 ln(2 mu/m_s) ]
         + O(m_s^6/mu^2)                                            (gap)

(Adding term (ii) to `eos/alphabag` as well would count the strange mass twice,
which is why its `cfl` phase deliberately omits it.)

Equation (gap) is measured, not asserted. At the shipped set, with
`eos/alphabag` evaluated at three equal potentials `mu_u = mu_d = mu_s = mu` so
that the two closures are compared on the same variable:

| `mu` [MeV] | `n_B` [fm^-3] | `dP` [MeV/fm^3] | `dP/P` | (gap) | ratio |
|---|---|---|---|---|---|
| 350 | 0.403 | -5.694 | -8.1e-2 | -5.734 | 0.9931 |
| 400 | 0.599 | -6.038 | -4.2e-2 | -6.068 | 0.9950 |
| 500 | 1.164 | -6.608 | -1.6e-2 | -6.627 | 0.9971 |
| 600 | 2.006 | -7.070 | -8.1e-3 | -7.083 | 0.9981 |
| 700 | 3.181 | -7.460 | -4.5e-3 | -7.469 | 0.9987 |
| 800 | 4.743 | -7.796 | -2.8e-3 | -7.804 | 0.9991 |

The last column is the check: the measured difference is the analytic `m_s^4`
term to within 0.7% at `mu = 350` MeV and 0.1% at `mu = 800` MeV, and the
shortfall itself falls like the `O(m_s^6/mu^2)` (gap) predicts. The two models
agree to exactly the order at which they are supposed to differ.
`verify/run_full_check.py` asserts this, requiring
`|dP/dP_(gap) - 1| < 1e-2` over `mu` in [350, 800] MeV at the shipped set.

Compared instead at matched `n_B` — the way a table pairs them — the same
difference reads, at the shipped set,

    dP/P     : -7.9e-2 -> -2.8e-3
    deps/eps :  1.3e-2 ->  1.1e-3        over n_B = 0.3 -> 3 fm^-3

the abpr pressure being the lower of the two throughout and its energy density
the higher. A study that needs the strange mass better than a percent at the
lowest densities should use `eos/alphabag`; one that needs a closed form should
use this.


## What a solved point returns

`solve_cfl` returns a `CFLPoint` — the same record name and the same fields as
the paired point of `eos/alphabag`, so a caller comparing the two reads one
layout. Every quantity in it is listed here, with the equation it comes from:

| field | symbol | from |
|---|---|---|
| `converged`, `error` | — | the residual of the closed-form inverse |
| `n_B`, `T` | `n_B`, `T = 0` | the request |
| `Delta0`, `Delta` | `Delta_0`, `Delta` | the parameter; equal, there being no `Delta(T)` |
| `mu_u`, `mu_d`, `mu_s` | `mu` | (cardano), all three equal |
| `mu_B`, `mu_C`, `mu_S` | `3 mu`, 0, 0 | (potentials) |
| `mu_e`, `mu_nu` | 0 | no leptons |
| `n_u`, `n_d`, `n_s` | `n_B` | (locking) |
| `P` | `P` | (P) |
| `eps` | `eps` | (eps) |
| `s` | `s = 0` | (s) |
| `f` | `f = eps` | (f) |
| `Y_u`, `Y_d`, `Y_s` | 1 each | (locking) |
| `Y_C`, `Y_S` | 0, +1 | (fractions) |

Unlike `eos.alphabag.CFLPoint`, this record carries no `Y_e` and no `Y_nu`:
there is no lepton sector to report a fraction of.


## The API surface

Three entry points, with the signatures every model in this repository carries:

    eos_point(par, mode="cfl", species, n_B=, T=0.0, SnB=, **conditions)
    eos_table(par, mode="cfl", species, axes=, rows=, progress=, verbose=)
    eos_response(par, mode="cfl", species, frozen="equilibrium", n_B=, T=0.0,
                 **conditions)

`par` comes first and is never optional. `eos_point` returns a result whose
`.ok` must be tested before `.point` is read; a request outside the phase — a
pressure below `-B`, an energy density below the bag — comes back as a status,
not an exception.

`eos_table` takes `axes={'nB': grid, 'T': [0.0]}`, and the temperature axis may
be omitted since `T = 0` is the only value the model has. There is no warm
start and no bisected continuation, and their absence is the physics rather
than a gap: the density inverse is the closed form (cardano), so no point needs
its neighbour and the grid is evaluated by array arithmetic. Nor is there a
`skip_errors` flag: a request outside the phase is a property of the target,
not of a solve that might have gone better from a different start. The
`progress` callback is invoked once per completed line — there is one — with
the dictionary every table builder in this repository reports,

    {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
     elapsed_s}

and `verbose=True` installs the built-in printer.

`eos_response` implements one freeze, `frozen="equilibrium"`, which is the only
conditioning this phase admits: flavour locking holds the composition at every
density, so "what is held fixed while the derivative is taken" has one answer
and the named freezes of the other models (`fast`, `slow`) would expand to the
same set. It returns

    {'cs2_isothermal': c_s^2}

from (cs2), differentiated analytically rather than by a stencil. At `T = 0`
the isothermal and adiabatic sound speeds coincide; the returned name says
which convention the number was computed under rather than leaving it to the
arguments. The heat capacities `C_V` and `C_P`, the thermal index and the
adiabatic index are not defined at `T = 0` and are not returned. The
susceptibilities `chi_ab = dn_a/dmu_b` are singular here, flavour locking
leaving `n_C` and `n_S` with no potential to respond to, and are not returned
either.


## Numerics

There are none to speak of, and that is the point of this model: no root find,
no integral, no warm start, no iteration count and no tolerance. Every public
function is a polynomial or a root of one, so a table is evaluated by array
arithmetic rather than by a loop over solved points, and a parameter scan costs
microseconds per point. The convergence status every public entry point carries
is nonetheless real: it reports the residual of the closed-form inverse against
the equation it inverts, judged on the same scaled gate
(`eos.general.solve.RESIDUAL_TOL = 1e-10`) as every other model in this
repository, so "converged" means the same thing here as it does in a model that
iterates.


## Layout

    parameters.py       Parameters, its default set, alpha_s and B
    species.py          SpeciesFlags -- every sector off, and setting one raises
    thermodynamics.py   P, n_B, eps, s, f, c_s^2 from mu
    solver.py           the inverse maps, solve_cfl, CFLPoint, the modes
    api.py              eos_point / eos_table / eos_response
    verify/             the invariants above, one entry point


## The self-bound surface, and the two-flavour arm

### What is computed

A parametrization whose pressure crosses zero at finite density describes
**self-bound** matter: the phase ends there, with no crust below it. The
quantity reported at that endpoint is the energy per baryon,

    E/A = eps / n_B      at   P(n_B) = 0,  T = 0      [MeV]

which is what a lump of this matter at rest weighs per baryon. The entry point
is `zero_pressure_point(par, species)`, and it returns a `ZeroPressurePoint`
carrying `n_B`, `E_per_A`, `mu_B`, `Y_S`, `mu_S`, the identity residual below,
the pressure actually reached, the flavour content requested, and whether
`E_per_A` fell below the 930.4 MeV of iron.

### The identity that makes the read self-checking

At T = 0 the Euler relation is

    eps + P = sum_i mu_i n_i ,

so at P = 0 the energy per baryon IS the Gibbs energy per baryon. Expanding
the species potentials in the conserved-charge basis, mu_i = B_i mu_B +
C_i mu_C + S_i mu_S, and using beta equilibrium (mu_C + mu_e = 0) together with
total electric neutrality (n_C = n_e), the charge term and the lepton term
cancel exactly:

    sum_i mu_i n_i = mu_B n_B + mu_C n_C + mu_S n_S + mu_e n_e
                   = mu_B n_B + mu_S n_S          (since mu_C n_C + mu_e n_e = 0)

and therefore

    E/A = mu_B + Y_S mu_S ,      Y_S = n_S / n_B .            (*)

**The full form is the one to use.** `E/A = mu_B` is the special case
Y_S mu_S = 0, which holds in every beta-equilibrium mode because strangeness
self-equilibrates there and mu_S = 0. It does NOT hold in a colour-flavour
locked phase, where the condensate pairs the three flavours at equal densities
and unequal masses force unequal potentials: on the CFL surface of `eos.alphabag`
at Delta_0 = 100 MeV, mu_S = 40.68 MeV, and mu_B alone gives 895.87 MeV where
E/A is 936.55 MeV. There are not two conventions here, only one identity and
the cases in which a term of it drops out.

`(*)` is checked at every located root; a root that misses it is a root of
something other than P.

### How the root is found

`eos.general.zero_pressure.locate_zero_pressure` samples P(n_B) on a grid,
takes the LOWEST density at which P rises through zero, and refines it by
Brent's method. The scan is not a convenience: P(n_B) can cross zero more than
once, and a crossing where P FALLS is the top of a mechanically unstable
region rather than a surface. It takes the state as a callable, so it holds no
model and lives in `general/`; a density where the solve does not converge
thins the scan rather than aborting it, and a set with no surface at all comes
back as a status, never as an exception.

### The Bodmer-Witten window

The pair of numbers is a two-sided gate on a parameter set:

| arm | condition | what it says |
|---|---|---|
| three-flavour | `E/A < 930.4 MeV` | strange quark matter is absolutely stable |
| two-flavour | `E/A > 930.4 MeV` | ordinary nuclei are not already decaying into it |

A set failing either is excluded. **Both facts are REPORTED and neither is
asserted**: whether a set sits in the window is a property of the set, so
`below_iron` is a field on the result and no `verify/` entry fails on it. Note
that the same `below_iron = True` reads in opposite directions on the two arms.

**The content requested and the content found can differ, and the result
carries both.** A three-flavour request returns whatever strangeness the
equilibrium actually populated: a set whose surface sits below the s quark's
threshold returns `Y_S = 0` and the two-flavour number from the three-flavour
call. Read the content off `Y_S`, never off `two_flavour`.

### There is no two-flavour arm

`cfl` is the only mode this model has, and colour-flavour locking fixes
Y_S = +1 identically: no strangeness fraction is free to switch off. The
two-flavour number therefore **does not exist** here, rather than being
unimplemented, and `SpeciesFlags(two_flavour=True)` **raises** saying so
instead of the entry point returning a nan. This is the statement the model
already makes about `gluons`, one sector further on — a fact about which phase
the model describes rather than a choice the caller has. The two-flavour half
of the window is asked of a model with an unpaired phase: `eos.vmit`,
`eos.alphabag`, `eos.njl` or `eos.ccdm`, in `beta_eq_neutrinoless`.

### The locator against the closed form

This model inverts P(mu) analytically, so `mu_from_P(0, par)` gives the
surface exactly and `zero_pressure_point` does not need a root find. It uses
the shared bracketed locator anyway, deliberately: ABPR is the only model in
the repository where both routes exist, which makes it the only place the
locator can be measured against an exact answer rather than against itself.
`verify/run_full_check.py` runs the two side by side; they agree to 5e-16, and
the shipped set gives

    E/A = 831.58 MeV      at   n_B = 0.2023 fm^-3,

below the 930.4 MeV of iron: absolutely stable strange quark matter for this
parametrization. That number is a golden reference (CLAUDE.md section 12).

## References

- M. Alford, M. Braby, M. Paris and S. Reddy, Astrophys. J. 629, 969 (2005) —
  the parametrization this model is.
- M. Alford, K. Rajagopal and F. Wilczek, Nucl. Phys. B 537, 443 (1999) — CFL
  and flavour locking.
- B. A. Freedman and L. D. McLerran, Phys. Rev. D 16, 1169 (1977) — the
  O(alpha_s) correction behind `a4`.
- A. Chodos et al., Phys. Rev. D 9, 3471 (1974) — the bag.
- E. Witten, Phys. Rev. D 30, 272 (1984) — the strange-matter hypothesis and
  the bare surface.
