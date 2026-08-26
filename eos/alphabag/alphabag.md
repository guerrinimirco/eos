# alphaBag — the perturbatively corrected bag model, unpaired and colour-flavour locked

Deconfined `u, d, s` quark matter inside a bag, with the leading perturbative
QCD correction carried as one constant coupling `alpha_s` multiplying the
free-gas pressure, in the arrangement of Fischer et al., ApJS 194, 39 (2011).
The strange quark is massive and gets the exact Fermi gas; the light flavours
are massless and close in elementary functions. Thermal gluons are an optional
sector with their own `alpha_s` correction. On top of the unpaired phase the
package carries a second phase, colour-flavour locked (CFL) quark matter
(Alford, Rajagopal and Wilczek, NPB 537 (1999) 443), obtained by adding the
pairing term of Alford, Braby, Paris and Reddy, ApJ 629, 969 (2005) to the same
potential and closing the system with flavour locking `n_u = n_d = n_s` instead
of with an equilibrium condition.

This file states every equation the code solves and every quantity it returns.
`alphabag.tex` is the same document typeset, compiled against
`../../docs/eos.bib`; neither defers to the other. Every equation here is
asserted by `verify/run_full_check.py` or by `test/baseline`.

There is no vector field — that is `eos/vmit`, and the two quark models stiffen
the equation of state by different mechanisms rather than being
reparametrisations of one another — and there is no gap equation: the quark
masses are parameters, not solutions, so the potential is explicit in `mu` and
the only solving to be done is that of the equilibrium conditions themselves.


## Conventions

Natural units inside the physics; the public boundary is fm-based — `n` in
fm^-3, `T` and every `mu` in MeV, `eps` and `P` in MeV/fm^3, `s` in fm^-3 —
with `(hc)^3 = 7.6835057e6 MeV^3 fm^3` applied once, at that boundary. Below,
`(hc)^3` written in a formula is exactly that conversion.

**Strangeness is S = +1 per s quark**, the opposite of the PDG sign, used
consistently throughout this repository. `C` is the electric charge of
strongly-interacting matter only: the leptons are excluded from it and enter
through the separate condition of total electric neutrality. Fractions are per
baryon, `Y_C = n_C/n_B` and `Y_S = n_S/n_B`.


## The thermodynamic potential

Uniform deconfined matter at flavour potentials `mu_u, mu_d, mu_s` and
temperature `T` is a gas of quarks and antiquarks confined by a constant vacuum
pressure `B`. The pressure of the strongly-interacting sector is

    P   = sum_q P_q(mu_q, T, m_q, alpha) + P_g(T, alpha) - B/(hc)^3      (P)

and, since the bag constant is a constant shifted out of the vacuum energy,

    eps = sum_q eps_q + eps_g + B/(hc)^3
    s   = sum_q s_q + s_g
    n_q = n_q(mu_q, T, m_q, alpha)                                     (eps)

with `q` running over `u, d, s`. The bag enters `eps` and `P` with opposite
signs and enters neither `s` nor any `n_q`, so it cancels out of `eps + P` and
the Euler relation holds with no bag term in it,

    eps + P = T s + sum_q mu_q n_q                                   (euler)

which is checked in `verify/` and holds to 4e-16 relative. The free energy
density is `f = eps - T s = -P + sum_q mu_q n_q`.


### The perturbative correction

The correction is the O(alpha_s) term of the free energy of a relativistic
quark-gluon gas (Freedman and McLerran, PRD 16 (1977) 1169), kept as two
multiplicative factors — one on the quark thermal term, one on everything
carrying a chemical potential — plus one for the gluons:

    c_q(a) = 1 - 2a/pi
    c_T(a) = 1 - 50a/(21 pi)
    c_g(a) = 1 - 15a/(4 pi)                                      (cfactors)

in the arrangement of Fischer et al. At the shipped `alpha_s = 0.3` these are
`c_q = 0.8090`, `c_T = 0.7726`, `c_g = 0.6419`. There is no running:
`alpha_s` is a constant of the parameter set, which is what makes it a
parameter an inference run can vary.


### Parameters

| symbol | code | value | meaning |
|---|---|---|---|
| `m_u`     | `m_u`   | 0 MeV | up-quark mass, treated as massless |
| `m_d`     | `m_d`   | 0 MeV | down-quark mass, treated as massless |
| `m_s`     | `m_s`   | 150 MeV | strange-quark mass |
| `alpha_s` | `alpha` | 0.3 | QCD coupling, constant |
| `B^(1/4)` | `B4`    | 165 MeV | bag constant, quoted as its fourth root |
| `B`       | `B`     | 7.412e8 MeV^4 | `= (B^(1/4))^4 = 96.466 MeV/fm^3` |
| `c_Tc`    | `tc_coeff` | `0.57 * 2^(1/3) = 0.71815` | CFL critical temperature as a multiple of `Delta0`; read by the `cfl` mode only |

`B` is a derived property of the parameter record, never stored, so a set
cannot carry a `B` and a `B^(1/4)` that disagree. The defaults are a central
choice within the ranges used for compact-star quark matter (Alford and Reddy,
PRD 67 (2003) 074024; Fischer et al.): `B^(1/4) ~ 145-180` MeV,
`alpha_s ~ 0.2-0.5`, `m_s ~ 90-150` MeV. They are named defaults and not
hardcoded values, and every entry point takes the parameter record as its first
argument.

The pairing gap `Delta0` of the CFL phase is deliberately **not** a field of
the parameter record. It selects a phase rather than tuning one, in the same
way the species flags do, and is passed per call.

For orientation, the shipped set at `T = 0` in charge-neutral beta equilibrium
reaches `P = 0` — the surface of a self-bound star — at
`n_B = 0.40309 fm^-3` with `eps/n_B = 1046.17` MeV, so this set is not
absolutely stable strange quark matter. Pairing lowers it: the CFL phase at
`Delta0 = 100` MeV has its surface at `n_B = 0.36286 fm^-3` with
`eps/n_B = 936.56` MeV.


**Three routes to a parameter set.** CLAUDE.md section 6 makes model
parameters arguments, so all three have to exist. *By name:*
`Parameters.default()` is the working set above, and
`Parameters.named('alphabag_default')` takes it by name. alphaBag ships exactly
one set, so the map has a single entry; it exists so that a caller sweeping
parameter sets need not know which models happen to have more than one. *A new
set:* every field carries a default, so `Parameters(alpha=..., B4=...)` names
only what changes; the dataclass is frozen, so `dataclasses.replace` is how a
set already in hand is modified. *From nuclear-matter parameters:* no route,
and none is missing -- alphaBag has no nuclear sector, so there is no `nmp.py`
and nothing to invert.

## Single-flavour thermodynamics

Each flavour is a gas of quarks and antiquarks of degeneracy
`g = 2 x 3 = 6` (spin x colour). Two cases are implemented and the choice is
made on the mass, `m < 1e-5` MeV selecting the massless branch.

### Massless flavours

For `m = 0` the Fermi integrals close in elementary functions, and with the
factors of (cfactors) folded in the flavour carries

    P_0(mu,T,a)   = [ (7/60) pi^2 T^4 c_T(a)
                      + ( T^2 mu^2/2 + mu^4/(4 pi^2) ) c_q(a) ] / (hc)^3   (P0)
    n_0(mu,T,a)   = ( mu T^2 + mu^3/pi^2 ) c_q(a) / (hc)^3                 (n0)
    s_0(mu,T,a)   = [ (7/15) pi^2 T^3 c_T(a) + T mu^2 c_q(a) ] / (hc)^3    (s0)
    eps_0(mu,T,a) = 3 P_0(mu,T,a)                                          (e0)

At `alpha_s = 0` these are the textbook massless Fermi gas at degeneracy 6 with
antiparticles. They are a consistent set: `n_0 = dP_0/dmu` and `s_0 = dP_0/dT`
hold identically (verified against numerical derivatives to 1e-10 relative),
and `eps_0 + P_0 = 4 P_0 = T s_0 + mu n_0` term by term, so (euler) holds for a
massless flavour at any `alpha_s`. This is why the correction is applied to
`P`, `n`, `eps` and `s` separately rather than to `P` alone: applying it to `P`
and taking the others as derivatives would give the same answer, and the code
writes all four out for readability.

At `T = 0` equations (P0)-(e0) reduce to

    P_0 = mu^4 c_q / (4 pi^2 (hc)^3),   n_0 = mu^3 c_q / (pi^2 (hc)^3),
    s_0 = 0,                            eps_0 = 3 P_0

### Massive flavours

For `m > 0` the kinetic thermodynamics is the exact Fermi gas,

    n_F   = g/(2 pi^2 (hc)^3) int_0^inf dk k^2 [ f(E_k - mu) - f(E_k + mu) ]
    P_F   = g/(6 pi^2 (hc)^3) int_0^inf dk (k^4/E_k)
                                        [ f(E_k - mu) + f(E_k + mu) ]
    eps_F = g/(2 pi^2 (hc)^3) int_0^inf dk k^2 E_k
                                        [ f(E_k - mu) + f(E_k + mu) ]
    s_F   = ( eps_F + P_F - mu n_F ) / T                                 (sF)

with `E_k = sqrt(k^2 + m^2)`, `f(x) = [1 + exp(x/T)]^-1` and `g = 6`. At
`T = 0` the antiparticle terms vanish, the occupations become step functions at
`k_F = sqrt(mu^2 - m^2)` (with `n_F = 0` when `mu <= m`), and

    n_F   = g k_F^3 / (6 pi^2 (hc)^3)
    eps_F = g/(16 pi^2 (hc)^3) [ k_F (2 k_F^2 + m^2) E_F
                                 - m^4 ln( (k_F + E_F)/m ) ]
    P_F   = g/(48 pi^2 (hc)^3) [ k_F (2 k_F^2 - 3 m^2) E_F
                                 + 3 m^4 ln( (k_F + E_F)/m ) ]
    s_F   = 0

with `E_F = sqrt(k_F^2 + m^2) = mu`. These integrals are not implemented in
this subpackage: they come from `eos.general.fermi_integrals`, which evaluates
them through the Johns-Ellis-Lattimer analytic approximation (ApJ 473 (1996)
1020), uniformly valid from the degenerate to the non-degenerate limit and
exact at `T = 0`. They are written out here all the same, because a paper-style
description must be self-contained.

The perturbative correction to a massive flavour is taken to be the same
function of `mu` and `T` as for a massless one:

    X(mu,T,m,a) = X_F(mu,T,m) + [ X_0(mu,T,a) - X_0(mu,T,0) ],
    X in {n, P, eps, s}                                             (massive)

where the bracket is `-(2 a/pi)` times the `mu`-dependent part of `X_0` and
`-(50 a/(21 pi))` times its thermal part. This is a prescription, not an
expansion of the massive result: the true O(alpha_s) correction to a gas of
mass `m` differs from (massive) at relative order `m^2/mu^2`, which for
`m_s = 150` MeV at `mu_s ~ 440` MeV is a ~12% correction of a ~19% correction.
What the prescription does guarantee is that (i) it reduces exactly to
(P0)-(e0) as `m -> 0`, and (ii) the correction is itself a consistent set —
`dn = d(dP)/dmu`, `ds = d(dP)/dT`, `deps = 3 dP` — so adding it to a free Fermi
gas that satisfies (euler) leaves a sector that still does.

### Gluons

Thermal gluons are `g_g = 2 x 8 = 16` massless bosons at `mu = 0`, with their
own correction factor:

    P_g   = (8 pi^2/45) T^4 c_g(a) / (hc)^3
    eps_g = 3 P_g
    s_g   = (32 pi^2/45) T^3 c_g(a) / (hc)^3
    n_g   = 0                                                        (gluons)

They satisfy `eps_g + P_g = 4 P_g = T s_g` and carry no conserved charge, so
switching them on shifts `P`, `eps` and `s` and nothing else. They vanish
identically at `T = 0`. `gluons` is this model's own sector flag: no other
model in the repository has one.


## The colour-flavour locked phase

At high density the favoured ground state of three-flavour quark matter is the
CFL condensate, in which every quark pairs and the common gap `Delta` enters
the pressure at order `Delta^2 mu^2`. The package implements this as a second
phase, not as a further correction to the first: it has its own potential, its
own solver, and its own closure.

### The gap

The gap is not solved for. It is imposed as a BCS-shaped function of
temperature with a single parameter, the zero-temperature gap `Delta0`:

    Delta(T) = Delta0 sqrt(1 - T^2/T_c^2)   for T < T_c, else 0
    T_c      = c_Tc Delta0 ,   c_Tc = 0.57 * 2^(1/3) = 0.71815          (gap)

so that `Delta0 = 100` MeV gives `T_c = 71.815` MeV. `c_Tc` is `tc_coeff` of
the parameter record, not a constant of the code: an inference run over CFL
pairing varies it like any other parameter, and the shipped value is the
weak-coupling BCS result with its colour factor. The entropy needs the
derivative, which follows by differentiating (gap):

    dDelta/dT = - Delta0 T / ( T_c^2 sqrt(1 - T^2/T_c^2) ),   T < T_c  (dgap)

and zero at `T = 0` and `T >= T_c`. Equation (dgap) diverges as `T -> T_c^-`;
the code returns zero once the square root falls below 1e-10, which bounds the
entropy correction rather than letting it blow up at the last grid point before
`T_c`.

### The paired potential

The pairing term is that of Alford, Braby, Paris and Reddy, written per flavour
so that the three potentials need not be equal:

    P_CFL = sum_q P_q(mu_q,T,m_q,a)
            + (Delta(T)^2/(pi^2 (hc)^3)) sum_q mu_q^2
            - B/(hc)^3                                              (Pcfl)

When the three potentials coincide at `mubar` the pairing term is
`3 mubar^2 Delta^2/(pi^2 (hc)^3)`, which is the `3 Delta^2 mu^2/pi^2` of that
reference and of `eos/abpr`. The `-3 m_s^2 mu^2/(4 pi^2)` term those references
carry alongside it is **not** added here: it is the leading expansion of the
massive strange Fermi gas, which (Pcfl) already contains exactly through
`P_s(mu_s, T, m_s, alpha)`, and adding both would count it twice.

Everything else follows from (Pcfl) as derivatives, which is what makes the
paired sector thermodynamically consistent:

    n_q = dP_CFL/dmu_q = n_q(mu_q,T,m_q,a)
                         + 2 mu_q Delta(T)^2 / (pi^2 (hc)^3)         (ncfl)
    s   = dP_CFL/dT    = sum_q s_q(mu_q,T,m_q,a)
                         + (2 Delta(T)/(pi^2 (hc)^3)) (dDelta/dT)
                           sum_q mu_q^2                              (scfl)
    f   = -P_CFL + sum_q mu_q n_q,     eps = f + T s                 (ecfl)

The energy density is *defined* by (ecfl) rather than summed from the flavours,
so (euler) holds in the paired phase by construction; it is checked all the
same, and holds to 2e-16 relative. The entropy correction (scfl) is negative
wherever the gap is falling (`dDelta/dT < 0`): pairing removes states from the
Fermi surface and the condensate carries less entropy than the gas it replaces.

The gluon term is not part of (Pcfl). In the CFL phase the gluons are all
massive through the Meissner effect and their thermal population is suppressed;
the sector remains available as a flag at the solver level and is added to the
totals there, as it is in the unpaired phase, but it is not inside the phase's
own potential.


## Conserved charges

With `S = +1` per `s` quark and `C` the electric charge of the
strongly-interacting matter alone, the quark quantum numbers `(B, C, S)` are
`(1/3, +2/3, 0)` for `u`, `(1/3, -1/3, 0)` for `d` and `(1/3, -1/3, +1)` for
`s`, so

    n_B = (n_u + n_d + n_s)/3
    n_C = (2 n_u - n_d - n_s)/3
    n_S = n_s                                                     (charges)

and the potentials map both ways,

    mu_B = mu_u + 2 mu_d
    mu_C = mu_u - mu_d
    mu_S = mu_s - mu_d                                              (basis)

inverted by

    mu_u = mu_B/3 + 2 mu_C/3
    mu_d = mu_B/3 - mu_C/3
    mu_s = mu_B/3 - mu_C/3 + mu_S

These maps are not written in this subpackage: they are `eos.general.basis`,
shared with every model, and a test asserts bit-for-bit agreement between the
two. The fractions are `Y_C = n_C/n_B` and `Y_S = n_S/n_B`, always per baryon
and always excluding the leptons; each flavour also reports `Y_q = n_q/n_B`, so
that `Y_u + Y_d + Y_s = 3` identically.

The sign of `mu_C` is fixed by `mu_C = mu_u - mu_d`, which is `mu_p - mu_n` in
the hadronic sector, so beta equilibrium reads `mu_C + mu_e = 0` here exactly
as it does there.


## Leptons, photons and the totals

The quantities above are those of the strongly-interacting sector. What a
solved point reports adds, from `eos.general.thermodynamics_leptons`, whichever
of these the mode and the flags call for:

- **electrons** (with positrons) at `mu_e`, a Fermi gas of mass 0.511 MeV and
  degeneracy 2, present wherever the mode has a lepton condition — the two
  Fermi integrals above at `g = 2`, `m = m_e`;
- **electron neutrinos** (with antineutrinos) at `mu_nue`, massless and of
  degeneracy 1, present only in the trapped mode;
- **photons**, `P_gamma = eps_gamma/3 = pi^2 T^4/(45 (hc)^3)`,
  `s_gamma = 4 eps_gamma/(3T)`;
- **thermal neutrinos**: the flavours *not* tracked in the composition, carried
  as `mu = 0` gases. Three flavours where the electron neutrino is
  free-streaming (`mu_nue = 0`), two where it is trapped and therefore already
  counted at its own potential.

None of these carries `B`, `C` or `S` in the sense of (charges) — `n_C` is
non-leptonic by definition — so they enter `P`, `eps` and `s` and the
neutrality condition, and nothing else. The totals are

    P   = P_matter   + P_e   + P_nue   + P_gamma   + P_nu_th
    eps = eps_matter + eps_e + eps_nue + eps_gamma + eps_nu_th
    s   = s_matter   + s_e   + s_nue   + s_gamma   + s_nu_th

with `P_matter` from (P) or (Pcfl), and `f = eps - T s`.


## Equilibrium modes and their closures

A mode fixes which quantities are imposed and which are unknown. In the
unpaired phase the unknowns are chemical potentials only — the potential is
explicit in `mu`, so unlike a mean-field model there is no field equation to
carry along and no density in the unknown vector. The rows below are given in
the order `solver.py` assembles them into the residual vector.

**`beta_eq_neutrinoless`** — conditions `(n_B, T)`; unknowns
`x = [mu_u, mu_d, mu_s, mu_e]`, four rows:

    r_1 = (n_u + n_d + n_s)/3 - n_B
    r_2 = n_C - n_e(mu_e, T)
    r_3 = mu_d - mu_u - mu_e            (= -mu_C - mu_e)
    r_4 = mu_s - mu_d                   (= mu_S)                 (res_beta)

Row `r_1` is baryon number, `r_2` total electric neutrality, `r_3` beta
equilibrium `d <-> u + e- + nubar_e` with free-streaming neutrinos, and `r_4`
strangeness equilibrium `s <-> d`, which is the statement `mu_S = 0`.

**`beta_eq_neutrino_trapped`** — conditions `(n_B, Y_Le, T)`; unknowns
`x = [mu_u, mu_d, mu_s, mu_e, mu_nue]`, five rows: `r_1`, `r_2` and `r_4`
unchanged, with

    r_3 = mu_d - mu_u - mu_e + mu_nue   (= -mu_C - mu_e + mu_nue)
    r_5 = (n_e + n_nue)/n_B - Y_Le

The neutrino gas carries its antiparticles, so `n_nue` is the net density and
`Y_Le` is the conserved electron-family number per baryon. The muon family is
not tracked, so `Y_Lmu` raises.

**`fixed_YC`** — conditions `(n_B, Y_C, T)`; unknowns `x = [mu_u, mu_d, mu_s]`,
three rows: `r_1` and `r_4` unchanged, with the charge fraction imposed instead
of neutrality,

    r_2' = n_C/n_B_calc - Y_C,     n_B_calc = (n_u + n_d + n_s)/3

With `leptons=False` the result is electrically charged quark matter, which is
what a mixed-phase construction needs per pure phase before global neutrality
is imposed. With `leptons=True` a neutralizing electron gas is added *after*
the solve, by inverting `n_e(mu_e, T) = n_C` for `mu_e` — a one-dimensional
inversion, not a further row, because the strongly-interacting sector of this
model does not respond to `mu_e` at fixed `Y_C`. The two therefore share the
same quark state and differ only in the lepton contribution to `P`, `eps` and
`s`.

**`fixed_YC_YS`** — conditions `(n_B, Y_C, Y_S, T)`; unknowns
`x = [mu_u, mu_d, mu_s]`, three rows: `r_1` and `r_2'` unchanged, with
strangeness equilibrium replaced by the imposed strangeness fraction,

    r_3' = n_s/n_B_calc - Y_S

This is the mode that separates a bag model from a nucleonic one: it is
meaningful here, where the `s` quark is a degree of freedom, and it raises in
`eos/zl`, which has none. Leptons are handled as in `fixed_YC`.

**The CFL phase** is a phase selector, not one of the four modes. It is closed
by flavour locking rather than by an equilibrium condition: the condensate
pairs the three flavours in equal numbers, so

    r_q = n_q_CFL(mu_q, T, m_q, alpha, Delta0) - n_B = 0,  q = u, d, s
                                                                 (res_cfl)

with `n_q_CFL` from (ncfl), is three rows for the three unknowns
`x = [mu_u, mu_d, mu_s]` at given `(n_B, T, Delta0)`. Two consequences are
worth stating because they are easy to expect wrongly:

- `n_C = (2 n_u - n_d - n_s)/3 = 0` identically, so the phase is *electrically
  neutral by construction* with no electrons at all, and `Y_C` is returned as
  zero to round-off (-1.7e-14 at `n_B = 0.8 fm^-3`) rather than solved for.
  This is the physical content of CFL neutrality.
- The phase is *not* in strangeness equilibrium. Equal densities at unequal
  masses require unequal potentials — at `n_B = 0.8 fm^-3`, `T = 0`,
  `Delta0 = 100` MeV the solve gives `mu_u = mu_d = 402.168` MeV and
  `mu_s = 434.121` MeV — so `mu_S = mu_s - mu_d != 0`, and
  `eps/n_B = mu_B + mu_S` at `P = 0` rather than `mu_B`. Reading `mu_B` as the
  energy per baryon of the paired phase is the standard way to misread it.

Wherever a temperature is accepted, entropy per baryon may be given in its
place, through an outer one-dimensional solve for `T` at fixed `n_B` and
fractions.


## What a solved point returns

Every mode returns the same record, so a caller reads one shape whichever mode
it asked for:

| field | symbol | note |
|---|---|---|
| `converged`, `error` | — | the status and the largest scaled residual |
| `n_B`, `T` | `n_B`, `T` | the conditions the point was solved at |
| `Y_C`, `Y_S`, `Y_L` | `Y_C`, `Y_S`, `Y_Le` | as realised, (charges) |
| `mu_u`, `mu_d`, `mu_s` | `mu_u`, `mu_d`, `mu_s` | the unknowns |
| `mu_e`, `mu_nu` | `mu_e`, `mu_nue` | zero where the mode has no lepton condition |
| `mu_B`, `mu_C`, `mu_S` | `mu_B`, `mu_C`, `mu_S` | derived, (basis) |
| `n_u`, `n_d`, `n_s` | `n_u`, `n_d`, `n_s` | net densities, antiquarks subtracted |
| `n_e`, `n_nu` | `n_e`, `n_nue` | net, antiparticles subtracted |
| `P_total`, `e_total` | `P`, `eps` | MeV/fm^3 |
| `s_total`, `f_total` | `s`, `f = eps - T s` | fm^-3, MeV/fm^3 |
| `Y_u`, `Y_d`, `Y_s`, `Y_e`, `Y_nu` | `n_i/n_B` | per baryon |

The CFL record carries the same fields and adds `Delta0` and `Delta` — the
parameter and its value at `T`, (gap). It **drops** `n_e` and `n_nu`, since the
phase has no leptons; `mu_e`, `mu_nu`, `Y_e` and `Y_nu` are present and stay at
zero. (`CFLPoint`'s own docstring says `Y_e` is absent too; it is not.)

There is no `n_s` in the sense of a scalar density in this model, and no
identity `n_s = (eps - 3P)/m*` to state: the masses are parameters and there is
no gap equation, so nothing here plays the part `m*` plays in a mean-field
model. The `n_s` this document lists is the strange-quark number density.

The entropy density is worth a word because nothing in the residual derives
from it and it is therefore easy to leave unstated. It is not integrated
separately: for a massless flavour it is the closed form (s0), for a massive
one it is (sF) — the Euler relation of the free gas, which is exact — plus the
correction of (massive), and in the paired phase it carries the further term of
(scfl). The gluon, photon and lepton entropies are added on top.


## The API surface

Three entry points, with the signatures every model in the repository carries:

    eos_point(par, mode, species, n_B=, T=|SnB=, leptons=, x0=, **conditions)
    eos_table(par, mode, species, axes=, fixed=, leptons=, skip_errors=,
              rows=, progress=, verbose=)
    eos_response(par, mode, species, frozen="equilibrium", n_B=, T=,
                 leptons=, rel_step=1e-3, **conditions)

`par` comes first and is never optional. `eos_point` returns a `PointResult`
whose `.ok` must be tested before `.point` is read; non-convergence is a return
value, not an exception. `eos_table` takes `axes={'nB': grid, 'T': grid}` plus
the mode's own fraction axis (`Y_C`, `Y_S`, `Y_Le`, or `Delta0` for the paired
phase) and returns arrays consumable by `eos.astro.tov` with no adapter; its
`progress` callback is invoked once per completed line with the repository's
standard dictionary

    {mode, line, n_lines, temp_key, temp, fracs, n_solved, n_requested,
     elapsed_s}

and `verbose=True` installs the built-in printer. Deep solver code never
prints.

`eos_response` implements one freeze, `frozen="equilibrium"`: everything
re-equilibrates under the perturbation, so the derivatives are taken along the
mode's own sequence,

    cs2_isothermal = dP/deps          at fixed T, along the sequence
    C_V            = (T/n_B) ds/dT    at fixed n_B

both by central differences over a relative step `rel_step` in the variable
differentiated, since alphaBag's residual has no analytic Jacobian in this
repository. `C_V` is returned only at `T > 0`, where it is defined. The
remaining freezes of CLAUDE.md §5 — frozen per-species composition, frozen
conserved fractions, the leptonic re-neutralization variants — raise
`NotImplementedError` naming the gap, and are recorded in `docs/DEFERRED.md`.
The derivative is taken at fixed `T`, and the key says so. The adiabatic
speed, larger by `C_P/C_V` at `T > 0`, is not computed by this model: `C_P` is
not among the returned quantities, so there is no factor to form it with.


## Numerics

Each mode is a system of three to five equations, solved with Powell's hybrid
method and, if that does not reach the gate, with Levenberg-Marquardt; when a
warm start was supplied and both fail, one further hybrid attempt is made from
the mode's own cold guess, since a warm start carried across the strange-quark
onset can land outside the basin. Every attempt is bounded internally and at
most three are made: a parameter scan must always get an answer back, and
non-convergence is a value carried on the record, never an exception and never
an unbounded loop.

Convergence is judged on a dimensionless residual. The rows carry mixed units —
densities of order 1e-1 fm^-3, fractions of order unity, equalities between
chemical potentials of order 1e3 MeV — so each row is divided by the scale of
the quantity it balances: `n_B` for a density row (`r_1`, `r_2`, and the three
rows of the CFL system), `|mu_B|` for a potential equality (`r_3`, `r_4`), and
unity for a row that is already dimensionless (`r_2'`, `r_3'`, `r_5`). The
state is accepted when the largest scaled component is below 1e-10. A gate on
the raw residual vector is dominated by whichever row happens to be largest and
accepts states satisfying the others only loosely; a gate on the solver's own
success flag reports whether the iteration terminated, which is a different
question again.

The cold guess estimates `mu ~ (pi^2 n_B)^(1/3) hbar c` from the massless
relation `n ~ mu^3/(pi^2 (hc)^3)` at one flavour per baryon, and adjusts it per
mode. Tables are built line by line — one line per temperature and per
combination of the fractions the mode fixes — and swept along the baryon
density with a warm start, each solved point seeding the next, with the step
bisected where a solve misses so that the sweep can walk through the
strange-quark onset rather than stopping at it. That loop is
`eos.general.tabulate`, shared with the other models; what this subpackage
supplies is which solver a mode name means and what part of a solved point
becomes the next guess.


## Not implemented

Raised, never silently ignored, and recorded in `docs/DEFERRED.md`: muons,
hyperons, deltas and thermal mesons (the last three are hadronic sectors with
no meaning in a deconfined phase, and strangeness enters through the `s` quark
instead), and every `eos_response` freeze beyond `equilibrium`.


## References

- A. Chodos et al., Phys. Rev. D 9, 3471 (1974) — the bag.
- B. A. Freedman and L. D. McLerran, Phys. Rev. D 16, 1169 (1977) — the
  O(alpha_s) correction.
- T. Fischer et al., Astrophys. J. Suppl. Ser. 194, 39 (2011) — the arrangement
  of the correction used here.
- M. Alford and S. Reddy, Phys. Rev. D 67, 074024 (2003) — parameter ranges.
- M. Alford, K. Rajagopal and F. Wilczek, Nucl. Phys. B 537, 443 (1999) — CFL.
- M. Alford, M. Braby, M. Paris and S. Reddy, Astrophys. J. 629, 969 (2005) —
  the pairing term.
- J. M. Lattimer et al. (Johns, Ellis, Lattimer), Astrophys. J. 473, 1020
  (1996) — the analytic Fermi integrals.
