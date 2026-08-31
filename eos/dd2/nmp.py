"""The map between DD2's couplings and the nuclear-matter parameters they
produce, in both directions.

`compute_nmp` extracts {n_sat, E_sat, m*/m, K_sat, Q_sat, K_sym, E_sym, L_sym}
from a `Parameters`; `invert_nmp` and `from_nmp` recover the couplings from a
subset of them, and `build_parametrization` composes that inverse with the
hyperon and Delta sector constructors, so one sample dict of nuclear-matter
parameters and single-particle potentials becomes one `Parameters`. The two
share this module because they share the derivatives: both take K_sat, Q_sat,
L_sym and K_sym from `snm_derivatives` below, so what the closure imposes is
exactly what the forward map reports and a round trip returns its own inputs.

Both directions sit ABOVE `solver.py` in the layer order, because every
quantity here is a property of solved symmetric nuclear matter at saturation.
That is why `from_nmp` lives here as a free function rather than as a
classmethod on `Parameters`, which is the bottom layer.

THE INVERSE MAP
---------------

The forward map (nmp.compute_nmp) extracts {n_sat, E_sat, m*/m, K_sat, Q_sat,
K_sym, E_sym, L_sym} from a Parameters. This inverts it.

The imposed set is {n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}:

  1. Isoscalar (4x4 root at FIXED n_sat, so no P=0 bracket search in the
     loop): free {Gamma_sigma, c_sigma, Gamma_omega, b_omega} matched to
     {P(n_sat)=0, E_sat, m*/m, K_sat}. The other two shape coefficients,
     b_sigma and c_omega, are PINNED at their published values -- see
     "Why two coefficients are pinned" below. m_sigma is fixed; a_i, d_i are
     derived internally (from_microscopic).
  2. Isovector (near-analytic): Gamma_rho(n_sat) from E_sym in closed form,
     then a_rho from L_sym by a 1-D root.
  3. The higher derivatives NOT imposed — Q_sat and K_sym — are computed
     forward from the recovered couplings and reported in
     InversionStatus.predictions. They are predictions of the closure, not
     inputs.

Imposing Q_sat instead of one pin is available (impose_Q_sat=True): the
isoscalar system is then the 5x5 {P, E_sat, m*/m, K_sat, Q_sat} over
{Gamma_sigma, b_sigma, c_sigma, Gamma_omega, b_omega}, with c_omega alone
pinned. It is not the default, but it IS usable now that Q_sat is analytic --
see "Q_sat is imposable now that it is analytic" below. The selector is the
argument only: the presence of "Q_sat" in the target dict decides nothing,
because a whole compute_nmp() dict carries Q_sat and would otherwise route the
natural round trip into a closure the caller never asked for.

There is no cross-constraint. It belongs to DD, not to DD2
----------------------------------------------------------
Earlier versions of this module closed the isoscalar sector with
f_sigma''(1) = f_omega''(1). That condition is real, but it is the DD
parametrization's, not DD2's. Typel, Phys. Rev. C 71, 064301 (2005), Sec. IV
imposes "f_sigma(1) = f_omega(1) = 1, f_sigma''(0) = f_omega''(0) = 0, and
f_sigma''(1) = f_omega''(1)" on the rational functions, for the stated reason
of reducing the number of free parameters, and counts EIGHT independent
parameters for DD. Typel et al., Phys. Rev. C 81, 015803 (2010) -- the DD2
paper -- states only the first two conditions and counts TEN. The difference
of one is exactly this constraint, and the published tables say the same
thing: f''_sigma(1) - f''_omega(1) is -6.0e-08 for DD and 2.200718e-03 for
DD2. DD2's fit never imposed it.

Imposing it here therefore closed DD2 with a condition its own fit had
dropped, which is why the published couplings were not a root of the closure
and why no seed recovered them. With the row gone they ARE a root: all four
default rows vanish at the published table, so a round trip through
compute_nmp recovers the published couplings rather than a set 3.9% away.

Why two coefficients are pinned, and why these two
--------------------------------------------------
E_sat and m*/m at fixed n_sat are blind to the shape coefficients -- they need
only f_i(1) = 1 -- so of the four default rows only P and K_sat carry any
shape information, and the four shape coefficients answer to two rows. Two
must be held.

Which two is decided in two steps, a local statistic and a global scan.
Build the isoscalar Jacobian d(NMP)/d(ln coupling) at the published DD2 point
-- rows divided by each parameter's own published magnitude (P by
1 MeV/fm^3), columns by the coupling -- and pin whichever subset leaves the
largest SMALLEST SINGULAR VALUE. Then confirm with a basin scan over a grid
of targets, which may VETO a locally-best choice: sigma_min is a statement
about one point, basin coverage is the statement about the space a sampler
actually walks.

The statistic is sigma_min and not `cond`, because cond = sigma_max/sigma_min
divides out the absolute strength of the weakest knob -- which is exactly the
number that decides whether that knob can reach an inference prior at all.
cond is quoted below beside sigma_min, never instead of it.

DEFAULT closure, rows {P, E_sat, m*/m, K_sat}, two of the four shape
coefficients pinned:

    b_sigma + c_omega    sigma_min 0.4657    cond 135    <- pinned here
    c_sigma + c_omega              0.3573         176
    b_sigma + b_omega              0.3230         195
    b_sigma + c_sigma              0.1919         326
    c_sigma + b_omega              0.1819         346
    b_omega + c_omega              0.1730         366

Both statistics agree, and the shipped pair is first under both. One
coefficient from each meson beats holding either shape whole, because what is
left free should be the least collinear surviving pair, and c_sigma against
b_omega at |cos| = 0.974 is the least collinear pair in the matrix.

Q_SAT closure, five rows, ONE pinned -- and here the local statistic and the
scan disagree, which is the case the second step exists for:

    c_sigma              sigma_min 2.9692e-01    cond 1236
    b_omega                        2.7379e-01         1378
    c_omega                        2.3706e-01         1590    <- pinned here
    b_sigma                        1.1771e-01         3092
    Gamma_sigma, Gamma_omega       ~1e-10 (numerically zero)

Neither vertex coupling can be pinned: holding one leaves the matrix rank
deficient, so only the four shape coefficients are candidates. Among those
sigma_min prefers c_sigma over the shipped c_omega by 25%, and cond ranks
them in the identical order -- so this is not the two statistics disagreeing
with each other. It is that c_omega was chosen on a Jacobian whose Q_sat row
still carried a third-difference stencil. That measurement ranked c_omega
259, b_omega 354, c_sigma 703, b_sigma 4191; those numbers are the stencil's
rather than the map's and are superseded by the table above.

THE BASIN SCAN VETOES c_sigma. Same targets, same seeds, both pins, counting
the targets each REACHES (status.ok):

                                             0 restarts   32 restarts
    72-cell grid, K_sat x Q_sat x m*/m x n_sat
      pin c_omega                              59/72         64/72
      pin c_sigma                              42/72         59/72
    200 random targets, the same four axes plus E_sat
      pin c_omega                             156/200       172/200
      pin c_sigma                             102/200       134/200

c_omega reaches more targets on both grids at both restart counts, and gets
there in about half the wall clock, because a target reached on the first
solve never pays for restarts. Among the targets BOTH reach the two are
indistinguishable -- worst relative error over the five imposed rows 2.6e-11
against 2.7e-11, medians ~1e-12 either way -- so what separates them is the
SIZE of the basin, not the accuracy inside it. c_omega stands: a 25% local
margin at one point does not survive the question of which targets are
findable from the published seed.

All eight counts above are the SAME on python.org 3.14.2 / numpy 2.3.5 /
scipy 1.17.0 and on anaconda 3.9.7 / numpy 1.26.4 / scipy 1.13.1, so the veto
is a property of the residual surface rather than of a solver version. The
sigma_min and cond tables agree to five digits across the two as well.

Q_sat is imposable now that it is analytic
------------------------------------------
It was not, while it was a third finite difference. The five-row closure
conditions at 259 and the stencil carried a relative floor near 1.5e-3, so a
solve inherited 259 x 1.5e-3 = 0.39 of relative coupling error; no choice of
pin rescued it (259 is the best of the four) because the collinearity behind
the 259 is a rank statement, not a coordinate one. At DD2's own point that
closure reached max|residual| = 1.4e-2 and imposed Q_sat only to 1.6 MeV,
saturating -- 64 and 128 restarts found nothing better.

With the derivatives taken by hand (`snm_derivatives`) the floor is gone and
the amplification has nothing left to amplify. The same closure at the same
point now reaches 1.5e-12 and imposes Q_sat to 1e-10 MeV, and perturbed
targets over dK_sat in [-20, +10] and dQ_sat in [-30, +100] MeV come back at
the same order. The default closure remains the default -- it imposes the
four nuclear-matter parameters anyone quotes and predicts the rest -- but
`impose_Q_sat=True` is now a usable branch rather than a documented trap.

What "converged" means here, and why the residual alone cannot say. A Powell
hybrid can give up on its first step and return its starting point bit for
bit, reporting the seed's own residual as though it were an answer, and
whether it does so at any given target is decided in that target's last bits
rather than by the SciPy version. That is a property of the solver, not of
the closure, and it survived the closure change: on a 105-cell
(K_sat 180-320) x (m*/m 0.45-0.75) grid at zero restarts, 18 targets missed
and 12 of those misses were stalls.

ANALYTIC DERIVATIVES APPEAR TO HAVE ENDED IT, and the mechanism is plain: the
residual hybr differences its own Jacobian from used to carry the stencil's
noise, so "not making good progress" was often a true report about the
surface rather than about the target. Re-measured after the change, neither
scan produces one -- 0 stalls in 7 misses over the 240-cell four-axis grid
quoted at ISO_GATE below, and 0 in 8 over the 30-cell (K_sat, m*/m) grid the
verify suite uses, at 0 and at 32 restarts alike. Two scans are not a proof of
absence and the guard is cheap, so `_stalled` and `STALL_RES` stay: what they
defend against is a solver handing back its input as an answer, which
section 6 forbids reporting silently whether or not it is currently
reachable.

What separates a stall from an answer is that the stall has not moved.
`InversionStatus` carries `coupling_shift`, the max relative distance from
the seed, and a solve that returns the seed unmoved on a residual above
STALL_RES is reported as ok=False rather than certified. The same condition
drives the restart loop, which is the substantive half: a stall whose
residual sits under the gate would otherwise keep the restarts from ever
running.

`coupling_shift` also answers a second question the residual never could:
"converged" and "recovered the published couplings" are different statements.
They now coincide at DD2's own point -- with the cross row gone the published
table IS a root of the default closure, and a round trip through compute_nmp
returns it to 1.1e-05 at a coupling_shift of the same order -- but they do
not coincide in general, and a caller inverting a moved target still needs to
be told how far the answer sits from where it started.

What limits which NMPs invert: the seed, not the physics
--------------------------------------------------------
A single solve from the published DD2 couplings converges only for targets
near DD2's own values, and the set it reaches traces a band through that seed
point. That band is a picture of one basin of attraction, NOT of the feasible
set, and reading it as physics is the mistake this module exists to prevent.
Restarts are what separate "these NMPs have no DD-RMF realisation" from "this
seed could not find it"; on the 30-cell (K_sat 160-320) x (m*/m 0.40-0.90)
grid the verify suite uses they take 22/30 to 27/30. Do NOT infer a
feasibility boundary from a scan run at low `n_restarts`.

Where they no longer buy anything is the Q_sat-imposing closure, which used
to be the harder surface of the two: it now reaches 30/30 at zero restarts
over K_sat 150-350 x Q_sat -400 to 800, three times the width of the grid
that gave 0/9 while the Q_sat row was a third difference.

(The 187-cell (K_sat, Q_sat) scan this section used to quote -- 7/187 at zero
restarts against 115/187 at sixty-four -- was measured with the retired
closure that imposed the cross-constraint and Q_sat together. Those numbers
do not carry over and are not restated here; the conclusion they supported
does, and is re-measured above.)

There is no stencil left
------------------------
K_sat, Q_sat, L_sym and K_sym were finite differences of quantities that are
themselves the output of a nonlinear solve, and every one of them carried
that floor: over h in [5e-5, 5e-4] Q_sat, a third difference, spanned 2.48
MeV and diverged outright by h = 1e-6, while K_sat spanned 5.2e-04 MeV. They
are differentiated by hand now -- see "THE DENSITY DERIVATIVES OF SATURATED
MATTER, IN CLOSED FORM" below -- and agree with the h-plateau of the stencil
they replaced rather than with any single h. Four published numbers moved by
that correction, all within their frozen tolerances:

    K_sat   242.724055 -> 242.724015      L_sym    55.033672 ->  55.033667
    Q_sat   168.713524 -> 168.786877      K_sym   -93.224031 -> -93.224009

n_sat, E_sat, m*/m and E_sym need no derivative and did not move at all.
"""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq, root

from eos.general.physics_constants import hc3
from eos.dd2.couplings import (
    SU6_HYPERON, DD2Y_HYPERON, MULTIPLET, vector_ratios, _POTENTIAL_KEY,
    scalar_ratio_from_potential, potential_from_scalar_ratio,
    rational_f, rational_df, rational_d2f, rational_d3f,
)
from eos.dd2.parameters import Parameters
from eos.dd2.thermodynamics import kF_from_n
from eos.dd2.solver import solve_snm, solve_snm_t0


# =============================================================================
# FORWARD:  couplings -> nuclear-matter parameters
# =============================================================================
def energy_per_baryon(par, n_B):
    """E/A [MeV] of symmetric nuclear matter at n_B [fm^-3]."""
    p = solve_snm_t0(par, n_B)
    return p.eps / n_B - par.m_nucleon


def _dirac_mass(point):
    """Nucleon Dirac mass m* [MeV] of a symmetric-matter point.

    The two nucleons share a kernel mass under the default
    `nucleon_mass_mode="average"`, so m*_n = m*_p and this is either of
    them; where the mode splits them, the isospin average is what the
    nuclear-matter parameters mean by m*.
    """
    m_eff = point.matter.m_eff_i
    return 0.5 * (m_eff["n"] + m_eff["p"])


def esym(par, n_B):
    """
    Symmetry energy E_sym(n_B) [MeV], mean-field closed form:
    kinetic/Dirac term + rho term in the tau_3 = ±1 convention.
    """
    p = solve_snm_t0(par, n_B)
    kF = kF_from_n(n_B * hc3, 4.0)
    EFs = np.sqrt(kF ** 2 + _dirac_mass(p) ** 2)
    _, _, Gr, _, _, _ = par.couplings_at(n_B)
    return kF ** 2 / (6.0 * EFs) + Gr ** 2 * (n_B * hc3) / (2.0 * par.m_rho ** 2)


# =============================================================================
# THE DENSITY DERIVATIVES OF SATURATED MATTER, IN CLOSED FORM
# =============================================================================
# K_sat, Q_sat, L_sym and K_sym used to be finite differences of quantities
# that are themselves the output of a nonlinear solve, which put a floor under
# each of them -- 1.5e-3 relative on Q_sat, a THIRD difference. They are
# written out here instead. Everything in this section is in natural units
# (n in MeV^3, kF and masses in MeV) and ' means d/dn.
#
# Symmetric matter at T = 0 carries one self-consistent field. Write
# S = Gamma_sigma(n) sigma, so that m* = m_N - S, and abbreviate
#
#     G(n) = Gamma_sigma(n)^2 / m_sigma^2,   W(n) = Gamma_omega(n)^2 / m_omega^2
#
# The sigma gap equation m_sigma^2 sigma = Gamma_sigma n_s is then
#
#     S = G(n) n_s(m_N - S, kF(n))                                       (gap)
#
# and, with omega_0 eliminated by its own field equation (m_omega^2 omega_0 =
# Gamma_omega n, so Gamma_omega omega_0 = W n),
#
#     eps = eps_kin(m*, kF) + S^2 / (2 G) + W n^2 / 2
#     mu  = E_F* + W n + W' n^2 / 2 - G' n_s^2 / 2
#
# the last two terms being Sigma^R = Gamma_omega' omega_0 n
# - Gamma_sigma' sigma n_s in these variables. Since P = mu n - eps at T = 0
# and E/A = eps/n - m_N,
#
#     (E/A)' = P / n^2,        P' = n mu'
#     K_sat  = 9 n^2 (E/A)''   = 9 n mu'                    } at P = 0,
#     Q_sat  = 27 n^3 (E/A)''' = 27 n (n mu'' - 3 mu')      } i.e. at n_sat
#
# so the third derivative of E/A costs only the SECOND derivative of mu. What
# that needs is S' and S'', from (gap) differentiated implicitly. Writing
# ns_m = dn_s/dm*, ns_k = dn_s/dkF and so on, and using dm*/dn = -S',
#
#     dn_s/dn = -ns_m S' + ns_k kF'                                      (dns)
#     S'  (1 + G ns_m) = G' n_s + G ns_k kF'
#     S'' (1 + G ns_m) = G'' n_s + 2 G' dn_s/dn
#                        + G (ns_mm S'^2 - 2 ns_mk S' kF'
#                             + ns_kk kF'^2 + ns_k kF'')
#
# The symmetry energy is already closed-form (`esym` above), so L_sym and
# K_sym follow from the same E_F* derivatives with no further machinery.
#
# Z_sat, the fourth derivative, is deliberately NOT reported. It would need a
# third derivative of the gap, and there is nothing to spend it on: no closure
# imposes Z_sat and nobody quotes it. The fourth finite difference it would
# replace spanned 4.8e+04 on a value of 4547 -- noise with a name.
#
# CONVENTION. These forms treat the two nucleons as one g = 4 gas at the
# average Dirac mass, which is `nucleon_mass_mode="average"` -- the convention
# the published DD2 nuclear-matter parameters are stated in, and the one
# `esym` above has always used. Under "physical" the kernel masses differ by
# 1.29 MeV and the derivatives are that convention's, not the parametrization's.

#: Nucleon degeneracy of symmetric matter treated as one gas: 2 spins x 2
#: isospins, at the common Dirac mass m* = m_N - Gamma_sigma sigma.
_G_SNM = 4.0


def _ns_partials(m, kF):
    """n_s [MeV^3] of the g = 4 nucleon gas and its partials in (m*, kF).

    With E_F = sqrt(kF^2 + m*^2) and L = asinh(kF/m*),

        n_s = (g / 4 pi^2) m* [kF E_F - m*^2 L]

    The kF partials are the integrand at the surface; the m* partials are the
    moments

        dn_s/dm*    =  (g / 2 pi^2) int_0^kF k^4 / E_k^3 dk
        d2n_s/dm*^2 = -(g / 2 pi^2) 3 m* int_0^kF k^4 / E_k^5 dk

    which k = m* sinh t turns into m*^2 int (cosh^2 t - 2 + sech^2 t) dt and
    int tanh^4 t dt, both elementary. Returns
    (n_s, ns_m, ns_k, ns_mm, ns_mk, ns_kk).
    """
    E = np.sqrt(kF ** 2 + m ** 2)
    L = np.arcsinh(kF / m)
    p = _G_SNM / (2.0 * np.pi ** 2)
    return (0.5 * p * m * (kF * E - m ** 2 * L),
            p * (0.5 * kF * E - 1.5 * m ** 2 * L + m ** 2 * kF / E),
            p * kF ** 2 * m / E,
            -3.0 * m * p * (L - kF / E - (kF / E) ** 3 / 3.0),
            p * kF ** 4 / E ** 3,
            p * m * kF * (2.0 * E ** 2 - kF ** 2) / E ** 3)


def _coupling_squares(par, n_nat):
    """(G, G', G'', G''') and (W, W', W'', W'''), d/dn in natural units.

    G = Gamma_sigma^2/m_sigma^2 and W = Gamma_omega^2/m_omega^2 are the
    combinations the closed forms above are written in; each is
    (Gamma_i(n_sat)/m_i)^2 f_i(x)^2 with x = n/n_sat, so the chain rule on
    f_i and its three x-derivatives is the whole content.
    """
    nsat_nat = par.n_sat * hc3
    x = n_nat / nsat_nat
    out = []
    for gamma, a, b, c, d, mass in (
            (par.gamma_sigma, par.a_sigma, par.b_sigma, par.c_sigma,
             par.d_sigma, par.m_sigma),
            (par.gamma_omega, par.a_omega, par.b_omega, par.c_omega,
             par.d_omega, par.m_omega)):
        f = rational_f(x, a, b, c, d)
        f1 = rational_df(x, a, b, c, d)
        f2 = rational_d2f(x, a, b, c, d)
        f3 = rational_d3f(x, a, b, c, d)
        K = (gamma / mass) ** 2
        out.append((K * f * f,
                    K * 2.0 * f * f1 / nsat_nat,
                    K * 2.0 * (f1 * f1 + f * f2) / nsat_nat ** 2,
                    K * 2.0 * (3.0 * f1 * f2 + f * f3) / nsat_nat ** 3))
    return out[0], out[1]


def snm_derivatives(par, n_B):
    """{K_sat, Q_sat, L_sym, K_sym} of symmetric matter at n_B [fm^-3].

    The nuclear-matter combinations 9 n^2 (E/A)'', 27 n^3 (E/A)''',
    3 n E_sym' and 9 n^2 E_sym'', analytically. K_sat and Q_sat are the
    saturation parameters only where P(n_B) = 0, which is where both callers
    evaluate them; the derivation is in the section header above.

    Solves symmetric matter ONCE, at n_B, and differentiates the closed forms
    around that solved point -- so it is also seven solves cheaper than the
    third-difference stencil it replaced.
    """
    point = solve_snm_t0(par, n_B)
    n = n_B * hc3
    m = _dirac_mass(point)                     # m* = m_N - Gamma_sigma sigma
    S = par.m_nucleon - m
    kF = kF_from_n(n, _G_SNM)
    kF1, kF2 = kF / (3.0 * n), -2.0 * kF / (9.0 * n ** 2)

    ns, ns_m, ns_k, ns_mm, ns_mk, ns_kk = _ns_partials(m, kF)
    (G, G1, G2, G3), (W, W1, W2, W3) = _coupling_squares(par, n)

    # --- the gap equation, differentiated implicitly ------------------------
    den = 1.0 + G * ns_m
    S1 = (G1 * ns + G * ns_k * kF1) / den
    dns = -ns_m * S1 + ns_k * kF1
    S2 = (G2 * ns + 2.0 * G1 * dns
          + G * (ns_mm * S1 ** 2 - 2.0 * ns_mk * S1 * kF1
                 + ns_kk * kF1 ** 2 + ns_k * kF2)) / den
    d2ns = (ns_mm * S1 ** 2 - 2.0 * ns_mk * S1 * kF1 + ns_kk * kF1 ** 2
            + ns_k * kF2 - ns_m * S2)

    # --- the Fermi energy at the moving mass and momentum -------------------
    E = np.sqrt(kF ** 2 + m ** 2)
    E1 = (kF * kF1 - m * S1) / E
    E2 = ((kF1 ** 2 + kF * kF2 + S1 ** 2 - m * S2) / E - E1 ** 2 / E)

    # --- isoscalar: mu, mu', mu'' -> K_sat, Q_sat ---------------------------
    mu1 = (E1 + W + 2.0 * W1 * n + 0.5 * W2 * n ** 2
           - 0.5 * G2 * ns ** 2 - G1 * ns * dns)
    mu2 = (E2 + 3.0 * W1 + 3.0 * W2 * n + 0.5 * W3 * n ** 2
           - 0.5 * G3 * ns ** 2 - 2.0 * G2 * ns * dns
           - G1 * (dns ** 2 + ns * d2ns))

    # --- isovector: E_sym = kF^2/(6 E_F*) + Gamma_rho^2 n / (2 m_rho^2) -----
    # R = Gamma_rho^2/m_rho^2 is a pure exponential, R' = -2 a_rho R / n_sat.
    k_rho = 2.0 * par.a_rho / (par.n_sat * hc3)
    R = (par.gamma_rho / par.m_rho) ** 2 * np.exp(-k_rho * (n - par.n_sat * hc3))
    R1, R2 = -k_rho * R, k_rho ** 2 * R
    u = kF ** 2
    u1, u2 = 2.0 * u / (3.0 * n), -2.0 * u / (9.0 * n ** 2)
    Es1 = u1 / (6.0 * E) - u * E1 / (6.0 * E ** 2)
    Es2 = (u2 / E - 2.0 * u1 * E1 / E ** 2 - u * E2 / E ** 2
           + 2.0 * u * E1 ** 2 / E ** 3) / 6.0

    return {
        "K_sat": 9.0 * n * mu1,
        "Q_sat": 27.0 * n * (n * mu2 - 3.0 * mu1),
        "L_sym": 3.0 * n * (Es1 + 0.5 * (R1 * n + R)),
        "K_sym": 9.0 * n ** 2 * (Es2 + 0.5 * (R2 * n + 2.0 * R1)),
    }


def compute_nmp(par, n_lo=0.12, n_hi=0.18):
    """
    Nuclear-matter parameters at saturation.

    Returns dict with n_sat [fm^-3], E_sat, K_sat, Q_sat, E_sym, L_sym,
    K_sym [MeV], m_eff_ratio, and P_sat [MeV/fm^3] (diagnostic, ~0 by
    construction). K_sym = 9 n^2 E_sym''(n) is reported because the NMP
    inversion treats it, like Q_sat, as a prediction of the closure rather
    than an input. Z_sat is not reported at all -- see the derivative section
    above for why.

    Every entry is exact: n_sat, E_sat, m*/m and E_sym need no derivative,
    and the four that do take theirs analytically (`snm_derivatives`) rather
    than by stencil.
    """
    n_sat = brentq(lambda n: solve_snm_t0(par, n).P, n_lo, n_hi, xtol=1e-12)
    at_sat = solve_snm_t0(par, n_sat)

    return {
        "n_sat": n_sat,
        "E_sat": energy_per_baryon(par, n_sat),
        "m_eff_ratio": _dirac_mass(at_sat) / par.m_nucleon,
        "E_sym": esym(par, n_sat),
        "P_sat": at_sat.P,
        **snm_derivatives(par, n_sat),
    }


# =============================================================================
# INVERSE:  nuclear-matter parameters -> couplings
# =============================================================================
#: Gate on the isoscalar residual.
#:
#: It was 2e-2, set wide to clear two scales that are both gone: the published
#: table's own 2.2e-3 violation of the cross-constraint, retired with that row,
#: and the third-difference noise behind Q_sat, retired with the stencil. What
#: is left is a residual whose rows are all exact, and the passing cells
#: separate by nine orders of magnitude. Measured over the four axes the
#: isoscalar residual actually has -- 240 random targets in
#: n_sat [0.140, 0.170] x E_sat [-17, -15] x m*/m [0.45, 0.75] x
#: K_sat [180, 320] -- 233 solves land at or below 4.6e-12, three sit in
#: [2.8e-3, 1.7e-2] and four are above 2e-2, with NOTHING in between. The old
#: gate certified those three without their being roots. 1e-8 sits three and a
#: half orders above the worst genuine root and five and a half below the
#: lowest non-root, so it is not a tuned number: anywhere in the gap does the
#: same job. The split is identical at 0 and 32 restarts, which says the three
#: are not seeds that could have been rescued.
ISO_GATE = 1e-8

#: Perturbed restarts attempted when the first isoscalar solve misses the
#: gate. They run ONLY on a miss, so an NMP set that inverts from the DD2 seed
#: costs exactly what it did before. What they buy is large and does not
#: saturate — see the module docstring — so this default is a compromise with
#: scan cost (a miss costs ~n_restarts x 40 ms), not a converged answer.
#: Raise it when mapping a boundary matters more than the wall clock.
N_RESTARTS = 32

#: Residual above which an UNMOVED seed is a stall rather than an answer.
#: `root(method="hybr")` can return its starting point bit for bit, reporting
#: the seed's own residual, and ISO_GATE is too coarse to notice a stall whose
#: residual happens to fall under it. What separates a stall from an answer is
#: that the stall has not moved at all. An unmoved seed is legitimate only
#: when the seed was already the root -- which is the case at DD2's own
#: nuclear-matter parameters, where the default closure returns 2.3e-08 -- so
#: this floor sits well above a genuine root and well below the misses.
#: Measured on
#: python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0.
STALL_RES = 1e-5


def _relative_shift(x, seed):
    """max |x_i - seed_i| / |seed_i| -- how far the solve left its seed."""
    x = np.asarray(x, dtype=float)
    seed = np.asarray(seed, dtype=float)
    return float(np.max(np.abs(x - seed) / np.abs(seed)))


def _stalled(x, seed, res):
    """The solve returned its seed unmoved while the residual is not zero.

    Bit-for-bit equality, not a tolerance: this is hybr giving up on its first
    step ("not making good progress", status 5), not a small final move.
    """
    return bool(np.array_equal(np.asarray(x, dtype=float),
                               np.asarray(seed, dtype=float))) and res > STALL_RES


#: The isoscalar shape coefficients held at their published DD2 values
#: (Typel et al. 2010), one tuple per closure. Two must be held when Q_sat is
#: predicted and one when it is imposed, because only P and K_sat carry shape
#: information among the default rows; the module docstring gives the measured
#: ranking behind each choice.
PINNED_DEFAULT = ("b_sigma", "c_omega")
PINNED_WITH_Q_SAT = ("c_omega",)


@dataclass
class InversionStatus:
    ok: bool
    message: str
    isoscalar_residual: float
    isovector_residual: float
    #: Higher derivatives the closure does not impose, computed forward from
    #: the recovered couplings with the same stencils as nmp.compute_nmp:
    #: {"Q_sat": MeV, "K_sym": MeV}. Empty only if the build itself failed.
    predictions: dict = field(default_factory=dict)
    #: How far the isoscalar solve left its seed, max relative over the free
    #: couplings. Exactly 0.0 means the solver never moved, which `ok` reads
    #: as a failure unless the seed was already the root (STALL_RES). Reported
    #: because "converged" and "recovered the published couplings" are
    #: different statements. They coincide at DD2's own NMPs, where the
    #: default closure returns the published table to 1.1e-05, but a moved
    #: target reaches a root some distance from the seed and the caller is
    #: told how far.
    coupling_shift: float = float("nan")


def _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma, Grho=3.0, a_rho=0.5):
    """Build a Parameters from free isoscalar params (a,d derived)."""
    return Parameters.from_microscopic(
        n_sat=n_sat, gamma_sigma=Gs, b_sigma=bS, c_sigma=cS,
        gamma_omega=Gw, b_omega=bW, c_omega=cW,
        gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)


def _isoscalar_quantities(par, n_sat):
    """{P, E/A, m*/m, K_sat, Q_sat} of SNM at n_sat (no P=0 search).

    The forward map's own quantities, so that the closure imposes exactly
    what `compute_nmp` reports. Q_sat costs nothing to carry now that it is
    analytic -- it used to be four extra solves, which is why the default
    closure once had to ask for it to be skipped.
    """
    at = solve_snm(par, n_sat)
    return dict(P=at.P, E_sat=at.eps / n_sat - par.m_nucleon,
                m_ratio=_dirac_mass(at) / par.m_nucleon,
                **snm_derivatives(par, n_sat))


def _restart_loop(iso_residual, seed, first, n_restarts, gate=ISO_GATE):
    """Keep the best of the first solve and up to n_restarts jittered ones.

    A STALL counts as a miss, exactly as an over-gate residual does. Without
    that, a hybr that gives up on its first step keeps a residual under the
    gate and the restarts never run -- which is how DD2's own nuclear-matter
    parameters used to come back as the published seed unmoved. They are not
    unreachable: the FIRST jittered restart drives that same system to 6.8e-08
    and recovers K_sat to 1e-4 MeV.

    Deterministic by construction: the same NMP must invert identically on
    every run and in every parallel worker, so the generator is seeded with a
    constant rather than left to entropy.
    """
    def missed(x, res):
        return res >= gate or _stalled(x, seed, res)

    best_x = first.x
    best_res = float(np.max(np.abs(iso_residual(best_x))))
    stalled = _stalled(best_x, seed, best_res)
    if missed(best_x, best_res) and n_restarts:
        rng = np.random.default_rng(0)
        base = np.asarray(seed, dtype=float)
        for _ in range(n_restarts):
            try:
                trial = root(iso_residual,
                             base * rng.uniform(0.75, 1.35, base.size),
                             method="hybr", tol=1e-12)
                res = float(np.max(np.abs(iso_residual(trial.x))))
            except Exception:      # a jittered seed that will not build a
                continue           # trial parametrization is not a finding
            # A jittered start is never the seed, so no trial is itself a
            # stall: the first one accepted always displaces one, even on a
            # worse residual. The stalled residual is the SEED's, not an
            # answer's, so keeping it would be keeping the wrong number.
            if stalled or res < best_res:
                best_x, best_res, stalled = trial.x, res, False
            if not missed(best_x, best_res):
                break
    return best_x, best_res


def invert_nmp(nmp, m_sigma=546.212459, seed=None, n_restarts=N_RESTARTS,
               impose_Q_sat=False):
    """Recover DD2 couplings from a target NMP dict.

    nmp needs {n_sat, E_sat, m_eff_ratio, K_sat, E_sym, L_sym}; "Q_sat" is
    consumed only when it is imposed. Returns (Parameters,
    InversionStatus). Raises ValueError only on a hard infeasibility — m*/m
    outside the physical window, or E_sym below the kinetic symmetry energy
    at a CONVERGED isoscalar solution. A soft failure (the isoscalar solve
    missing its gate) is reported via status.ok=False, and the returned
    parametrization is then None: there is no meaningful coupling set to
    hand back, and the isovector sector is never fitted on a garbage point.

    impose_Q_sat selects the isoscalar closure:
      False — the default, and the only one that ships as usable: Q_sat is a
              PREDICTION. 4x4 over {Gamma_sigma, c_sigma, Gamma_omega,
              b_omega} with b_sigma and c_omega pinned at their published
              values; conditions {P(n_sat)=0, E_sat, m*/m, K_sat}, all of
              which are h-exact.
      True  — Q_sat is imposed: 5x5 over {Gamma_sigma, b_sigma, c_sigma,
              Gamma_omega, b_omega} with c_omega alone pinned. The Q_sat row
              is a third finite difference and the closure amplifies its
              ~1.5e-3 relative floor by ~259, so a target that is not already
              near a known root inherits O(0.4) of relative coupling error.
              Available because the caller may want the branch; NOT a closure
              to trust until the derivative is analytic. See the module
              docstring.

    There is no cross-constraint row in either closure: f''_sigma(1) =
    f''_omega(1) is the DD parametrization's condition, not DD2's (module
    docstring, with the sources). Presence of "Q_sat" in the dict selects
    nothing — a whole compute_nmp() dict carries it, and routing the natural
    round trip into the noisier closure on that accident is what this
    argument's old None default did.

    Either way the recovered couplings' Q_sat and K_sym are computed forward
    (same stencils as nmp.compute_nmp) and reported in status.predictions.

    `n_restarts` perturbed seeds are tried when the first isoscalar solve
    misses ISO_GATE. This is not a refinement: it is what separates "these
    NMPs have no DD-RMF realisation" from "this seed could not find it" —
    see the module docstring. Set it to 0 for single-seed behaviour.
    """
    if impose_Q_sat and "Q_sat" not in nmp:
        raise ValueError("impose_Q_sat=True but the NMP dict carries no Q_sat")
    n_sat = nmp["n_sat"]
    # Feasibility: m*/m too small drives Gamma_sigma sigma -> m_N
    # (scalar collapse); outside a physical RMF window there is no DD2-form fit.
    if not (0.35 < nmp["m_eff_ratio"] < 0.95):
        raise ValueError(
            f"NMP inversion infeasible: m*/m = {nmp['m_eff_ratio']} outside the "
            f"physical (0.35, 0.95) window (scalar collapse / no DD2-form fit)")

    ref = Parameters.default()
    pinned = PINNED_WITH_Q_SAT if impose_Q_sat else PINNED_DEFAULT
    held = {name: getattr(ref, name) for name in pinned}

    if impose_Q_sat:
        if seed is None:
            # DD2-class NMPs sit near the published couplings, and the
            # residual surface has spurious basins a generic seed falls into.
            seed = [ref.gamma_sigma, ref.b_sigma, ref.c_sigma,
                    ref.gamma_omega, ref.b_omega]
        tgt = np.array([0.0, nmp["E_sat"], nmp["m_eff_ratio"],
                        nmp["K_sat"], nmp["Q_sat"]])

        def iso_residual(p):
            Gs, bS, cS, Gw, bW = p
            cW = held["c_omega"]
            if cS <= 0 or Gs <= 0 or Gw <= 0:
                return [1e3] * 5
            try:
                par = _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma)
                q = _isoscalar_quantities(par, n_sat)
            except (ValueError, RuntimeError):
                return [1e3] * 5
            return [q["P"] - tgt[0], q["E_sat"] - tgt[1],
                    q["m_ratio"] - tgt[2], (q["K_sat"] - tgt[3]) * 1e-2,
                    (q["Q_sat"] - tgt[4]) * 1e-2]

        def couplings_of(x):
            Gs, bS, cS, Gw, bW = x
            return Gs, bS, cS, Gw, bW, held["c_omega"]
    else:
        if seed is None:
            seed = [ref.gamma_sigma, ref.c_sigma, ref.gamma_omega, ref.b_omega]
        tgt = np.array([0.0, nmp["E_sat"], nmp["m_eff_ratio"], nmp["K_sat"]])

        def iso_residual(p):
            Gs, cS, Gw, bW = p
            bS, cW = held["b_sigma"], held["c_omega"]
            if cS <= 0 or Gs <= 0 or Gw <= 0:
                return [1e3] * 4
            try:
                par = _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma)
                q = _isoscalar_quantities(par, n_sat)
            except (ValueError, RuntimeError):
                return [1e3] * 4
            return [q["P"] - tgt[0], q["E_sat"] - tgt[1],
                    q["m_ratio"] - tgt[2], (q["K_sat"] - tgt[3]) * 1e-2]

        def couplings_of(x):
            Gs, cS, Gw, bW = x
            return (Gs, held["b_sigma"], cS, Gw, bW, held["c_omega"])

    first = root(iso_residual, seed, method="hybr", tol=1e-12)
    best_x, iso_res = _restart_loop(iso_residual, seed, first, n_restarts)
    Gs, bS, cS, Gw, bW, cW = couplings_of(best_x)
    shift = _relative_shift(best_x, seed)

    if _stalled(best_x, seed, iso_res):
        # The solver never left the seed, and the seed is not a root: this
        # is the seed couplings handed straight back with their own residual.
        # ISO_GATE is too coarse to catch every such case, so the verdict is
        # made here instead. Section 6: a non-convergence is a reported
        # return value, never a silent wrong answer.
        return None, InversionStatus(
            ok=False,
            message=f"the isoscalar solve returned its seed unmoved at "
                    f"residual {iso_res:.2e}; {n_restarts} restarts did not "
                    f"find a root (the seed is a stationary point of the "
                    f"residual norm, not a zero of it)",
            isoscalar_residual=iso_res, isovector_residual=float("nan"),
            coupling_shift=shift)

    if iso_res >= ISO_GATE:
        # The isoscalar sector did not converge. Fitting the isovector sector
        # on top would read the Dirac mass off a meaningless point, and the
        # "E_sym below the kinetic symmetry energy" hard-infeasibility test
        # would then fire or not fire depending on numerical garbage. A miss
        # here is a SOFT failure by contract — the caller scores it and moves
        # on — so report it and return no parametrization.
        return None, InversionStatus(
            ok=False,
            message=f"isoscalar residual {iso_res:.2e} above the "
                    f"{ISO_GATE:.0e} floor after {n_restarts} restarts (the "
                    f"targets are probably inconsistent with the closure at "
                    f"this K_sat)",
            isoscalar_residual=iso_res, isovector_residual=float("nan"),
            coupling_shift=shift)

    # --- isovector: Gamma_rho analytic, a_rho by 1-D root -------------------
    # Built from best_x — the restart winner — not the first solve: the
    # kinetic symmetry energy below reads m_eff off this parametrization, and
    # evaluating it on a rejected solution would fit Gamma_rho to the wrong
    # Dirac mass.
    par_iso = _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma)
    at = solve_snm(par_iso, n_sat)
    kF = kF_from_n(n_sat * hc3, 4.0)
    EFs = float(np.sqrt(kF ** 2 + _dirac_mass(at) ** 2))
    kin = kF ** 2 / (6.0 * EFs)
    n_nat = n_sat * hc3
    rho_term = nmp["E_sym"] - kin
    if rho_term <= 0:
        raise ValueError(
            f"NMP inversion infeasible: E_sym={nmp['E_sym']} below the "
            f"kinetic symmetry energy {kin:.2f} MeV (no real Gamma_rho)")
    # E_sym = kF^2/(6 EF*) + Gamma_rho^2 n/(2 m_rho^2)  ->  Gamma_rho analytic
    Grho = float(np.sqrt(rho_term * 2.0 * par_iso.m_rho ** 2 / n_nat))

    def Lsym_of_arho(a_rho):
        p = Parameters.from_microscopic(
            n_sat=n_sat, gamma_sigma=Gs, b_sigma=bS, c_sigma=cS,
            gamma_omega=Gw, b_omega=bW, c_omega=cW,
            gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)
        return snm_derivatives(p, n_sat)["L_sym"]

    a_rho = brentq(lambda a: Lsym_of_arho(a) - nmp["L_sym"], -2.0, 5.0,
                   xtol=1e-10)
    isov_res = abs(Lsym_of_arho(a_rho) - nmp["L_sym"])

    par = Parameters.from_microscopic(
        n_sat=n_sat, gamma_sigma=Gs, b_sigma=bS, c_sigma=cS,
        gamma_omega=Gw, b_omega=bW, c_omega=cW,
        gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)

    # --- report what the closure predicts, with the forward map's stencils --
    final = snm_derivatives(par, n_sat)
    predictions = {"Q_sat": final["Q_sat"], "K_sym": final["K_sym"]}

    status = InversionStatus(
        ok=(isov_res < 1e-3),                # isoscalar gate already passed
        message="converged" if isov_res < 1e-3 else
        f"isovector residual {isov_res:.2e} above 1e-3",
        isoscalar_residual=iso_res, isovector_residual=float(isov_res),
        predictions=predictions, coupling_shift=shift)
    return par, status


def from_nmp(nmp, m_sigma=546.212459, return_status=False):
    """Nuclear-matter parameters -> a `Parameters` carrying those couplings.

    `nmp` is a dict with {n_sat, E_sat, m_eff_ratio, K_sat, E_sym, L_sym}.
    The inversion always uses the default closure -- four rows over four
    couplings with b_sigma and c_omega pinned -- and reports Q_sat and K_sym
    as predictions in the status. A "Q_sat" key is ignored here: imposing it
    is `invert_nmp(..., impose_Q_sat=True)` and is not a closure to reach by
    accident. Returns the `Parameters`, or (Parameters, InversionStatus) when
    `return_status`.

    **Raises when the inversion did not converge**, since a caller asking only
    for parameters has nowhere to put a failure. `invert_nmp` is the CLAUDE.md
    section 6 boundary and returns `(Parameters, InversionStatus)`; this is the
    face for a caller that has declared it will not score failures, so use
    `invert_nmp` directly to score a target instead of raising on it. Returning
    `None` here was the third answer among the three models with an inversion,
    and the one that carried a failure two layers down: the None travelled
    until `solver.py` raised `'NoneType' object has no attribute
    'kernel_masses'`.

    The hyperon and Delta sectors attach on top of the result through
    `from_hyperon_potentials` / `from_delta_potential` below, once the
    nucleon sector is set; they are not folded in here.
    """
    par, status = invert_nmp(nmp, m_sigma=m_sigma)
    if not status.ok:
        raise RuntimeError(f"NMP inversion failed: {status.message}")
    return (par, status) if return_status else par


# ==========================================================================
# THE PUBLISHED NUCLEAR-MATTER PARAMETERS, TWICE
# ==========================================================================
# Two dicts, because they answer two different questions. A reader checking
# this model against the paper's table needs the digits the paper prints; a
# caller starting an inference "around" the published set needs the numbers
# the published COUPLINGS actually produce, which the printed digits are a
# rounding of. Neither is a gate.

#: The nuclear-matter parameters as PRINTED by Typel, Roepke, Klaehn,
#: Blaschke & Wolter, Phys. Rev. C 81, 015803 (2010) -- the DD2 paper -- for
#: the DD2 parametrization. Four to six significant figures.
PUBLISHED_NMP = {
    "n_sat": 0.149065, "E_sat": -16.02, "m_eff_ratio": 0.5625,
    "K_sat": 242.7, "E_sym": 31.67, "L_sym": 55.04,
}

#: The same six at full precision: `compute_nmp(Parameters.default())` on the
#: published couplings, frozen here so that reading them costs no saturation
#: solve. Regenerate with that call.
#:
#: What the paper's rounding costs, measured as the worst relative distance
#: between the couplings `invert_nmp` returns and the published ones, over
#: the eight free couplings (Gamma_sigma, b_sigma, c_sigma, Gamma_omega,
#: b_omega, c_omega, Gamma_rho, a_rho):
#:
#:     from PUBLISHED_NMP         8.5e-05
#:     from PUBLISHED_NMP_EXACT   7.6e-05
#:
#: i.e. nothing measurable. Both sit at the isoscalar solve's own convergence
#: floor, because DD2 prints enough digits that its rounding is below it. The
#: same measurement costs SFHo a factor of 25 (see `eos/sfho/nmp.py`), on one
#: two-digit entry, which is why the twin is shipped for every model rather
#: than for the one where it happened to matter.
PUBLISHED_NMP_EXACT = {
    "n_sat": 0.1490767283263872, "E_sat": -16.022620282213552,
    "m_eff_ratio": 0.5625212010574624, "K_sat": 242.72401473229172,
    "E_sym": 31.67006103137942, "L_sym": 55.033666576106114,
}


# ==========================================================================
# THE HYPERON AND DELTA SECTORS FROM THEIR SINGLE-PARTICLE POTENTIALS
# ==========================================================================
# Free functions rather than classmethods on `Parameters`, and here rather
# than in `parameters.py`, for the reason stated at the top of this module:
# both invert a potential by re-solving symmetric nuclear matter at
# saturation, so both sit ABOVE `solver.py` in the layer order, while
# `parameters.py` is its bottom (CLAUDE.md section 5).

def from_hyperon_potentials(U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0,
                            base=None):
    """
    Nucleon + hyperon octet whose scalar couplings are *inverted* from the
    hyperon potentials U_Y in SNM at saturation (report §2.4b), on top of the
    vector couplings `base` declares. This is the mechanism that regenerates
    the DD2Y R_sigma table (U_Xi = -18) and the route for non-DD2Y potentials.
    Hyperon masses default to the DD2Y (Marques) values.

    base: an existing Parameters to attach the hyperon sector to (e.g.
    an NMP-inverted nucleon par, so NMP + hyperons compose); defaults to
    nucleonic DD2. The scalar inversion re-solves SNM on ``base``, so it
    adapts to that par's nucleon couplings automatically.

    **The vector sector comes from `base`'s nine SU(6)-breaking factors, and
    the inversion runs AFTER them**, which is the whole reason this is one
    call rather than two. U_Y = -Gamma_sigmaY sigma + Gamma_omegaY omega0 +
    Sigma^R holds the scalar and vector couplings TOGETHER, so a rescaled
    x_omegaY changes the x_sigmaY that reproduces the same depth; inverting
    first and rescaling after would silently move U_Y. To break SU(6), set the
    factors on the base and let this function close the depths on them:

        base = replace(Parameters.default(), y_omega_Lambda=1.5,
                       y_phi_Lambda=1.5, ...)
        par  = from_hyperon_potentials(U_Xi=-14.0, base=base)

    `y_phi_Lambda = y_phi_Sigma = y_phi_Xi = 0.0` on the base is how a
    hyperonic set is built with no phi sector at all -- the coupling carries
    that statement, there is no flag for it.
    """
    base = replace(base if base is not None else Parameters.default(),
                   U_Lambda=U_Lambda, U_Sigma=U_Sigma, U_Xi=U_Xi)
    sat = solve_snm(base, base.n_sat)
    Gs_sat, Gw_sat, _, _, _, _ = base.couplings_at(base.n_sat)
    U_map = {"U_Lambda": U_Lambda, "U_Sigma": U_Sigma, "U_Xi": U_Xi}
    y = base.su6_breaking

    rows = []
    for name in SU6_HYPERON:
        x_omega, _, _ = vector_ratios(name, *y[MULTIPLET[name]])
        x_sigma = scalar_ratio_from_potential(
            U_map[_POTENTIAL_KEY[name]], x_omega, Gs_sat, Gw_sat,
            sat.matter.fields["sigma"], sat.matter.fields["omega0"],
            sat.matter.Sigma_R)
        rows.append((name, DD2Y_HYPERON[name]["mass"], x_sigma))
    return replace(base, hyperon_couplings=tuple(rows))


def from_delta_potential(U_Delta=-50.0, x_Delta_omega=1.0,
                         x_Delta_rho=1.0, base=None, x_Delta_sigma=None):
    """
    Δ-isobar couplings from the Δ single-particle potential in SNM at
    saturation (report v11 §2.4). There is no canonical DD2Δ coupling
    table, so the default is universal coupling (x_Δσ = x_Δω = x_Δρ = 1);
    this constructor instead fixes x_Δσ by inverting

        U_Δ = -x_Δσ Γ_σN σ̄ + x_Δω Γ_ωN ω0 + Σ^R      (all at n_sat)

    for a chosen Δ potential (literature U_Δ ∈ [-100, -50] MeV, default -50)
    and vector ratios x_Delta_omega, x_Delta_rho -- the free variables of
    this sector, carrying the same names as the `Parameters` fields they set.
    base: an existing Parameters to attach the Δ sector to (e.g. a DD2Y
    octet); defaults to nucleonic DD2.

    x_Delta_sigma may be given INSTEAD, in which case it is taken as it stands
    and `U_Delta` is ignored: the ratio is the parameter and needs no solve,
    the depth is the constrained quantity and does. Whichever is given, the
    other follows -- `delta_potential` reads back the depth of a par whose
    ratio was chosen, and it is the literature range above that says whether
    the choice landed anywhere physical.
    """
    base = base or Parameters.default()
    if x_Delta_sigma is None:
        if not (-100.0 <= U_Delta <= -50.0):
            raise ValueError(
                f"U_Delta = {U_Delta} MeV outside the literature range "
                f"[-100, -50]; pass an explicit value in range or widen it")
        sat = solve_snm(base, base.n_sat)
        Gs_sat, Gw_sat, _, _, _, _ = base.couplings_at(base.n_sat)
        x_Delta_sigma = scalar_ratio_from_potential(
            U_Delta, x_Delta_omega, Gs_sat, Gw_sat, sat.matter.fields["sigma"],
            sat.matter.fields["omega0"], sat.matter.Sigma_R)
    return replace(base, x_Delta_sigma=x_Delta_sigma,
                   x_Delta_omega=x_Delta_omega, x_Delta_rho=x_Delta_rho)


# --------------------------------------------------------------------------
# THE OTHER DIRECTION: a parametrization reports its own depths
# --------------------------------------------------------------------------
# The two constructors above impose a potential and solve for a coupling
# ratio. These read a finished `Parameters` and report the potentials its
# ratios amount to -- the forward half of the same one-line map, and what
# says whether a directly chosen ratio landed anywhere the literature knows.
# Both re-solve SNM at saturation on the par they are given, because that is
# where a single-particle potential is defined and an inverted or rescaled
# par does not saturate where nucleonic DD2 does.

def _saturation_terms(par):
    """(Gamma_sigmaN, Gamma_omegaN, sigma, omega0, Sigma^R) at n_sat, in SNM."""
    sat = solve_snm(par, par.n_sat)
    Gs_sat, Gw_sat, _, _, _, _ = par.couplings_at(par.n_sat)
    return (Gs_sat, Gw_sat, sat.matter.fields["sigma"],
            sat.matter.fields["omega0"], sat.matter.Sigma_R)


def delta_potential(par):
    """U_Delta (MeV) of `par`'s Delta sector: the inverse of
    `from_delta_potential`, and the way to read the depth of a par whose
    x_Delta_sigma was chosen rather than inverted."""
    Gs, Gw, sigma, omega0, SigmaR = _saturation_terms(par)
    return potential_from_scalar_ratio(par.x_Delta_sigma, par.x_Delta_omega,
                                       Gs, Gw, sigma, omega0, SigmaR)


def hyperon_potentials(par):
    """{U_Lambda, U_Sigma, U_Xi} (MeV) of `par`'s hyperon sector.

    The inverse of `from_hyperon_potentials`, and the check that a change to
    the nine SU(6)-breaking factors did what was asked: the factors move the
    VECTOR couplings, so at fixed x_sigma they move these depths, and after a
    re-inversion on the rescaled base they come back to what was imposed. One
    entry per multiplet, read off its first member -- the three charge states
    of a multiplet share a scalar coupling and hence a depth.
    """
    if not par.hyperon_couplings:
        return {}
    Gs, Gw, sigma, omega0, SigmaR = _saturation_terms(par)
    y = par.su6_breaking
    out = {}
    for name, _mass, x_sigma in par.hyperon_couplings:
        key = _POTENTIAL_KEY[name]
        if key in out:
            continue
        x_omega, _, _ = vector_ratios(name, *y[MULTIPLET[name]])
        out[key] = potential_from_scalar_ratio(x_sigma, x_omega, Gs, Gw,
                                               sigma, omega0, SigmaR)
    return out

# ==========================================================================
# NMPs + SECTOR POTENTIALS -> ONE PARAMETRIZATION
# ==========================================================================

#: The nine SU(6)-breaking factors, one per (vector meson, multiplet) pair.
#: They scale the hyperon VECTOR couplings (`couplings.vector_ratios`), so
#: they are applied to the inverted base BEFORE the hyperon depths are closed
#: on it -- at fixed depth they move x_sigma, at fixed x_sigma they move the
#: depth, and only the first order is a re-fit rather than a redefinition.
SU6_FACTOR_KEYS = ("y_omega_Lambda", "y_omega_Sigma", "y_omega_Xi",
                   "y_rho_Lambda", "y_rho_Sigma", "y_rho_Xi",
                   "y_phi_Lambda", "y_phi_Sigma", "y_phi_Xi")

#: Hadronic-sector coupling knobs that may be carried *inside* an NMP sample
#: dict, alongside the nuclear-matter parameters themselves, so that one
#: sample describes the whole hadronic parametrization. `x_Delta_omega` and
#: `x_Delta_rho` are the Delta vector coupling ratios; the scalar one may be
#: named either way round -- `U_Delta` inverts a depth into `x_Delta_sigma`,
#: `x_Delta_sigma` takes the ratio as it stands, and giving both is an error
#: rather than a precedence rule, because they are two names for one number.
SECTOR_KEYS = (("U_Lambda", "U_Sigma", "U_Xi",
                "U_Delta", "x_Delta_sigma", "x_Delta_omega", "x_Delta_rho")
               + SU6_FACTOR_KEYS)

#: The Delta depth `from_delta_potential` publishes, used when a sample names
#: neither side of the scalar pair.
DEFAULT_U_DELTA = -50.0


def _split_sample(sample, hyperon_potentials=None, U_Delta=None):
    """Separate a sample dict into (nmp, sector kwargs).

    A sample may carry any of the `SECTOR_KEYS` next to the nuclear-matter
    parameters; those override the corresponding keyword arguments. This is
    what lets one dict put L_sym, U_Xi, y_omega_Xi and U_Delta on axes
    together -- they are all "hadronic parameters" to the caller even though
    the inversion treats them in separate stages. Keys absent from both the
    sample and the keyword arguments are left out, so the sector constructors
    below apply their own published defaults.

    The Delta scalar sector is named ONCE: `U_Delta` (a depth, inverted) or
    `x_Delta_sigma` (a ratio, taken as it stands). Both together raise --
    there is no reading of two different values of one number that is not a
    caller's mistake.
    """
    nmp = {k: v for k, v in sample.items() if k not in SECTOR_KEYS}
    pots = dict(hyperon_potentials or {})
    pots.update({k: float(sample[k]) for k in ("U_Lambda", "U_Sigma", "U_Xi")
                 if k in sample})

    depth = sample.get("U_Delta", U_Delta)
    ratio = sample.get("x_Delta_sigma")
    if depth is not None and ratio is not None:
        raise ValueError(
            "the Delta scalar sector is given twice: U_Delta = "
            f"{depth} MeV and x_Delta_sigma = {ratio}. Name one -- the depth "
            "to invert it, the ratio to set it -- and read the other back "
            "with `delta_potential`.")
    if depth is None and ratio is None:
        depth = DEFAULT_U_DELTA

    sector = {"hyperon_potentials": pots,
              "su6": {k: float(sample[k]) for k in SU6_FACTOR_KEYS
                      if k in sample},
              "U_Delta": None if depth is None else float(depth),
              "x_Delta_sigma": None if ratio is None else float(ratio),
              "x_Delta_omega": float(sample.get("x_Delta_omega", 1.0)),
              "x_Delta_rho": float(sample.get("x_Delta_rho", 1.0))}
    return nmp, sector


def build_parametrization(nmp, flags, hyperon_potentials=None,
                          U_Delta=None):
    """Nuclear-matter parameters to a `Parameters` with the strange and
    resonant sectors attached, as `flags` requires.

    `invert_nmp` inverts the NUCLEON sector only -- it carries no hyperon
    couplings, so `SpeciesFlags(hyperons=True)` on its output would fail deep
    in a coupling lookup. The hyperon and Delta sectors are attached on top
    here, each by inverting its single-particle potential in symmetric matter
    at saturation *on the inverted base*, so they adapt to that base's nucleon
    couplings rather than assuming DD2's.

    `nmp` may also carry any of the `SECTOR_KEYS` -- the hyperon depths
    U_Lambda, U_Sigma, U_Xi; the Delta sector as either U_Delta or
    x_Delta_sigma, with x_Delta_omega and x_Delta_rho beside it; and the nine
    SU(6)-breaking factors `SU6_FACTOR_KEYS`. Those take precedence over the
    keyword arguments, so a single dict can put nuclear-matter parameters,
    sector potentials and coupling ratios on axes together.

    **The SU(6) factors are applied between the two stages**, on the inverted
    nucleon base and before the hyperon depths are closed on it. That order is
    the whole point of them being here rather than in a `replace` on the
    result: the factors scale the VECTOR couplings, so imposing them first
    re-inverts x_sigma and HOLDS the depths at what was asked, while applying
    them afterwards leaves x_sigma alone and moves the depths instead. Both
    are legitimate physics and only the first is a re-fit at fixed U_Y; a
    caller who wants the second does it on the returned par and reads the new
    depths back with `hyperon_potentials`.

    Returns `(par, stage, message)`. `stage` is 'ok', 'inversion_failed' when
    the NMPs have no DD-RMF realisation at all, or 'sectors_failed' when they
    do but the hyperon/Delta scalar inversion does not converge on them -- the
    second can happen even when the first succeeded, which is why they are
    reported separately. `par` is None unless `stage` is 'ok'. A sample that
    names the Delta scalar sector twice RAISES instead of being scored: that
    is a malformed call rather than a point of parameter space, and no
    sampler reaches it without a bug in how its axes were declared.
    """
    nmp, sector = _split_sample(dict(nmp), hyperon_potentials, U_Delta)
    par, status = invert_nmp(nmp)       # not from_nmp: this scores failures
    if not status.ok:
        return None, "inversion_failed", status.message
    try:
        if sector["su6"]:
            par = replace(par, **sector["su6"])
        if flags.hyperons:
            par = from_hyperon_potentials(
                base=par, **sector["hyperon_potentials"])
        if flags.deltas:
            par = from_delta_potential(
                U_Delta=sector["U_Delta"] if sector["U_Delta"] is not None
                else DEFAULT_U_DELTA,
                x_Delta_sigma=sector["x_Delta_sigma"],
                x_Delta_omega=sector["x_Delta_omega"],
                x_Delta_rho=sector["x_Delta_rho"], base=par)
    except Exception as exc:
        return None, "sectors_failed", f"{type(exc).__name__}: {exc}"
    return par, "ok", ""
