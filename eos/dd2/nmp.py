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

  1. Isoscalar, CLOSED FORM and iterating nothing: Gamma_sigma and
     Gamma_omega follow from (n_sat, E_sat, m*/m) alone -- that part of the
     system is triangular -- and then {P(n_sat)=0, K_sat} fix the sigma and
     omega slopes at saturation, K_sat being an exact quadratic in
     f'_sigma(1) along the P=0 line. The two shape coefficients c_sigma
     and c_omega are PINNED at their published values -- see "Why two
     coefficients are pinned" below. m_sigma is fixed; a_i, d_i are derived
     internally (from_microscopic).
  2. Isovector, also closed form: Gamma_rho(n_sat) from E_sym, then a_rho
     from L_sym, which is affine in it.
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

THERE IS NO SEED ANY MORE, and the machinery that guarded against one is
gone with it. The isoscalar sector used to be a 4-D Powell hybrid seeded at
the published couplings, with 32 jittered restarts on a miss and a `_stalled`
guard for the case where hybr gives up on its first step and hands its own
starting point back as an answer. All three are retired, because the system
they solved turned out to be TRIANGULAR:

    Gamma_sigma(n_sat), Gamma_omega(n_sat)   closed forms in (n_sat, E_sat,
                                             m*/m); no shape dependence at all
    P(n_sat) = 0                             LINEAR in the two f'(1)
    K_sat                                    QUADRATIC in f'_sigma(1)

so the default closure is a quadratic formula and the Q_sat closure is one
scalar root. Measured over 400 draws from a 7-NMP inference prior: 400/400
reproduce their targets (against 394/400 before), worst round trip 5.7e-12,
and the wall clock falls 88x on the default closure and 13x with Q_sat
imposed -- the old solver's cost was almost entirely its tail, 5.4 s at the
worst single target against 2.2 ms now.

WHAT THAT FIXED IS NOT SPEED. The seed did not only cost time, it chose the
ANSWER: the map from NMPs to couplings is many-to-one, and over 117 targets
from the same prior, 53 came back with DIFFERENT couplings depending on which
seed was used -- all of them reproducing the same six nuclear-matter
parameters to better than 1e-6. A posterior built that way mixes branches on
a criterion that is not in the physics. The closed form has two roots too --
the quadratic does -- but it takes the one on DD2's side of the vertex, by a
rule written down at the branch and the same for every target.

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
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq

from eos.general.physics_constants import hc3
from eos.dd2.couplings import (
    SU6_HYPERON, DD2Y_HYPERON, MULTIPLET, vector_ratios, _POTENTIAL_KEY,
    scalar_ratio_from_potential, potential_from_scalar_ratio,
    rational_f, rational_df, rational_d2f, rational_d3f,
    derived_a, derived_d,
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
#: Residual floor below which a closed-form inversion counts as converged.
#:
#: The inverse map no longer iterates on the isoscalar sector, so this is not a
#: solver gate: it is a CHECK. `invert_nmp` evaluates its own conditions at the
#: couplings it returns and reports the worst row, and anything above this is a
#: bug in the algebra rather than a target that could not be reached. Measured
#: over 400 draws from a 7-NMP inference prior, the residual is at or below
#: 6e-12 on every one; 1e-8 is four orders above that and still far below any
#: miss the old iterative closure produced.
ISO_GATE = 1e-8

#: The isoscalar shape coefficients held at their published DD2 values
#: (Typel et al. 2010), one tuple per closure.
#:
#: WHY c_sigma AND NOT b_sigma UNDER THE DEFAULT CLOSURE. What the saturation
#: rows constrain is the TAYLOR DATA of Gamma_i at n_sat, and (b_i, c_i) are
#: folded coordinates for it: with b_sigma held, c_sigma -> f'_sigma(1) turns
#: over at c_sigma ~ 0.07, and with c_omega held, b_omega -> f'_omega(1) turns
#: over TWICE, at b_omega ~ -0.40 and ~ -0.31. Inversions over the empirical
#: box produce b_omega from -39 to +42, straight through both folds, which is
#: why the old 4-D solve returned a different root depending on where it
#: started -- 53 of 117 targets, measured.
#:
#: The ratio r_i = f''_i(1)/f'_i(1) depends on c_i ALONE and is strictly
#: monotone (see `r_of_c`), so PINNING c_i IS PINNING r_i, and with r_i held
#: the conditions become polynomial in f'_i(1) with an explicit branch rule.
#: Same count of held coefficients, same physics content -- two shape
#: directions no saturation row can see -- in a coordinate that does not fold.
#:
#: Under impose_Q_sat the held set is unchanged from the iterative closure:
#: c_omega alone, with b_sigma, c_sigma and b_omega all fitted.
PINNED_DEFAULT = ("c_sigma", "c_omega")
PINNED_WITH_Q_SAT = ("c_omega",)

#: Every shape coefficient either closure ever holds, which is what separates
#: "this closure fits that one" from "that is not a shape coefficient at all"
#: in the error `invert_nmp` raises.
_SHAPE_COEFFICIENTS = frozenset(PINNED_DEFAULT) | frozenset(PINNED_WITH_Q_SAT) \
    | frozenset(("b_sigma", "b_omega"))


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
    #: How far the recovered couplings sit from DD2's published ones, max
    #: relative over the fitted isoscalar set. Kept because "converged" and
    #: "recovered the published couplings" are different statements, and a
    #: caller inverting a moved target still wants to be told how far it went.
    #: No longer a convergence diagnostic: there is no seed to leave.
    coupling_shift: float = float("nan")


# --------------------------------------------------------- the closed forms
#
# THE ISOSCALAR SYSTEM IS TRIANGULAR, and that is the whole of this section.
#
# Write S = m_N - m*, n_s = n_s(m*, kF) and G = Gamma_sigma^2/m_sigma^2,
# W = Gamma_omega^2/m_omega^2 as above. m* and kF are fixed by the TARGETS
# (m*/m and n_sat), so n_s is too, and the gap equation and the energy density
#
#     S = G n_s                       ->  G = S / n_s
#     eps = eps_kin + S n_s / 2 + W n^2 / 2   ->  W = 2(eps - eps_kin - S n_s/2)/n^2
#
# (using S^2/(2G) = S n_s / 2) give Gamma_sigma(n_sat) and Gamma_omega(n_sat)
# with NO dependence on the shape coefficients whatever. Two of the old 4x4's
# four unknowns never needed a solver. Verified bit for bit: across shape
# coefficients drawn at random at fixed (n_sat, E_sat, m*/m), the recovered
# gammas have spread 0.0 and 5.9e-15.
#
# What is left is the shape, and there the conditions are polynomial:
#
#   P(n_sat) = 0   is LINEAR in (f'_sigma(1), f'_omega(1)) and does not
#                  involve the second derivatives at all,
#   K_sat          is QUADRATIC in f'_sigma(1) and LINEAR in f''_sigma(1),
#   Q_sat          brings in f'''(1), which the two-parameter rational form
#                  does not leave free -- it is a function of (f'(1), f''(1)).
#
# So the default closure is a quadratic formula and the Q_sat closure is one
# scalar root find. Neither has a seed.


def eps_kin_snm(m, kF):
    """Kinetic energy density [MeV^4] of the g = 4 nucleon gas at T = 0."""
    E = np.sqrt(kF ** 2 + m ** 2)
    L = np.arcsinh(kF / m)
    return _G_SNM / (16.0 * np.pi ** 2) * (kF * E * (2.0 * kF ** 2 + m ** 2)
                                           - m ** 4 * L)


def r_of_c(c):
    """f''(1)/f'(1) of the Typel-Wolter rational form. A function of c ALONE.

    With d = 1/sqrt(3c) and a fixed by f(1) = 1, both derivatives carry the
    same factor a(b - c), so it cancels from the ratio. Strictly decreasing
    over c in (0, inf), which is what makes `c_of_r` single-valued.
    """
    d = derived_d(c)
    U = (1.0 + d) ** 2
    return (1.0 - 3.0 * c * U) / ((1.0 + d) * (1.0 + c * U))


#: Range of r over c in (0, inf). r is strictly decreasing, so these are its
#: limits and any target outside them is unreachable BY THE FUNCTIONAL FORM --
#: a refusal about DD2's rational ansatz, not about the numerics.
R_MIN, R_MAX = r_of_c(1e8), r_of_c(1e-8)


def c_of_r(r):
    """Invert r(c). CLOSED FORM: this inversion is a cubic, not a root find.

    In t = d = 1/sqrt(3c) the ratio is RATIONAL -- the square roots that make
    r(c) look transcendental are exactly the ones d already carries:

        r = -3(1 + 2t) / ((1 + t)(4t^2 + 2t + 1))

    so clearing the denominator leaves 4r t^3 + 6r t^2 + (3r + 6) t + (r + 3),
    and depressing that with t = y - 1/2 leaves y^3 + p y + q with p = 3/(2r)
    and q = 1/8 CONSTANT. r is confined to (-3, 0) by R_MIN/R_MAX, so the
    discriminant is positive for every reachable ratio: three real roots, of
    which exactly one clears t > 0, with no case split to get wrong. One
    Newton step on the cubic polishes the trigonometric root, which holds the
    1e-14 the brentq this replaced was asked for.
    """
    if not (R_MIN < r < R_MAX):
        raise ValueError(
            f"f''(1)/f'(1) = {r:.6g} is outside ({R_MIN:.4f}, {R_MAX:.4f}), the "
            f"range the Typel-Wolter form can realise: no (b, c) has this "
            f"curvature-to-slope ratio at saturation")
    # k = 0 of y = 2 sqrt(-p/3) cos(arccos((3q/2p) sqrt(-3/p))/3 - 2 pi k/3):
    # the largest root, and the only one left positive by t = y - 1/2.
    t = math.sqrt(-2.0 / r) * math.cos(math.acos(
        max(-1.0, min(1.0, r * math.sqrt(-2.0 * r) / 8.0))) / 3.0) - 0.5
    f = ((4.0 * r * t + 6.0 * r) * t + 3.0 * r + 6.0) * t + r + 3.0
    df = (12.0 * r * t + 12.0 * r) * t + 3.0 * r + 6.0
    if df != 0.0:
        t -= f / df
    return 1.0 / (3.0 * t * t)


def bc_from_taylor(f1, f2):
    """(f'(1), f''(1)) -> (b, c), closed form.

    r = f''/f' gives c, then A = a(b - c) from f'(1), then b algebraically.
    Round-trips against `rational_df`/`rational_d2f` to a median 5.9e-16 over
    3000 random shapes, with no failures.
    """
    if f1 == 0.0:
        raise ValueError("f'(1) = 0: the coupling is flat at saturation and "
                         "the rational form's shape is undetermined")
    c = c_of_r(f2 / f1)
    d = derived_d(c)
    U = (1.0 + d) ** 2
    D = 1.0 + c * U
    A = f1 * D ** 2 / (2.0 * (1.0 + d))          # A = a (b - c)
    den = D - A * U
    if den == 0.0:
        raise ValueError("degenerate shape: no finite b realises this f'(1)")
    return (c * D + A) / den, c


def b_at_pinned_c(f1, c):
    """b from f'(1) at a c the caller PINNED, without recovering c from r.

    `bc_from_taylor` inverts r(c) numerically, which returns a pinned c to
    1e-16 rather than exactly. A held coefficient must come back bit for bit
    -- a caller who pinned 0.9 and reads 0.9000000000000009 cannot tell a pin
    from a fit -- so wherever c is known, only b is computed.
    """
    d = derived_d(c)
    U = (1.0 + d) ** 2
    D = 1.0 + c * U
    A = f1 * D ** 2 / (2.0 * (1.0 + d))
    den = D - A * U
    if den == 0.0:
        raise ValueError("degenerate shape: no finite b realises this f'(1)")
    return (c * D + A) / den


def s_of_c(c):
    """f'''(1)/f'(1). A function of c ALONE -- b does not enter.

    With u = (x+d)^2 the form is a(1+bu)/(1+cu), whose u-derivative is
    a(b-c)/(1+cu)^2, so EVERY x-derivative at x = 1 carries the same overall
    factor a(b-c) and their ratios drop it. The same cancellation that makes
    r = f''/f' depend on c alone makes f'''/f' depend on c alone:

        s(c) = 12 c (cU - 1) / D^2,   U = (1+d)^2,  D = 1 + cU

    Verified against `rational_d3f`/`rational_df` at 3000 random (b, c) to a
    max relative error of 8.6e-16.
    """
    d = derived_d(c)
    U = (1.0 + d) ** 2
    return 12.0 * c * (c * U - 1.0) / (1.0 + c * U) ** 2


def _f3(f1, f2, c=None):
    """f'''(1), which the two-parameter form does NOT leave free.

    `c` short-circuits the r -> c inversion where the caller already knows it,
    which is every pinned channel. Neither branch needs a or b any more: by
    `s_of_c` the third derivative is just f'(1) times a function of c, which
    takes `derived_a` and `rational_d3f` out of the Q_sat scan entirely.
    """
    if c is None:
        c = c_of_r(f2 / f1)
    return f1 * s_of_c(c)


def _isoscalar_at_saturation(G, W, n, kF, m, ns_all, fs1, fs2, fw1, fw2,
                             want_Q=False, c_s=None, c_w=None):
    """(P, K_sat, Q_sat) at n_sat from the couplings' Taylor data at n_sat.

    The same closed forms `snm_derivatives` uses, written over (f', f'')
    instead of over a built `Parameters`, so the inverse map can evaluate them
    without constructing one. Solves nothing: every quantity here is algebra
    on the ns partials, which depend only on (m*, kF).
    """
    ns, ns_m, ns_k, ns_mm, ns_mk, ns_kk = ns_all
    G1 = 2.0 * G * fs1 / n
    G2 = 2.0 * G * (fs1 ** 2 + fs2) / n ** 2
    W1 = 2.0 * W * fw1 / n
    W2 = 2.0 * W * (fw1 ** 2 + fw2) / n ** 2

    kF1 = kF / (3.0 * n)
    kF2 = -2.0 * kF / (9.0 * n ** 2)
    den = 1.0 + G * ns_m
    S1 = (G1 * ns + G * ns_k * kF1) / den
    dns = -ns_m * S1 + ns_k * kF1

    E = np.sqrt(kF ** 2 + m ** 2)
    E1 = (kF * kF1 - m * S1) / E
    mu = E + W * n + 0.5 * W1 * n ** 2 - 0.5 * G1 * ns ** 2
    mu1 = (E1 + W + 2.0 * W1 * n + 0.5 * W2 * n ** 2
           - 0.5 * G2 * ns ** 2 - G1 * ns * dns)
    # eps = eps_kin + S^2/(2G) + W n^2/2, and S^2/(2G) = S n_s/2 by the gap
    P = mu * n - (eps_kin_snm(m, kF) + 0.5 * (G * ns) * ns + 0.5 * W * n ** 2)
    if not want_Q:
        return P, 9.0 * n * mu1, None

    G3 = 2.0 * G * (3.0 * fs1 * fs2 + _f3(fs1, fs2, c_s)) / n ** 3
    W3 = 2.0 * W * (3.0 * fw1 * fw2 + _f3(fw1, fw2, c_w)) / n ** 3
    S2 = (G2 * ns + 2.0 * G1 * dns
          + G * (ns_mm * S1 ** 2 - 2.0 * ns_mk * S1 * kF1
                 + ns_kk * kF1 ** 2 + ns_k * kF2)) / den
    d2ns = (ns_mm * S1 ** 2 - 2.0 * ns_mk * S1 * kF1 + ns_kk * kF1 ** 2
            + ns_k * kF2 - ns_m * S2)
    E2 = ((kF1 ** 2 + kF * kF2 + S1 ** 2 - m * S2) / E - E1 ** 2 / E)
    mu2 = (E2 + 3.0 * W1 + 3.0 * W2 * n + 0.5 * W3 * n ** 2
           - 0.5 * G3 * ns ** 2 - 2.0 * G2 * ns * dns
           - G1 * (dns ** 2 + ns * d2ns))
    return P, 9.0 * n * mu1, 27.0 * n * (n * mu2 - 3.0 * mu1)


def _esym_kinetic_slope(G, n, kF, m, ns_all, fs1):
    """d/dn of the KINETIC part of E_sym at n_sat, in natural units.

    `snm_derivatives` writes E_sym = kF^2/(6 E_F*) + R n / 2 and differentiates
    the first term through E_F*(n), which moves with the Dirac mass. Every
    ingredient is already fixed by the targets and the sigma shape, so this
    needs no solve -- which is what lets `a_rho` be a closed form below.
    """
    ns, ns_m, ns_k = ns_all[0], ns_all[1], ns_all[2]
    G1 = 2.0 * G * fs1 / n
    kF1 = kF / (3.0 * n)
    S1 = (G1 * ns + G * ns_k * kF1) / (1.0 + G * ns_m)
    E = np.sqrt(kF ** 2 + m ** 2)
    E1 = (kF * kF1 - m * S1) / E
    u = kF ** 2
    u1 = 2.0 * u / (3.0 * n)
    return u1 / (6.0 * E) - u * E1 / (6.0 * E ** 2)


def _gammas_and_state(nmp, m_N, m_sigma, m_omega):
    """The triangular half: Gamma_sigma, Gamma_omega and the fixed state."""
    m = nmp["m_eff_ratio"] * m_N
    S = m_N - m
    n = nmp["n_sat"] * hc3
    kF = kF_from_n(n, _G_SNM)
    ns_all = _ns_partials(m, kF)
    G = S / ns_all[0]
    W = 2.0 * (n * (nmp["E_sat"] + m_N) - eps_kin_snm(m, kF)
               - 0.5 * S * ns_all[0]) / n ** 2
    if W <= 0.0:
        raise ValueError(
            f"NMP inversion infeasible: the omega coupling squared comes out "
            f"{W * m_omega ** 2:.4g} < 0 -- at m*/m = {nmp['m_eff_ratio']} the "
            f"sigma field alone already binds more than E_sat = "
            f"{nmp['E_sat']} MeV, so no repulsion can be fitted")
    return m_sigma * np.sqrt(G), m_omega * np.sqrt(W), G, W, m, S, n, kF, ns_all


def invert_nmp(nmp, m_sigma=546.212459, seed=None, n_restarts=None,
               impose_Q_sat=False, pinned=None):
    """Recover DD2 couplings from a target NMP dict.

    nmp needs {n_sat, E_sat, m_eff_ratio, K_sat, E_sym, L_sym}; "Q_sat" is
    consumed only when it is imposed. Returns (Parameters, InversionStatus).
    Raises ValueError on a hard infeasibility -- m*/m outside the physical
    window, an E_sat the sigma field already overshoots, a K_sat or Q_sat past
    the closure's own reachable bound, or E_sym below the kinetic symmetry
    energy. A closure that cannot be met is reported through status.ok=False
    with no parametrization: there is no meaningful coupling set to hand back,
    and the isovector sector is never fitted on a garbage point.

    THIS INVERSION DOES NOT ITERATE ON THE ISOSCALAR SECTOR and has no seed.
    Gamma_sigma(n_sat) and Gamma_omega(n_sat) are closed forms in
    (n_sat, E_sat, m*/m) alone -- the old 4x4 was triangular and nobody had
    noticed -- and the shape conditions are polynomial in the couplings'
    Taylor data at saturation. `seed` and `n_restarts` are accepted and
    IGNORED, kept only so existing callers do not break; they name a solver
    that is gone.

    impose_Q_sat selects the isoscalar closure:
      False - the default: Q_sat is a PREDICTION. c_sigma and c_omega are
              held, {P(n_sat)=0, E_sat, m*/m, K_sat} are imposed, and K_sat
              is an exact quadratic in f'_sigma(1) along the P=0 line, so the
              shape comes from the quadratic formula. Two roots exist and
              both reproduce all four rows; the one on DD2's side of the
              vertex is returned, which is the whole of the choice the old
              solver made implicitly through which basin its seed fell into.
      True  - Q_sat is imposed: c_omega alone is held -- the same pin the
              iterative closure used -- and b_sigma, c_sigma, b_omega are all
              fitted. P=0 still fixes f'_omega(1) linearly and K_sat
              fixes f''_sigma(1) as an exact QUADRATIC in f'_sigma(1),
              so what is left is ONE scalar equation in f'_sigma(1).
              Q_sat reaches it through f'''(1), which the two-parameter
              rational form does not leave free, and that equation is
              ALGEBRAIC rather than rational: it is bracketed and solved,
              not written down. What IS closed form is the set of
              f'_sigma(1) the form can realise (`_feasible_intervals`), so
              the bracket search runs only on admissible ground and cannot
              lose a root off the edge of it. Two to four roots exist over
              the empirical box; the one nearest the published f'_sigma(1)
              is returned, the same branch rule the default closure applies
              through its vertex.

    `pinned` sets the HELD shape coefficients -- {"c_sigma": ..., "c_omega":
    ...} for the default closure, {"c_omega": ...} when Q_sat is imposed --
    and defaults to the published DD2 values. They are held rather than fitted
    because the saturation rows carry shape information only through P and
    K_sat, so pinning them is a CHOICE the caller is entitled to make
    differently. It is not a cosmetic one: the six NMPs are reproduced to
    ~1e-12 whatever the held pair is, because the shape acts above saturation
    and the imposed rows do not, so it is a direction an NMP likelihood cannot
    see and a stellar one can. Naming a coefficient this closure FITS raises:
    pinning and fitting the same number are two different requests.
    """
    if impose_Q_sat and "Q_sat" not in nmp:
        raise ValueError("impose_Q_sat=True but the NMP dict carries no Q_sat")
    # Feasibility: m*/m too small drives Gamma_sigma sigma -> m_N
    # (scalar collapse); outside a physical RMF window there is no DD2-form fit.
    if not (0.35 < nmp["m_eff_ratio"] < 0.95):
        raise ValueError(
            f"NMP inversion infeasible: m*/m = {nmp['m_eff_ratio']} outside the "
            f"physical (0.35, 0.95) window (scalar collapse / no DD2-form fit)")

    ref = Parameters.default()
    names = PINNED_WITH_Q_SAT if impose_Q_sat else PINNED_DEFAULT
    held = {name: getattr(ref, name) for name in names}
    for name, value in (pinned or {}).items():
        if name not in names:
            fitted = "fits" if name in _SHAPE_COEFFICIENTS else "does not carry"
            raise ValueError(
                f"pinned={{{name!r}: ...}}: this closure {fitted} "
                f"{name!r}; it holds {names}. With impose_Q_sat="
                f"{impose_Q_sat} b_sigma is "
                f"{'fitted' if impose_Q_sat else 'fitted'} and c_sigma is "
                f"{'fitted' if impose_Q_sat else 'held'}.")
        held[name] = float(value)

    Gs, Gw, G, W, m, S, n, kF, ns_all = _gammas_and_state(
        nmp, 0.5 * (ref.m_n + ref.m_p), m_sigma, ref.m_omega)
    ns = ns_all[0]
    r_w = r_of_c(held["c_omega"])

    # P(n_sat) = 0 is linear in (f'_sigma(1), f'_omega(1)) and carries no
    # second derivative at all:  W n^2 fw1 - G n_s^2 fs1 = rhs
    E_F = float(np.sqrt(kF ** 2 + m ** 2))
    rhs = eps_kin_snm(m, kF) + 0.5 * S * ns - n * E_F - 0.5 * W * n ** 2
    coef_w, coef_s = W * n ** 2, -G * ns ** 2

    def fw1_of(fs1):
        return (rhs - coef_s * fs1) / coef_w

    if not impose_Q_sat:
        r_s = r_of_c(held["c_sigma"])

        def K_of(fs1):
            fw1 = fw1_of(fs1)
            return _isoscalar_at_saturation(G, W, n, kF, m, ns_all, fs1,
                                            r_s * fs1, fw1, r_w * fw1)[1]

        # EXACTLY quadratic: three evaluations determine it, to 2.4e-15.
        k0, kp, km = K_of(0.0), K_of(1.0), K_of(-1.0)
        A = 0.5 * (kp + km) - k0
        B = 0.5 * (kp - km)
        C = k0 - nmp["K_sat"]
        if A == 0.0:
            if B == 0.0:
                raise ValueError(
                    "NMP inversion infeasible: K_sat does not depend on the "
                    "sigma shape at this (n_sat, E_sat, m*/m)")
            fs1 = -C / B
        else:
            disc = B * B - 4.0 * A * C
            if disc < 0.0:
                extremum = k0 - B * B / (4.0 * A)
                raise ValueError(
                    f"NMP inversion infeasible: K_sat = {nmp['K_sat']} is past "
                    f"the {'minimum' if A > 0 else 'maximum'} "
                    f"{extremum:.2f} MeV this closure can reach at "
                    f"(n_sat, E_sat, m*/m) = ({nmp['n_sat']}, {nmp['E_sat']}, "
                    f"{nmp['m_eff_ratio']})")
            # THE BRANCH RULE, explicit. Both roots reproduce all four rows;
            # they are two parametrizations of the same nuclear matter that
            # differ above saturation. DD2's own sits on the far side of the
            # vertex from the origin, and that is the branch returned.
            root = np.sqrt(disc)
            fs1 = (-B + root) / (2.0 * A) if A > 0 else (-B - root) / (2.0 * A)
        fs2 = r_s * fs1
    else:
        # c_omega alone is held. P=0 fixes f'_omega(1) from f'_sigma(1), and
        # K_sat is LINEAR in f''_sigma(1) at fixed f'_sigma(1), so f''_sigma(1)
        # is closed form too. One scalar equation is left: Q_sat.
        # K_sat enters f''_sigma(1) only through G2 = 2G(fs1^2 + fs2)/n^2, so
        # dK_sat/d f''_sigma(1) = -9 G n_s^2 / n EXACTLY, independent of
        # f'_sigma(1), and K_sat at fs2 = 0 is exactly quadratic in
        # f'_sigma(1). Together those make f''_sigma(1) a closed-form
        # QUADRATIC in f'_sigma(1): three evaluations pin it here, and the
        # Q_sat scan below then evaluates K_sat not once.
        slope = -9.0 * G * ns ** 2 / n
        if slope == 0.0:
            raise ValueError("NMP inversion infeasible: K_sat does not "
                             "depend on f''_sigma(1) here")

        def _k_at_zero(fs1):
            fw1 = fw1_of(fs1)
            return _isoscalar_at_saturation(G, W, n, kF, m, ns_all,
                                            fs1, 0.0, fw1, r_w * fw1)[1]

        k_z, k_p, k_m = _k_at_zero(0.0), _k_at_zero(1.0), _k_at_zero(-1.0)
        quad = (-(0.5 * (k_p + k_m) - k_z) / slope,
                -(0.5 * (k_p - k_m)) / slope,
                (nmp["K_sat"] - k_z) / slope)

        def fs2_of(fs1):
            return (quad[0] * fs1 + quad[1]) * fs1 + quad[2]

        def q_residual(fs1):
            fw1 = fw1_of(fs1)
            return _isoscalar_at_saturation(
                G, W, n, kF, m, ns_all, fs1, fs2_of(fs1), fw1, r_w * fw1,
                want_Q=True, c_w=held["c_omega"])[2] - nmp["Q_sat"]

        fs1 = _scan_for_root(q_residual, _published_f1(ref), quad)
        if fs1 is None:
            return None, InversionStatus(
                ok=False,
                message=f"no f'_sigma(1) in the scanned range reproduces "
                        f"Q_sat = {nmp['Q_sat']} at K_sat = {nmp['K_sat']}; the "
                        f"two are inconsistent under this closure",
                isoscalar_residual=float("inf"),
                isovector_residual=float("nan"))
        fs2 = fs2_of(fs1)

    fw1 = fw1_of(fs1)
    try:
        # c_omega is always held; c_sigma is held by the default closure and
        # fitted under impose_Q_sat. A held value is returned as it was given.
        c_w = held["c_omega"]
        b_w = b_at_pinned_c(fw1, c_w)
        if impose_Q_sat:
            b_s, c_s = bc_from_taylor(fs1, fs2)
        else:
            c_s = held["c_sigma"]
            b_s = b_at_pinned_c(fs1, c_s)
    except ValueError as err:
        return None, InversionStatus(
            ok=False, message=f"the shape the rows demand is not realisable by "
                              f"the Typel-Wolter form: {err}",
            isoscalar_residual=float("inf"), isovector_residual=float("nan"))

    # The CHECK, not a gate: evaluate the imposed rows at the answer.
    P_at, K_at, Q_at = _isoscalar_at_saturation(
        G, W, n, kF, m, ns_all, fs1, fs2, fw1, r_w * fw1,
        want_Q=impose_Q_sat, c_w=held["c_omega"])
    iso_res = max(abs(P_at) / (hc3 * max(abs(nmp["n_sat"]), 1e-12)),
                  abs(K_at - nmp["K_sat"]) * 1e-2)
    if impose_Q_sat:
        iso_res = max(iso_res, abs(Q_at - nmp["Q_sat"]) * 1e-2)
    shift = max(abs(Gs - ref.gamma_sigma) / ref.gamma_sigma,
                abs(Gw - ref.gamma_omega) / ref.gamma_omega,
                abs(b_s - ref.b_sigma) / abs(ref.b_sigma),
                abs(b_w - ref.b_omega) / abs(ref.b_omega))

    if iso_res >= ISO_GATE:
        return None, InversionStatus(
            ok=False,
            message=f"isoscalar residual {iso_res:.2e} above the "
                    f"{ISO_GATE:.0e} floor -- the closed forms did not "
                    f"reproduce their own conditions, which is an algebra bug "
                    f"rather than an unreachable target",
            isoscalar_residual=iso_res, isovector_residual=float("nan"),
            coupling_shift=shift)

    # --- isovector: Gamma_rho analytic, a_rho by 1-D root -------------------
    kin = kF ** 2 / (6.0 * E_F)
    rho_term = nmp["E_sym"] - kin
    if rho_term <= 0:
        raise ValueError(
            f"NMP inversion infeasible: E_sym={nmp['E_sym']} below the "
            f"kinetic symmetry energy {kin:.2f} MeV (no real Gamma_rho)")
    # E_sym = kF^2/(6 EF*) + Gamma_rho^2 n/(2 m_rho^2)  ->  Gamma_rho analytic
    Grho = float(np.sqrt(rho_term * 2.0 * ref.m_rho ** 2 / n))

    # a_rho IS A CLOSED FORM, not a root. At n = n_sat the rho coupling's
    # exponential is 1 and its log-derivative is k_rho n = 2 a_rho exactly, so
    #
    #     L_sym = 3 n E_s'(n) + 1.5 n R (1 - 2 a_rho),   R = (Gamma_rho/m_rho)^2
    #
    # is AFFINE in a_rho. Measured over a_rho in [-2, 3]: the affine fit's
    # residual is 1.8e-14, i.e. it is an identity and not a fit. Inverting it
    # here removes the last root find from the inversion, and with it the last
    # five symmetric-matter solves: the isoscalar sector already solved none.
    Es1 = _esym_kinetic_slope(G, n, kF, m, ns_all, fs1)
    R = (Grho / ref.m_rho) ** 2
    a_rho = 0.5 * (1.0 - (nmp["L_sym"] - 3.0 * n * Es1) / (1.5 * n * R))

    par = Parameters.from_microscopic(
        n_sat=nmp["n_sat"], gamma_sigma=Gs, b_sigma=b_s, c_sigma=c_s,
        gamma_omega=Gw, b_omega=b_w, c_omega=c_w,
        gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)

    # --- report what the closure predicts, with the forward map's stencils --
    # This call is the one place the inversion still touches symmetric matter,
    # and it is a CHECK rather than a step: L_sym comes back from `eos`'s own
    # forward map, independent of the algebra above, so a mistake in the closed
    # form for a_rho shows up as an isovector residual instead of hiding.
    final = snm_derivatives(par, nmp["n_sat"])
    predictions = {"Q_sat": final["Q_sat"], "K_sym": final["K_sym"]}
    isov_res = abs(final["L_sym"] - nmp["L_sym"])

    status = InversionStatus(
        ok=(isov_res < 1e-3),                # isoscalar check already passed
        message="converged" if isov_res < 1e-3 else
        f"isovector residual {isov_res:.2e} above 1e-3",
        isoscalar_residual=iso_res, isovector_residual=float(isov_res),
        predictions=predictions, coupling_shift=shift)
    return par, status


def _published_f1(ref):
    """f'_sigma(1) of the published DD2 parametrization, the scan's anchor."""
    return rational_df(1.0, ref.a_sigma, ref.b_sigma, ref.c_sigma, ref.d_sigma)


#: Where the Q_sat closure looks for its root, and how finely.
#:
#: Q_sat(f'_sigma(1)) is not a polynomial -- f'''(1) reaches it through the
#: rational form -- so the one scalar equation left is bracketed rather than
#: solved. The window is centred on the published f'_sigma(1) = -0.1298 and
#: spans the range inversions over the empirical box actually produce; `steps`
#: is the resolution ACROSS THE FEASIBLE PART of it, which `_scan_for_root`
#: computes rather than probes. Roots are taken NEAREST THE PUBLISHED VALUE,
#: which is the same branch rule the default closure applies through its vertex.
#:
#: `steps` was 30 when the grid still had to cover infeasible ground. Measured
#: over 400 draws from the wide empirical box against a 1500-point reference:
#: 18 and 14 both agree 400/400, 10 first disagrees (398/400) and 8 falls to
#: 362/400. 18 keeps a 1.8x margin on the first failure for four evaluations.
_Q_SCAN = (-3.0, 1.5, 18)


def _quad_roots(a, b, c):
    """Real roots of a x^2 + b x + c, degenerate cases included."""
    if a == 0.0:
        return () if b == 0.0 else (-c / b,)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return ()
    root = math.sqrt(disc)
    return tuple(sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))))


def _feasible_intervals(quad, lo, hi):
    """The f'_sigma(1) for which the sigma shape is REALISABLE, in closed form.

    f''_sigma(1) is the quadratic `quad` in f'_sigma(1), so the curvature-to-
    slope ratio r = f''/f' is a rational function of it and the admissibility
    test R_MIN < r < R_MAX is two QUADRATIC inequalities. Their roots, plus the
    pole at f'_sigma(1) = 0, cut the window into at most four pieces, each
    wholly in or wholly out -- so one midpoint test per piece settles it.

    The scan used to discover this domain by catching ValueError on a fixed
    grid, which silently discarded any cell with one endpoint outside: a root
    adjacent to the boundary was then lost, and which roots survived depended
    on how the grid happened to land. Checked against 90000 brute-force probes
    with no disagreement.
    """
    qa, qb, qc = quad
    cuts = {lo, hi, 0.0}
    for bound in (R_MIN, R_MAX):
        cuts.update(r for r in _quad_roots(qa, qb - bound, qc) if lo < r < hi)
    cuts = sorted(x for x in cuts if lo <= x <= hi)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (a + b)
        if mid == 0.0 or b - a <= 0.0:
            continue
        if R_MIN < ((qa * mid + qb) * mid + qc) / mid < R_MAX:
            if out and a - out[-1][1] <= 0.0:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
    return out


def _scan_for_root(residual, ref_f1, quad):
    """Bracket and solve `residual`, returning the root nearest `ref_f1`.

    The brackets are collected first and solved NEAREST-FIRST, stopping once
    the root in hand is closer than the nearest edge of every bracket left.
    The rule is the same one as before -- the root nearest `ref_f1` -- but a
    scan over the empirical box carries two to four brackets and this solves
    one of them, where solving all four and discarding three was most of the
    cost left in the closure.
    """
    lo, hi, steps = _Q_SCAN
    brackets = []
    for a, b in _feasible_intervals(quad, lo, hi):
        # r hits its bound exactly at an endpoint, so step just inside it.
        pad = 1e-12 * max(b - a, 1.0)
        a, b = a + pad, b - pad
        if b <= a:
            continue
        count = max(3, int(round((b - a) / (hi - lo) * steps)) + 1)
        points = np.linspace(a, b, count)
        values = []
        for x in points:
            try:
                values.append(residual(float(x)))
            except (ValueError, ZeroDivisionError, FloatingPointError):
                values.append(np.nan)
        for i in range(count - 1):
            u, v = values[i], values[i + 1]
            if not (np.isfinite(u) and np.isfinite(v)) or u * v > 0.0:
                continue
            brackets.append((float(points[i]), float(points[i + 1])))

    def _gap(bracket):
        return max(bracket[0] - ref_f1, ref_f1 - bracket[1], 0.0)

    brackets.sort(key=_gap)
    best = None
    for a, b in brackets:
        # Every remaining bracket starts at least `_gap` away, so nothing left
        # can beat what is already in hand.
        if best is not None and abs(best - ref_f1) <= _gap((a, b)):
            break
        try:
            root = brentq(residual, a, b, xtol=1e-14, rtol=8.9e-16)
        except (ValueError, ZeroDivisionError):
            continue
        if best is None or abs(root - ref_f1) < abs(best - ref_f1):
            best = root
    return best


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

def forward_maps(par):
    """Everything `build_parametrization` takes, computed back off `par`.

    The three forward maps in one call -- `compute_nmp` for the nuclear-matter
    parameters, `hyperon_potentials` and `delta_potential` for the depths --
    keyed in the SAME vocabulary a sample uses, so the result can be handed
    straight back to `build_parametrization` and should reproduce the
    couplings it came from. That round trip is what the function is for; the
    three pieces are separately available when only one is wanted.

    Sectors the par does not carry are simply absent: a nucleonic set gets no
    U_Y keys, and one with no Delta quartet no U_Delta. Q_sat and K_sym come
    along from `compute_nmp` as predictions and are NOT part of the round trip
    unless the caller also sets impose_Q_sat.
    """
    out = dict(compute_nmp(par))
    out.update(hyperon_potentials(par))
    if par.x_Delta_sigma is not None:
        out["U_Delta"] = delta_potential(par)
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
               + SU6_FACTOR_KEYS + PINNED_DEFAULT)

#: The Delta depth `from_delta_potential` publishes, used when a sample names
#: neither side of the scalar pair.
DEFAULT_U_DELTA = -50.0


def _split_sample(sample, hyperon_potentials=None, U_Delta=None, pinned=None):
    """Separate a sample dict into (nmp, sector kwargs).

    A sample may carry any of the `SECTOR_KEYS` next to the nuclear-matter
    parameters; those override the corresponding keyword arguments, key by key
    rather than dict by dict, so naming one held coefficient in the sample
    does not discard the other from `pinned`. This is
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
              "pinned": {**{k: float(v) for k, v in (pinned or {}).items()},
                         **{k: float(sample[k]) for k in PINNED_DEFAULT
                            if k in sample}},
              "U_Delta": None if depth is None else float(depth),
              "x_Delta_sigma": None if ratio is None else float(ratio),
              "x_Delta_omega": float(sample.get("x_Delta_omega", 1.0)),
              "x_Delta_rho": float(sample.get("x_Delta_rho", 1.0))}
    return nmp, sector


def build_parametrization(nmp, flags, hyperon_potentials=None,
                          U_Delta=None, pinned=None, impose_Q_sat=False):
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

    The shape coefficients the closure HOLDS rather than fits -- `b_sigma` and
    `c_omega`, `PINNED_DEFAULT` -- are the `pinned` argument, and may equally
    ride in the sample like any other coupling knob; the sample wins per key,
    the same precedence `hyperon_potentials` follows. Both default to the
    published DD2 values. They are worth an
    axis: the six NMPs come back to ~1e-12 whatever `b_sigma` is, so an NMP
    likelihood cannot see it, while over +-30% it moves M_max by ~0.18 M_sun.
    `c_omega` over the same range moves it by ~0.02, which is why one of them
    is a real free direction and the other is very nearly not.

    `impose_Q_sat` is passed straight to `invert_nmp` and picks the isoscalar
    closure: False (the default) predicts Q_sat and pins `PINNED_DEFAULT`,
    True imposes it, pins only `PINNED_WITH_Q_SAT` and needs "Q_sat" in the
    sample. Read `invert_nmp` before using True -- the Q_sat row is a third
    finite difference and the closure amplifies its noise floor by ~259.

    Returns `(par, stage, message)`. `stage` is 'ok', 'inversion_failed' when
    the NMPs have no DD-RMF realisation at all, or 'sectors_failed' when they
    do but the hyperon/Delta scalar inversion does not converge on them -- the
    second can happen even when the first succeeded, which is why they are
    reported separately. `par` is None unless `stage` is 'ok'. A sample that
    names the Delta scalar sector twice RAISES instead of being scored: that
    is a malformed call rather than a point of parameter space, and no
    sampler reaches it without a bug in how its axes were declared.
    """
    nmp, sector = _split_sample(dict(nmp), hyperon_potentials, U_Delta, pinned)
    # not from_nmp: this scores failures. impose_Q_sat selects the isoscalar
    # closure and is the caller's, not an inference from the sample carrying a
    # Q_sat key -- see `invert_nmp`, which says why that inference was wrong.
    par, status = invert_nmp(nmp, impose_Q_sat=impose_Q_sat,
                             pinned=sector["pinned"])
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
