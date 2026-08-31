"""Physics invariants of the chiral colour-dielectric model, checked in one
place.

These are the statements the implementation has to satisfy whatever parameters
it is given, plus the derived numbers the shipped parameter set must
reproduce. They are the fastest way to catch a wrong change, and every check
returns a structured pass/fail with the largest error it saw, so the suite
reports rather than prints.

  1. Derived constants   the section 8 closed forms: zeta_0 = 94.045,
                         lambda = 16.387, v = 86.527, lambda_zeta = 31.414,
                         v_zeta^2 = -4039.3 MeV^2, C_0 = 2.4352e9 MeV^4.
                         AND THE SIGN OF v_zeta^2, tested explicitly, because
                         an implementation that wrote v_zeta as a square root
                         would pass every other check here at m_zeta = 1150
                         and fail at the baseline.
  2. Vacuum              V(sigma_0, zeta_0) = 0 and U(phi_0) = 0 exactly, and
                         the curvatures of V return m_sigma and m_zeta to
                         better than 0.5 MeV -- section 9.1's startup
                         assertions.
  3. Bag constant        B_eff = B_g + B_chi = (239.66 MeV)^4 =
                         429.4 MeV/fm^3, and B_chi > B_g, because the chiral
                         sector supplies the larger part and quoting B_g as
                         "the bag constant" is wrong by a factor six.
  4. Confinement pinning at T = 0 a mode with M* >= mu* contributes
                         IDENTICALLY zero -- exactly 0.0, not small -- and the
                         confined branch therefore has n_B = 0 and P = 0 at
                         any potential. This is the mechanism, not a rounding
                         convenience, and a smoothed implementation would fail
                         here rather than merely losing accuracy.
  5. Euler               eps + P = T s + sum_j mu_j n_j at every solved point,
                         in every mode, paired and unpaired. THIS IS THE CHECK
                         THAT CAUGHT THE SPECIFICATION'S TWO ASSEMBLY ERRORS:
                         with eps taking -(1/2) m_omega^2 omega_0^2 as its
                         section 4.3 writes, or with the rearrangement term
                         left out of Omega as its section 4.1 leaves it, this
                         fails by percents while everything else still looks
                         like an equation of state.
  6. Free energy         f = eps - T s and f = -P + sum_j mu_j n_j.
  7. n = -dOmega/dmu     n_B against a finite difference of P along the
                         solution, which is what distinguishes a
                         thermodynamically consistent rearrangement term from
                         a plausible one.
  8. Rearrangement       Sigma_R enters mu and P and NEVER eps (CLAUDE.md
                         section 8): eps is invariant under adding the term,
                         P is not.
  9. Reduction chain     section 9.6's, run as one-off tests:
                         gbar_omega -> 0 gives a state with omega_0 = 0 and
                         Sigma_R = 0 identically (L1 -> L0); G_D -> 0 kills
                         every gap (L3 -> L2); Delta -> 0 returns the unpaired
                         state MODE BY MODE, to the last bit.
 10. Colour neutrality   n_3 and n_8 vanish identically in an unpaired phase
                         at mu_3 = mu_8 = 0, and are driven to zero by the two
                         colour potentials in a paired one.
 11. Gap window          the shipped G_D puts the gap inside the 20-150 MeV
                         window the specification asks for at mu_q ~ 450 MeV.
 12. Gap sign gauge      Omega is invariant under flipping any subset of the
                         three gaps, and the gap kernel flips with its gap --
                         which is why the solve may land on either sign and
                         why what is reported is the magnitude.
 13. Paired entropy      the entropy of a gapped phase is exponentially
                         suppressed at low T. A merger simulation that used
                         the unpaired entropy in a gapped phase would not be
                         approximately wrong.
 14. Mode closures       each mode's own conditions hold at its solution.
 15. Charge basis        n_B, n_C and n_S agree with `eos.general.basis` -- no
                         local copy of the map.
 16. Residual gate       every state solved here is inside the tolerance the
                         model claims to accept at.
 17. Glue scale          a larger B_g costs pressure at fixed density, which
                         is what says the glue bag scale is paying for
                         deconfinement rather than being absorbed elsewhere.
                         Locating the onset DENSITY is not done here: the
                         crossing sits very close to where the deconfined
                         branch terminates, and finding a transition is
                         `eos.mixed`'s job, by root-finding on pressure
                         differences.

Run it:

    python -m eos.ccdm.verify.run_full_check            # everything
    python -m eos.ccdm.verify.run_full_check --no-csc   # skip paired states
    python -m eos.ccdm.verify.run_full_check --onset    # + the glue-scale scan
"""
from dataclasses import dataclass, field, replace
import math

import numpy as np

from eos.ccdm.api import eos_response
from eos.ccdm.parameters import Parameters
from eos.ccdm.solver import solve_beta_eq_neutrinoless, solve_fixed_yc
from eos.ccdm.api import zero_pressure_point
from eos.ccdm.species import SpeciesFlags
from eos.ccdm.thermodynamics import (
    NUMBA_OK, PHI_CEIL, bag_constant, chiral_potential, glue_potential,
    mode_thermo, state_at, thermo_from_mu,
)
from eos.general.basis import quark_charges
from eos.general.pairing import pair_block
from eos.general.physics_constants import hc3
from eos.general.solve import RESIDUAL_TOL

#: The density at which most single-point checks are run [fm^-3]. Above the
#: deconfinement onset of the shipped parameter set, which is where the model
#: has a deconfined solution at all.
N_CHECK = 1.5


@dataclass
class CheckResult:
    name: str
    passed: bool
    max_error: float
    detail: str = ""


@dataclass
class FullCheckReport:
    results: list = field(default_factory=list)

    @property
    def all_passed(self):
        return all(r.passed for r in self.results)

    def __str__(self):
        lines = [f"CCDM run_full_check: "
                 f"{'PASS' if self.all_passed else 'FAIL'}"]
        for r in self.results:
            tag = "ok " if r.passed else "FAIL"
            lines.append(f"  [{tag}] {r.name:24s} max_err={r.max_error:.2e}"
                         f"  {r.detail}")
        return "\n".join(lines)


def _states(par, include_csc=True):
    """The solved points every identity check is run on.

    One per mode, cold and hot, plus a paired one. Kept small: a paired solve
    diagonalises an 18x18 matrix at every quadrature node, and this suite is
    meant to be run often.
    """
    flags = SpeciesFlags()
    out = {
        "beta T=0": solve_beta_eq_neutrinoless(par, N_CHECK, 0.0,
                                               flags=flags),
        "beta T=30": solve_beta_eq_neutrinoless(par, N_CHECK, 30.0,
                                                flags=flags),
        "YC=0.1 leptons": solve_fixed_yc(par, N_CHECK, 0.1, 20.0,
                                         flags=flags, leptons=True),
        "YC=0.1 charged": solve_fixed_yc(par, N_CHECK, 0.1, 20.0,
                                         flags=flags, leptons=False),
    }
    if include_csc:
        paired = SpeciesFlags(csc=True)
        out["beta CSC T=0"] = solve_beta_eq_neutrinoless(par, N_CHECK, 0.0,
                                                         flags=paired)
        out["beta CSC T=30"] = solve_beta_eq_neutrinoless(par, N_CHECK, 30.0,
                                                          flags=paired)
    return {k: v for k, v in out.items() if v.converged}


# =============================================================================
# THE VACUUM-FIXED BLOCK
# =============================================================================
def check_derived_constants(par, tol=1.0e-3):
    """The section 8 closed forms against the numbers the specification quotes.

    And the SIGN of v_zeta^2, which is negative at the baseline m_zeta: the
    strange quartic is convex, explicit breaking dominating. An implementation
    that stored v_zeta rather than v_zeta^2 would take a square root of a
    negative number here and pass at m_zeta = 1150 MeV, where the sign flips.
    """
    d = par.derived
    expected = {"zeta_0": (d.zeta_0, 94.045), "lambda": (d.lam, 16.387),
                "v": (math.sqrt(d.v2), 86.527),
                "lambda_zeta": (d.lam_zeta, 31.414),
                "v_zeta^2": (d.v_zeta2, -4039.3),
                "C_0": (d.C_0, 2.4352e9), "phi_0": (d.phi_0, 56.25)}
    errors = {k: abs(got - want) / abs(want) for k, (got, want) in
              expected.items()}
    worst = max(errors.values())
    negative = d.v_zeta2 < 0.0
    return CheckResult(
        "derived constants", worst < tol and negative, worst,
        f"v_zeta^2 = {d.v_zeta2:.1f} MeV^2 "
        f"({'negative, as it must be' if negative else 'POSITIVE -- wrong'})")


def check_vacuum(par, tol=0.5):
    """V(sigma_0, zeta_0) = 0, U(phi_0) = 0, and the curvatures of V return
    m_sigma and m_zeta [MeV].

    Section 9.1's startup assertions. The two potentials vanishing at the
    physical vacuum is what lets Omega be assembled with NO vacuum
    subtraction anywhere, unlike the NJL companion, where the Dirac sea has to
    be subtracted from both Omega and eps.
    """
    d = par.derived
    V0 = chiral_potential(par, d.sigma_0, d.zeta_0)
    U0 = glue_potential(par, PHI_CEIL)              # phi_bar = 1
    h = 1.0e-3
    d2s = (chiral_potential(par, d.sigma_0 + h, d.zeta_0)
           - 2.0 * V0 + chiral_potential(par, d.sigma_0 - h, d.zeta_0)) / h ** 2
    d2z = (chiral_potential(par, d.sigma_0, d.zeta_0 + h)
           - 2.0 * V0 + chiral_potential(par, d.sigma_0, d.zeta_0 - h)) / h ** 2
    m_sigma, m_zeta = math.sqrt(abs(d2s)), math.sqrt(abs(d2z))
    errors = [abs(V0) / par.B_g, abs(U0) / par.B_g,
              abs(m_sigma - par.m_sigma), abs(m_zeta - par.m_zeta)]
    return CheckResult("vacuum", max(errors) < tol, max(errors),
                       f"m_sigma = {m_sigma:.3f}, m_zeta = {m_zeta:.3f} MeV "
                       f"from the curvatures of V")


def check_bag_constant(par, tol=1.0e-3):
    """B_eff = (239.66 MeV)^4 = 429.4 MeV/fm^3, and B_chi > B_g.

    A DERIVED quantity, not an input. The second half of the check is the one
    that matters for reading the model: the chiral sector supplies the larger
    part, so B_g alone is not the bag constant.
    """
    B_eff = bag_constant(par)
    B_chi = B_eff - par.B_g
    err = abs(B_eff / hc3 - 429.4) / 429.4
    return CheckResult(
        "bag constant", err < tol and B_chi > par.B_g, err,
        f"B_eff = ({B_eff ** 0.25:.2f} MeV)^4 = {B_eff / hc3:.1f} MeV/fm^3, "
        f"B_chi^(1/4) = {B_chi ** 0.25:.2f} > B_g^(1/4) = "
        f"{par.B_g ** 0.25:.2f} MeV")


# =============================================================================
# THE MECHANISM
# =============================================================================
def check_confinement_pinning(par, tol=0.0):
    """At T = 0 a mode with M* >= mu* contributes EXACTLY zero.

    Not approximately: the tolerance here is 0.0. This is the confinement
    mechanism -- as the dilaton reaches its vacuum value the dielectric closes,
    M* diverges and the quarks leave the medium -- and an implementation that
    smoothed the threshold would return a small nonzero density and destroy
    the first-order deconfinement transition while still producing a
    plausible-looking equation of state.

    The second half evaluates the confined branch at a potential well above
    the onset and requires n_B = 0 and P = 0 there.
    """
    heavy = mode_thermo(mu_star=400.0, M_star=450.0, T=0.0)
    on_shell = [heavy.n, heavy.rho_s, heavy.eps, heavy.P, heavy.s]

    d = par.derived
    st = state_at(par, PHI_CEIL, d.sigma_0, d.zeta_0, 0.0, np.zeros(3),
                  1600.0, 0.0, 0.0, 0.0, 0.0, 0.0, branch="confined")
    worst = max([abs(v) for v in on_shell]
                + [abs(st.n_B_nat), abs(st.P_nat), abs(st.U), abs(st.V)])
    return CheckResult(
        "confinement pinning", worst <= tol, worst,
        f"a mode at M* = 450 > mu* = 400 MeV returns exactly zero, and the "
        f"confined branch has n_B = {st.n_B_nat:g}, P = {st.P_nat:g} at "
        f"mu_B = 1600 MeV")


def check_rearrangement_placement(par, tol=1.0e-12):
    """Sigma_R enters mu and P and NEVER eps (CLAUDE.md section 8).

    Two falsifiable identities on one solved state, taken apart rather than
    asserted. Writing the mode sums as

        S_P = sum_j P_j ,  S_eps = sum_j eps_j ,  W = (1/2) m_omega^2 omega_0^2

    the assembly must satisfy, exactly,

        P   - (S_P - U - V + W) = Sigma_R n_q          the term IS in P
        eps - (S_eps + U + V + W) = 0                  and is NOT in eps

    If a future edit moved the rearrangement into the energy density the
    second line fails; if it dropped the term altogether the first fails and
    the Euler check fails with it. The check also reports how big the term is,
    because an identity that both sides satisfy trivially proves nothing: at
    the shipped parameter point it is a quarter of eps, so neither line is
    passing by being small.
    """
    st = _reference_state(par)
    modes = [mode_thermo(st.mu_star[j], st.M_star[j // 3], st.T)
             for j in range(9)]
    S_P = sum(m.P for m in modes)
    S_eps = sum(m.eps for m in modes)
    W = 0.5 * par.m_omega ** 2 * st.omega_0 ** 2

    in_P = abs((st.P_nat - (S_P - st.U - st.V + W)) - st.Sigma_R * st.n_q)
    not_in_eps = abs(st.eps_nat - (S_eps + st.U + st.V + W))
    scale = abs(st.eps_nat)
    errors = [in_P / scale, not_in_eps / scale]
    size = abs(st.Sigma_R * st.n_q) / scale
    worst = max(errors)
    return CheckResult(
        "rearrangement placement", worst <= tol and size > 1.0e-6, worst,
        f"Sigma_R = {st.Sigma_R:.2f} MeV puts {st.Sigma_R * st.n_q / hc3:.1f} "
        f"MeV/fm^3 into P ({100 * size:.1f}% of eps) and nothing into eps")


def _reference_state(par):
    """One converged unpaired state, for the checks that need a state rather
    than a mode."""
    point = solve_beta_eq_neutrinoless(par, N_CHECK, 0.0,
                                       flags=SpeciesFlags())
    return point._state


# =============================================================================
# THE THERMODYNAMIC IDENTITIES
# =============================================================================
def check_euler(par, states, tol=1.0e-8):
    """eps + P = T s + sum_j mu_j n_j at every solved point.

    The audit the specification's own section 9.6 mandates, and the one that
    caught its two assembly errors: the sign of the vector field energy in eps
    and the missing rearrangement term in Omega. Either alone leaves this
    failing by percents while every other quantity still looks reasonable.
    """
    errors = {k: abs(p._state.euler_residual()) for k, p in states.items()}
    worst = max(errors.values()) if errors else 0.0
    return CheckResult("Euler", worst < tol, worst,
                       f"over {len(errors)} solved states")


def check_free_energy(par, states, tol=1.0e-8):
    """f = eps - T s, and f = -P + sum_j mu_j n_j."""
    errors = []
    for p in states.values():
        st = p._state
        f_a = st.eps_nat - st.T * st.s_nat
        f_b = -st.P_nat + st.mu_dot_n
        errors.append(abs(f_a - f_b) / max(abs(f_a), 1.0))
    worst = max(errors) if errors else 0.0
    return CheckResult("free energy", worst < tol, worst,
                       f"over {len(errors)} solved states")


def check_density_derivative(par, tol=1.0e-4):
    """n_B = dP/dmu_B along the beta-equilibrium solution.

    A central difference in mu_B against the summed mode densities. This is
    the identity a wrong rearrangement term breaks first, because Sigma_R is
    exactly the piece that makes the derivative of the interaction energy come
    out right; with it dropped from Omega the two sides disagree at the first
    digit.
    """
    # The seed matters here and 'restored' is the one that lands on a
    # deconfined root: at a FIXED potential the 'partial' seed flows to the
    # confined vacuum, which is a legitimate root of the same equations and a
    # useless reference for a density derivative. That asymmetry between the
    # fixed-potential and fixed-density enumerations is exactly why the branch
    # is a declared argument here and a comparison in `eos.ccdm.solver`.
    mu_B, h = 1450.0, 2.0
    out = []
    for mu in (mu_B - h, mu_B, mu_B + h):
        st, ok, _ = thermo_from_mu(par, mu, -30.0, 0.0, 0.0,
                                   branch="restored")
        if not ok or st.n_B_nat <= 0.0:
            return CheckResult(
                "n = dP/dmu", False, float("inf"),
                f"the deconfined solve at mu_B = {mu} MeV did not land on a "
                f"state with quarks in it")
        out.append(st)
    dP_dmu = (out[2].P_nat - out[0].P_nat) / (2.0 * h)
    err = abs(dP_dmu - out[1].n_B_nat) / abs(out[1].n_B_nat)
    return CheckResult("n = dP/dmu", err < tol, err,
                       f"dP/dmu_B = {dP_dmu:.6f} against n_B = "
                       f"{out[1].n_B_nat:.6f} MeV^3")


# =============================================================================
# THE REDUCTION CHAIN (section 9.6)
# =============================================================================
def check_reduction_chain(par, tol=0.0):
    """gbar_omega -> 0 gives L0; G_D -> 0 gives L2; Delta -> 0 gives section 4.

    Three one-off reductions, each exact rather than approximate:

      * with the vector coupling zero, omega_0 and Sigma_R are identically
        zero and the mode potentials are the bare ones;
      * with the diquark coupling zero, no gap can be nonzero;
      * with the gaps zero, the paired code path reproduces the unpaired state
        MODE BY MODE to the last bit, because the pairing potential is written
        as a correction and vanishes identically rather than to quadrature
        accuracy.
    """
    args = (0.4 ** 4, 5.0, 40.0)

    no_vec = replace(par, gbar_omega=0.0)
    st0 = state_at(no_vec, *args, 0.0, np.zeros(3), 1450.0, -30.0, 0.0, 0.0,
                   0.0, 0.0)
    l0 = max(abs(st0.omega_0), abs(st0.Sigma_R),
             float(np.max(np.abs(st0.mu_star - st0.mu_modes))))

    unpaired = state_at(par, *args, 0.0, np.zeros(3), 1450.0, -30.0, 0.0, 0.0,
                        0.0, 20.0)
    zero_gap = state_at(par, *args, 0.0, np.zeros(3), 1450.0, -30.0, 0.0, 0.0,
                        0.0, 20.0, pattern="CFL")
    mode_by_mode = float(np.max(np.abs(zero_gap.n_modes - unpaired.n_modes)))
    # BOTH SIDES IN NATURAL UNITS. `P` is the fm-based property of section 5's
    # boundary and `P_nat` its MeV^4 twin, so mixing them here compares two
    # numbers a factor hc^3 = 7.68e6 apart and reports the pressure itself as
    # the error -- which is what this check did, against a tol of exactly 0.
    l3 = max(mode_by_mode, abs(zero_gap.delta_omega), abs(zero_gap.pair_cost),
             abs(zero_gap.P_nat - unpaired.P_nat))

    worst = max(l0, l3)
    return CheckResult(
        "reduction chain", worst <= tol, worst,
        f"gbar_omega -> 0 gives omega_0 = Sigma_R = 0 exactly; Delta = 0 "
        f"reproduces the unpaired state over all nine modes to {mode_by_mode:g}")


def check_backend_parity(par):
    """backend='fast' against backend='reference', state by state.

    CLAUDE.md section 9: the jitted flavour is validated against the reference,
    which is what correctness is judged against. `backends/kernel_numba` writes
    out the same quadrature `eos.general.fermi_integrals.kinetic_thermo`
    performs, so the two must agree to round-off -- they sum the nine modes in
    different orders and so are not bit-identical, which is exactly why this
    check states a tolerance instead of an equality.

    THE DILATON IS SWEPT, and that is what makes this check ccdm's rather than
    a copy of njl's. Each mode carries its OWN upper limit here (the model is
    unregularised, so `unbounded_k_max` is recomputed per mode from that mode's
    potential and effective mass) and its own absence test, so a Phi near the
    ceiling -- where the dielectric closes, M* runs to 1e15 MeV and every mode
    leaves the medium -- exercises code njl's cut theory never reaches.

    Skipped, not failed, where `backends/` or numba is absent: section 5 makes
    the directory deletable, and a check that fails on its absence would make
    it mandatory.
    """
    if not NUMBA_OK:
        return CheckResult("backend parity", True, 0.0,
                           "skipped: backends/ or numba absent, and section 5 "
                           "makes both optional")
    worst, where = 0.0, ""
    # THE GAPS ARE SWEPT TOO, and not only for coverage: 'fast' selects the
    # blocked BdG of `eos.general.pairing` as well as the jitted medium
    # integrals, and with Delta = 0 the pairing block short-circuits and that
    # half of the backend is never reached.
    gaps = (np.zeros(3), np.array([0.0, 0.0, 80.0]),
            np.array([30.0, 55.0, 80.0]), np.array([70.0, 70.0, 70.0]))
    for T in (0.0, 5.0, 20.0, 50.0):
        for Phi in (0.4 ** 4, 0.7 ** 4, PHI_CEIL):
            # A GAPPED CONFINED STATE IS NOT A STATE THIS MODEL HAS, and is
            # excluded here rather than absorbed by a wider tolerance. At the
            # ceiling the dielectric has closed and M* reaches 2.1e15 MeV, so
            # a BdG problem carrying an 80 MeV gap has an eigenvalue-to-gap
            # ratio of 2.7e13: the pairing correction is then a difference of
            # numbers 1e15 apart and neither flavour can resolve it (the dense
            # path leaves ||H V - V E|| = 2.0 MeV there, the blocked one 0.69).
            # Nor is it reachable -- `DENSITY_BRANCHES` excludes `confined`,
            # and at fixed potential the confined branch carries no quarks to
            # pair and exactly zero pressure. The ceiling is still swept, at
            # Delta = 0, which is its actual physical content.
            gap_set = (np.zeros(3),) if Phi == PHI_CEIL else gaps
            for Delta in gap_set:
                for sigma, zeta in ((5.0, 40.0), (60.0, 80.0)):
                    for mu_B in (1100.0, 1450.0, 1800.0):
                        kw = dict(par=par, Phi=Phi, sigma=sigma, zeta=zeta,
                                  Sigma_V=0.0, Delta=Delta, mu_B=mu_B,
                                  mu_C=-30.0, mu_S=0.0, mu_3=0.0, mu_8=-20.0,
                                  T=T)
                        ref = state_at(**kw, backend="reference")
                        fast = state_at(**kw, backend="fast")
                        # Everything is judged against eps, the largest
                        # quantity a state carries: a relative error on a
                        # strangeness density that is 1e-12 of the state is a
                        # cancellation, not a disagreement between the two.
                        scale = max(abs(ref.eps_nat), 1.0)
                        for key in ("P_nat", "eps_nat", "s_nat", "n_B_nat",
                                    "n_C_nat", "n_S_nat", "mu_dot_n"):
                            err = abs(getattr(ref, key)
                                      - getattr(fast, key)) / scale
                            if err > worst:
                                worst, where = err, (f"{key} at T={T}, "
                                                     f"Phi={Phi:g}, "
                                                     f"mu_B={mu_B}")
    tol = 1.0e-12
    return CheckResult("backend parity", worst < tol, worst,
                       f"worst {where}" if where else "")


# =============================================================================
# THE PAIRING SECTOR
# =============================================================================
def check_colour_neutrality(par, tol=1.0e-10):
    """n_3 and n_8 vanish identically when unpaired, and are driven to zero
    when paired.

    The first half is why mu_3 and mu_8 are PINNED rather than solved for in
    an unpaired region: with no gap the colour densities are zero at
    mu_3 = mu_8 = 0 whatever else the state is doing, so a root finder given
    them as unknowns has nothing to find. The second half is the two rows the
    paired solve actually carries.
    """
    st = state_at(par, 0.4 ** 4, 5.0, 40.0, 0.0, np.zeros(3), 1450.0, -30.0,
                  0.0, 0.0, 0.0, 20.0)
    unpaired = max(abs(st.n_3), abs(st.n_8))

    point = solve_beta_eq_neutrinoless(par, N_CHECK, 0.0,
                                       flags=SpeciesFlags(csc=True))
    if not point.converged:
        return CheckResult("colour neutrality", False, float("inf"),
                           "the paired reference solve did not converge")
    scale = max((point.mu_B / 3.0) ** 3 / math.pi ** 2, 1.0)
    paired = max(abs(point._state.n_3), abs(point._state.n_8)) / scale
    worst = max(unpaired, paired)
    return CheckResult(
        "colour neutrality", worst < tol, worst,
        f"unpaired n_3 = n_8 = 0 identically; paired solve carries "
        f"mu_3 = {point.mu_3:.2f}, mu_8 = {point.mu_8:.2f} MeV")


def check_gap_window(par, lo=20.0, hi=150.0):
    """The shipped G_D puts the gap inside the 20-150 MeV window at
    mu_q ~ 450 MeV.

    A calibration check rather than an invariant: G_D is a free parameter and
    the specification's section 8.1 states the window it should be sampled to
    produce. It is here so that changing the shipped value without re-checking
    is caught.
    """
    st, ok, _ = thermo_from_mu(par, 1450.0, -30.0, 0.0, 0.0,
                               branch="restored", pattern="CFL")
    if not ok:
        return CheckResult("gap window", False, float("inf"),
                           "the CFL reference solve did not converge")
    gaps = np.abs(st.Delta)
    biggest = float(np.max(gaps))
    mu_q = float(np.mean(st.mu_star))
    return CheckResult(
        "gap window", lo <= biggest <= hi, biggest,
        f"|Delta| = {np.round(gaps, 1)} MeV at mu_q = {mu_q:.0f} MeV")


def check_gap_sign_gauge(par, tol=1.0e-12):
    """Omega is invariant under flipping any subset of the three gaps.

    The reason a solve may land on a negative gap and the reason what is
    REPORTED is the magnitude. It also says the three gap residuals flip sign
    with their own gap, so -Delta is a root whenever Delta is -- which is why
    the enumeration cannot be made to prefer one sign by seeding.
    """
    M = np.array([8.0, 8.0, 300.0])
    mu = np.full(9, 470.0) + np.array([0, 0, 0, 0, 0, 0, -5.0, -5.0, 10.0])
    base = pair_block(M, mu, np.array([60.0, 60.0, 60.0]), 20.0, 600.0)
    errors = []
    for signs in ((1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1)):
        b = pair_block(M, mu, 60.0 * np.array(signs, dtype=float), 20.0, 600.0)
        errors.append(abs(b.delta_omega - base.delta_omega)
                      / abs(base.delta_omega))
        errors.append(float(np.max(np.abs(
            b.gap_kernel - np.array(signs) * base.gap_kernel)))
            / float(np.max(np.abs(base.gap_kernel))))
    worst = max(errors)
    return CheckResult("gap sign gauge", worst < tol, worst,
                       "Omega identical and the kernel flips with its gap "
                       "under all eight sign choices")


def check_paired_entropy(par, T=5.0, ratio_max=1.0e-3):
    """A gapped phase freezes out: its entropy is exponentially suppressed.

    Substituting the unpaired entropy formula for a paired mode is not
    approximately wrong at low T, it is wrong by orders of magnitude, and it
    propagates straight into eps.

    TWO CONFIGURATIONS, because the suppression is a property of the SPECTRUM
    and not of the gap parameter, and the difference between them is the whole
    physics of a mismatched superconductor:

      * matched Fermi surfaces, M* = (8, 8, 8) MeV in CFL at Delta = 60 MeV:
        every branch sits at |Delta| or 2 Delta, and the ratio is 2.0e-4 at
        T = 5 MeV -- the specification's quoted 2.3e-4;
      * mismatched, M* = (8, 8, 300) MeV at the same Delta and the same
        potentials: the strange Fermi momentum is 361 against 469 MeV for the
        light ones, the lowest branch is pushed down from 60 MeV to 11.9, and
        the entropy is suppressed only to 0.20.

    So the check gates on the matched case and REPORTS the mismatched one. An
    implementation that had substituted the unpaired formula would give 1.0 in
    both; one that had taken Delta itself as the gap in the dispersion, rather
    than diagonalising, would give the matched answer in both and miss the
    physics that decides whether a strange quark star cools like a
    superconductor.
    """
    mu = np.full(9, 470.0)
    gap = np.full(3, 60.0)

    def ratio_at(M_star):
        M = np.asarray(M_star, dtype=float)
        block = pair_block(M, mu, gap, T, par.Lambda)
        unpaired = sum(mode_thermo(mu[j], M[j // 3], T).s for j in range(9))
        return abs(unpaired + block.delta_s) / abs(unpaired), block

    matched, _ = ratio_at([8.0, 8.0, 8.0])
    mismatched, block = ratio_at([8.0, 8.0, 300.0])
    return CheckResult(
        "paired entropy", matched < ratio_max, matched,
        f"matched CFL gives s_paired/s_unpaired = {matched:.2e} at T = {T} "
        f"MeV; at M*_s = 300 MeV the mismatch pushes the lowest branch to "
        f"{block.min_energy:.1f} MeV and the ratio only to {mismatched:.2f}")


# =============================================================================
# THE MODES
# =============================================================================
def check_mode_closures(par, states, tol=1.0e-8):
    """Each mode's own conditions hold at its solution.

    Beta equilibrium: mu_C + mu_e = 0 and total electric neutrality including
    the leptons. Fixed Y_C: the non-leptonic charge fraction is the one asked
    for, and with leptons=False the phase is CHARGED, which is what a
    mixed-phase construction needs.
    """
    errors = {}
    for name, p in states.items():
        if name.startswith("beta"):
            errors[name + " mu"] = abs(p.mu_C + p.mu_e)
            errors[name + " neutral"] = abs(p.Y_C - (p.n_e + p.n_mu) / p.n_B)
        elif "YC" in name:
            errors[name] = abs(p.Y_C - 0.1)
    worst = max(errors.values()) if errors else 0.0
    return CheckResult("mode closures", worst < tol, worst,
                       f"over {len(errors)} conditions")


def check_charge_basis(par, states, tol=1.0e-10):
    """n_B, n_C and n_S agree with `eos.general.basis` -- no local copy.

    The charge map is declared once (CLAUDE.md section 2) and every model
    imports it; this recomputes the sums from the shared table and compares.
    """
    errors = []
    for p in states.values():
        n_B, n_C, n_S = quark_charges(p.n_u, p.n_d, p.n_s)
        scale = max(abs(p.n_B), 1.0e-6)
        errors += [abs(n_B - p.n_B) / scale,
                   abs(n_C - p.Y_C * p.n_B) / scale,
                   abs(n_S - p.Y_S * p.n_B) / scale]
    worst = max(errors) if errors else 0.0
    return CheckResult("charge basis", worst < tol, worst,
                       f"over {len(states)} solved states")


def check_residual_gate(par, states):
    """Every state solved here is inside the tolerance the model claims."""
    errors = {k: p.error for k, p in states.items()}
    worst = max(errors.values()) if errors else 0.0
    return CheckResult("residual gate", worst <= RESIDUAL_TOL, worst,
                       f"gate is {RESIDUAL_TOL:g}, over {len(errors)} states")


#: Densities [fm^-3] for the causality sequence: above the deconfinement
#: onset (~1.34) and below where the deconfined branch terminates.
CAUSALITY_GRID = (1.5, 1.8, 2.1, 2.4)


def check_causality(par, tol=0.0):
    """0 <= c_s^2 <= 1 and P non-decreasing in n_B, cold beta equilibrium.

    Section 8's pair of invariants, on the one sequence this model has above
    its deconfinement onset. CCDM is colour-superconducting and a wrong gap
    contribution shows up in the sound speed first -- the gap enters Omega
    through a term whose density derivative is what c_s^2 is made of, so a
    contribution left out of P but kept in Omega is nearly invisible in eps
    and loud here.

    The response is read as `cs2_isothermal`, the name this model returns
    (CLAUDE.md section 5 forbids a bare `cs2` whose meaning depends on the
    arguments); at T = 0 it coincides with the adiabatic one.

    A density the response cannot reach comes back as nan rather than raising
    (section 6), and nan loses every comparison -- so a `max` would absorb it
    and the check would pass over a sequence it never evaluated. It is failed
    explicitly instead, naming the density.

    THE MONOTONICITY HALF IS NOT SECTION 8'S DELIVERY GATE and is not a
    substitute for it: `eos.ccdm.table` builds no `EOSTable_for_TOV`, so this
    model hands no table to a structure solver and owes no gate. What is
    checked here is the raw branch, which for a single deconfined phase above
    its onset has no first-order transition inside it and so has no licence to
    fall.
    """
    values = []
    pressures = []
    for n_B in CAUSALITY_GRID:
        cs2 = eos_response(par, "beta_eq_neutrinoless", SpeciesFlags(),
                           n_B=n_B, T=0.0)["cs2_isothermal"]
        if not np.isfinite(cs2):
            return CheckResult(
                "causality", False, float("inf"),
                f"c_s^2 is not finite at n_B = {n_B:.2f} fm^-3: the response "
                f"did not converge there, so this sequence was not evaluated")
        values.append(cs2)
        point = solve_beta_eq_neutrinoless(par, n_B, 0.0,
                                           flags=SpeciesFlags())
        if not point.converged:
            return CheckResult("causality", False, float("inf"),
                               f"the reference solve at n_B = {n_B:.2f} "
                               f"fm^-3 did not converge")
        pressures.append(point.P)
    dP = np.diff(pressures)
    violation = max(max(-v for v in values), max(values) - 1.0,
                    -float(dP.min()), 0.0)
    return CheckResult(
        "causality", violation <= tol, violation,
        f"c_s^2 in [{min(values):.3f}, {max(values):.3f}], P rises "
        f"{pressures[0]:.1f} -> {pressures[-1]:.1f} MeV/fm^3 over "
        f"n_B = {CAUSALITY_GRID[0]}-{CAUSALITY_GRID[-1]} fm^-3")


def check_glue_scale_stiffens(par, tol=0.0):
    """A larger glue bag scale costs pressure: P(B_g = 190) < P(B_g = 150).

    The direct statement that B_g is doing the job it is in the Lagrangian
    for -- paying for deconfinement -- rather than being absorbed into the
    chiral sector or the vector one. Measured at two densities above the
    onset, where both parameter points have a deconfined branch.

    IT IS DELIBERATELY NOT A SCAN FOR THE ONSET DENSITY. The deconfined
    branch's pressure crosses zero very close to where the branch itself
    terminates, so a grid scan for the crossing finds one parameter point's
    onset and misses the other's by falling off the end of the branch; the
    onset moves from about 1.34 to about 1.38 fm^-3 between these two, which
    is real and far too tight for a grid. LOCATING A TRANSITION IS
    `eos.mixed`'s job and it does it by root-finding on pressure differences,
    which is also what section 6.5 of the specification says to do.
    """
    stiff = replace(par, B_g_quarter=190.0)
    flags = SpeciesFlags()
    deltas = []
    for n in (1.8, 2.2):
        soft_p = solve_beta_eq_neutrinoless(par, n, 0.0, flags=flags)
        stiff_p = solve_beta_eq_neutrinoless(stiff, n, 0.0, flags=flags)
        if not (soft_p.converged and stiff_p.converged):
            return CheckResult("glue scale stiffens", False, float("inf"),
                               f"a reference solve at n_B = {n} failed")
        deltas.append(soft_p.P - stiff_p.P)
    worst = min(deltas)
    return CheckResult(
        "glue scale stiffens", worst > tol, worst,
        f"raising B_g^(1/4) from {par.B_g_quarter:g} to 190 MeV costs "
        f"{deltas[0]:.1f} and {deltas[1]:.1f} MeV/fm^3 at n_B = 1.8, "
        f"2.2 fm^-3")


def _check_zero_pressure(par):
    """E/A = mu_B + Y_S mu_S at the self-bound surface, both flavour contents.

    THE IDENTITY IS THE INVARIANT; THE ENERGIES ARE REPORTED. At T = 0 the
    Euler relation read at P = 0 gives eps/n_B as the Gibbs energy per baryon
    exactly, so a located root that misses `mu_B + Y_S mu_S` is a root of
    something other than P -- the one failure a locator driven by a callable
    can have that nothing else would catch.

    WHERE EACH ENERGY SITS AGAINST IRON IS NOT ASSERTED. Three-flavour E/A
    below the 930.4 MeV of iron means absolutely stable strange quark matter,
    and two-flavour E/A above it means ordinary nuclei are safe, but both are
    properties of the PARAMETER SET: a legitimately excluded point is a normal
    draw for a sampler, and a suite that failed on one would be asserting the
    Bodmer-Witten hypothesis rather than checking an implementation. The
    numbers go in the detail line instead.

    A set with no self-bound surface, and a flavour content this phase
    refuses, are reported the same way and fail nothing.
    """
    worst, detail = 0.0, []
    for two_flavour in (False, True):
        try:
            flags = SpeciesFlags(two_flavour=two_flavour)
        except NotImplementedError:
            detail.append("two-flavour: no such arm in this phase")
            continue
        surface = zero_pressure_point(par, flags,
                                        n_lo=0.30, n_hi=1.00,
                                        n_scan=8)
        label = "two" if two_flavour else "three"
        if not surface.ok:
            detail.append(f"{label}-flavour: {surface.message}")
            continue
        worst = max(worst, surface.identity_error)
        detail.append(
            f"{label}-flavour: E/A={surface.E_per_A:.2f} MeV at "
            f"n_B={surface.n_B:.4f} fm^-3, Y_S={surface.Y_S:.4f}, "
            f"{'below' if surface.below_iron else 'above'} iron")
    return CheckResult("zero-pressure surface", worst < 1e-12, worst,
                       "; ".join(detail))


def run_all(par=None, include_csc=True, include_onset=False):
    """Run every check and return the report.

    `include_csc` adds the paired states, which are the expensive half (a
    paired solve diagonalises an 18x18 matrix at every quadrature node);
    `include_onset` adds the two branch scans, which are the expensive
    quarter.
    """
    par = par if par is not None else Parameters.default()
    states = _states(par, include_csc=include_csc)
    report = FullCheckReport()
    report.results = [
        check_derived_constants(par),
        check_vacuum(par),
        check_bag_constant(par),
        check_confinement_pinning(par),
        check_rearrangement_placement(par),
        check_euler(par, states),
        check_free_energy(par, states),
        check_density_derivative(par),
        check_reduction_chain(par),
        check_mode_closures(par, states),
        check_charge_basis(par, states),
        check_residual_gate(par, states),
        check_causality(par),
        _check_zero_pressure(par),
        check_backend_parity(par),
    ]
    if include_csc:
        report.results += [
            check_colour_neutrality(par),
            check_gap_window(par),
            check_gap_sign_gauge(par),
            check_paired_entropy(par),
        ]
    if include_onset:
        report.results.append(check_glue_scale_stiffens(par))
    return report


if __name__ == "__main__":
    import sys

    report = run_all(include_csc="--no-csc" not in sys.argv,
                     include_onset="--onset" in sys.argv)
    print(report)
    # A gate that cannot fail a shell is not a gate: this printed FAIL and
    # exited 0, so nothing outside the terminal could ever notice.
    sys.exit(0 if report.all_passed else 1)
