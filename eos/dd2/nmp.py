"""The map between DD2's couplings and the nuclear-matter parameters they
produce, in both directions.

`compute_nmp` extracts {n_sat, E_sat, m*/m, K_sat, Q_sat, K_sym, E_sym, L_sym}
from a `Parameters`; `invert_nmp` and `from_nmp` recover the couplings from a
subset of them. The two share this module because they share the stencils: the
finite-difference bias cancels on a round trip only while both sides difference
the same way, so a change to `h` made in one and not the other stops the
inversion reproducing its own inputs.

Both directions sit ABOVE `solver.py` in the layer order, because every
quantity here is a property of solved symmetric nuclear matter at saturation.
That is why `from_nmp` lives here as a free function rather than as a
classmethod on `Parameters`, which is the bottom layer.

THE INVERSE MAP
---------------

The forward map (nmp.compute_nmp) extracts {n_sat, E_sat, m*/m, K_sat, Q_sat,
K_sym, E_sym, L_sym} from a Parameters. This inverts it.

The imposed set is {n_sat, E_sat, m*/m, K_sat, E_sym, L_sym}:

  1. Isoscalar (5x5 root at FIXED n_sat, so no P=0 bracket search in the
     loop): free {Gamma_sigma, b_sigma, c_sigma, Gamma_omega, b_omega}
     matched to {P(n_sat)=0, E_sat, m*/m, K_sat, and the cross-constraint
     f_sigma''(1) = f_omega''(1)}. The sixth coupling, c_omega, is PINNED at
     its published value — that pin plus the cross-constraint are the model's
     structural closure of the isoscalar sector. m_sigma is fixed; a_i, d_i
     are derived internally (from_microscopic).
  2. Isovector (near-analytic): Gamma_rho(n_sat) from E_sym in closed form,
     then a_rho from L_sym by a 1-D root.
  3. The higher derivatives NOT imposed — Q_sat and K_sym — are computed
     forward from the recovered couplings and reported in
     InversionStatus.predictions. They are predictions of the closure, not
     inputs.

Why c_omega is the pinned coefficient: Q_sat is carried almost entirely by
the omega shape. Pinning c_omega anchors it — on DD2's own NMPs the round
trip returns the published couplings unchanged and predicts Q_sat within the
0.1 MeV stencil noise — while pinning a sigma-side coefficient lets the
cross-constraint drag the omega shape and moves the predicted Q_sat by
20-30 MeV.

Imposing Q_sat instead of the pin remains available (impose_Q_sat=True): the
isoscalar system is then the 6x6 {P, E_sat, m*/m, K_sat, Q_sat, cross} over
all six couplings. When the caller does not say, the presence of "Q_sat" in
the target dict decides — so existing callers that always supplied it keep
the closure they were written against.

The isoscalar cross-constraint is DD2's own, and the published table obeys it
to 2.2e-3, not exactly. The 5x5 residual therefore has a floor of that order
at DD2-like targets (the physics rows hold the couplings at the published
values and the cross row reports the table's own violation); ISO_GATE covers
it. A round trip reproduces the imposed NMPs exactly; shape coefficients to
~2e-3 in the 6x6 path.

What limits which NMPs invert: the seed, not the physics
--------------------------------------------------------
A single solve from the published DD2 couplings converges only for targets
near DD2's own values, and the set it reaches traces a narrow band through
that seed point. That band is a picture of one basin of attraction, NOT of
the feasible set, and reading it as physics is the mistake this module exists
to prevent. Measured on a 187-cell (K_sat, Q_sat) grid over K_sat = 200-300
and Q_sat = 0-400 MeV with the 6x6 closure:

    restarts     0      32      64
    inverting    7/187  68/187  115/187

The single seed reports 4% of the plane as representable; sixty-four restarts
find 61% of it, still without saturating, and the remaining failures are
scattered rather than bounding a region — so they are very likely further
seed failures too. The residual surface has a spurious basin in which the
cross-constraint is satisfied but Q_sat is wrong, and which basin a solve
lands in is a property of where it started. Hence N_RESTARTS. Do NOT infer a
(K_sat, Q_sat) constraint curve from a scan run at low `n_restarts`.

The remaining numerical limit is the stencil
--------------------------------------------
Q_sat is a third finite difference of E/A, which is itself the output of a
nonlinear solve, and the h = 1e-4 step used here and in the forward map is
just past the truncation/roundoff optimum (~3e-4 to 1e-3): Q_sat carries
~0.1 MeV of stencil noise at h = 1e-4 and diverges outright by h = 1e-6.
The forward map (nmp.compute_nmp) uses the IDENTICAL stencil on purpose, so
the finite-difference bias cancels exactly on a round trip; any change to h
must be made in both places together or the round trip stops reproducing its
own inputs. The same applies to the predicted Q_sat this module reports.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, root

from eos.general.physics_constants import hc3
from eos.dd2.couplings import rational_d2f, derived_a, derived_d
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


def compute_nmp(par, h=1e-4, n_lo=0.12, n_hi=0.18):
    """
    Nuclear-matter parameters at saturation.

    Returns dict with n_sat [fm^-3], E_sat, K_sat, Q_sat, E_sym, L_sym,
    K_sym [MeV], m_eff_ratio, and P_sat [MeV/fm^3] (diagnostic, ~0 by
    construction). K_sym = 9 n^2 E_sym''(n) is reported because the NMP
    inversion treats it, like Q_sat, as a prediction of the closure rather
    than an input.
    """
    n_sat = brentq(lambda n: solve_snm_t0(par, n).P, n_lo, n_hi, xtol=1e-12)
    at_sat = solve_snm_t0(par, n_sat)

    EA = lambda n: energy_per_baryon(par, n)
    d2 = (EA(n_sat + h) - 2.0 * EA(n_sat) + EA(n_sat - h)) / h ** 2
    d3 = (EA(n_sat + 2 * h) - 2.0 * EA(n_sat + h)
          + 2.0 * EA(n_sat - h) - EA(n_sat - 2 * h)) / (2.0 * h ** 3)
    dEs = (esym(par, n_sat + h) - esym(par, n_sat - h)) / (2.0 * h)
    d2Es = (esym(par, n_sat + h) - 2.0 * esym(par, n_sat)
            + esym(par, n_sat - h)) / h ** 2

    return {
        "n_sat": n_sat,
        "E_sat": EA(n_sat),
        "m_eff_ratio": _dirac_mass(at_sat) / par.m_nucleon,
        "K_sat": 9.0 * n_sat ** 2 * d2,
        "Q_sat": 27.0 * n_sat ** 3 * d3,
        "E_sym": esym(par, n_sat),
        "L_sym": 3.0 * n_sat * dEs,
        "K_sym": 9.0 * n_sat ** 2 * d2Es,
        "P_sat": at_sat.P,
    }


# =============================================================================
# INVERSE:  nuclear-matter parameters -> couplings
# =============================================================================
#: Gate on the isoscalar residual. Two floors meet here, both below it: the
#: finite-difference third derivative behind Q_sat in the 6x6 path (~0.1 MeV,
#: scaled by 1e-2 in the residual), and the published table's own 2.2e-3
#: violation of the cross-constraint in the 5x5 path. A tighter gate would
#: reject converged solutions for reasons that have nothing to do with
#: whether the NMPs are representable.
ISO_GATE = 2e-2

#: Perturbed restarts attempted when the first isoscalar solve misses the
#: gate. They run ONLY on a miss, so an NMP set that inverts from the DD2 seed
#: costs exactly what it did before. What they buy is large and does not
#: saturate — see the module docstring — so this default is a compromise with
#: scan cost (a miss costs ~n_restarts x 40 ms), not a converged answer.
#: Raise it when mapping a boundary matters more than the wall clock.
N_RESTARTS = 32

#: The pinned isoscalar shape coefficient of the default closure, at its
#: published DD2 value (Typel et al. 2010). See the module docstring for why
#: it is c_omega and not a sigma-side coefficient.
PINNED_COEFF = "c_omega"


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


def _f2_at1(b, c):
    """f_i''(1) with a_i, d_i derived from (b, c)."""
    d = float(derived_d(c))
    a = float(derived_a(b, c, d))
    return rational_d2f(1.0, a, b, c, d)


def _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma, Grho=3.0, a_rho=0.5):
    """Build a Parameters from free isoscalar params (a,d derived)."""
    return Parameters.from_microscopic(
        n_sat=n_sat, gamma_sigma=Gs, b_sigma=bS, c_sigma=cS,
        gamma_omega=Gw, b_omega=bW, c_omega=cW,
        gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)


def _isoscalar_quantities(par, n_sat, h=1e-4, want_Q=True):
    """{P, E/A, m*/m, K_sat, (Q_sat)} of SNM at n_sat (no P=0 search).

    want_Q=False skips the third-difference stencil and its four extra
    solves — the default closure does not impose Q_sat, so its residual
    never needs it.
    """
    EA = lambda n: solve_snm(par, n).eps / n - par.m_nucleon
    at = solve_snm(par, n_sat)
    d2 = (EA(n_sat + h) - 2 * EA(n_sat) + EA(n_sat - h)) / h ** 2
    out = dict(P=at.P, E_sat=EA(n_sat), m_ratio=_dirac_mass(at) / par.m_nucleon,
               K_sat=9 * n_sat ** 2 * d2)
    if want_Q:
        d3 = (EA(n_sat + 2 * h) - 2 * EA(n_sat + h)
              + 2 * EA(n_sat - h) - EA(n_sat - 2 * h)) / (2 * h ** 3)
        out["Q_sat"] = 27 * n_sat ** 3 * d3
    return out


def _restart_loop(iso_residual, seed, first, n_restarts, gate=ISO_GATE):
    """Keep the best of the first solve and up to n_restarts jittered ones.

    Deterministic by construction: the same NMP must invert identically on
    every run and in every parallel worker, so the generator is seeded with a
    constant rather than left to entropy.
    """
    best_x = first.x
    best_res = float(np.max(np.abs(iso_residual(best_x))))
    if best_res >= gate and n_restarts:
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
            if res < best_res:
                best_x, best_res = trial.x, res
            if best_res < gate:
                break
    return best_x, best_res


def invert_nmp(nmp, m_sigma=546.212459, seed=None, n_restarts=N_RESTARTS,
               impose_Q_sat=None):
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
      False — the default convention: Q_sat is a PREDICTION. 5x5 over
              {Gamma_sigma, b_sigma, c_sigma, Gamma_omega, b_omega} with
              c_omega pinned at its published value; conditions
              {P(n_sat)=0, E_sat, m*/m, K_sat, cross-constraint}.
      True  — Q_sat is imposed: 6x6 over all six couplings, the pin replaced
              by the Q_sat condition.
      None  — decided by the dict: True iff "Q_sat" is present, so a caller
              that always supplied Q_sat keeps the closure it was written
              against.

    Either way the recovered couplings' Q_sat and K_sym are computed forward
    (same stencils as nmp.compute_nmp) and reported in status.predictions.

    `n_restarts` perturbed seeds are tried when the first isoscalar solve
    misses ISO_GATE. This is not a refinement: it is what separates "these
    NMPs have no DD-RMF realisation" from "this seed could not find it" —
    see the module docstring. Set it to 0 for single-seed behaviour.
    """
    if impose_Q_sat is None:
        impose_Q_sat = "Q_sat" in nmp
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
    pinned_value = getattr(ref, PINNED_COEFF)
    if seed is None:
        # DD2-class NMPs sit near the published couplings, and the residual
        # surface has a spurious basin (cross-constraint satisfied but the
        # omega shape wrong) that a generic seed can fall into.
        seed = [ref.gamma_sigma, ref.b_sigma, ref.c_sigma,
                ref.gamma_omega, ref.b_omega]
        if impose_Q_sat:
            seed = seed + [ref.c_omega]

    if impose_Q_sat:
        tgt = np.array([0.0, nmp["E_sat"], nmp["m_eff_ratio"],
                        nmp["K_sat"], nmp["Q_sat"]])

        def iso_residual(p):
            Gs, bS, cS, Gw, bW, cW = p
            if cS <= 0 or cW <= 0 or Gs <= 0 or Gw <= 0:
                return [1e3] * 6
            try:
                par = _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma)
                q = _isoscalar_quantities(par, n_sat)
            except (ValueError, RuntimeError):
                return [1e3] * 6
            cross = _f2_at1(bS, cS) - _f2_at1(bW, cW)
            return [q["P"] - tgt[0], q["E_sat"] - tgt[1],
                    q["m_ratio"] - tgt[2], (q["K_sat"] - tgt[3]) * 1e-2,
                    (q["Q_sat"] - tgt[4]) * 1e-2, cross]

        def couplings_of(x):
            return tuple(x)                     # (Gs, bS, cS, Gw, bW, cW)
    else:
        tgt = np.array([0.0, nmp["E_sat"], nmp["m_eff_ratio"], nmp["K_sat"]])

        def iso_residual(p):
            Gs, bS, cS, Gw, bW = p
            cW = pinned_value
            if cS <= 0 or Gs <= 0 or Gw <= 0:
                return [1e3] * 5
            try:
                par = _trial_par(n_sat, Gs, bS, cS, Gw, bW, cW, m_sigma)
                q = _isoscalar_quantities(par, n_sat, want_Q=False)
            except (ValueError, RuntimeError):
                return [1e3] * 5
            cross = _f2_at1(bS, cS) - _f2_at1(bW, cW)
            return [q["P"] - tgt[0], q["E_sat"] - tgt[1],
                    q["m_ratio"] - tgt[2], (q["K_sat"] - tgt[3]) * 1e-2,
                    cross]

        def couplings_of(x):
            Gs, bS, cS, Gw, bW = x
            return Gs, bS, cS, Gw, bW, pinned_value

    first = root(iso_residual, seed, method="hybr", tol=1e-12)
    best_x, iso_res = _restart_loop(iso_residual, seed, first, n_restarts)
    Gs, bS, cS, Gw, bW, cW = couplings_of(best_x)

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
            isoscalar_residual=iso_res, isovector_residual=float("nan"))

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
        from eos.dd2.nmp import esym
        dEs = (esym(p, n_sat + 1e-4) - esym(p, n_sat - 1e-4)) / 2e-4
        return 3.0 * n_sat * dEs

    a_rho = brentq(lambda a: Lsym_of_arho(a) - nmp["L_sym"], -2.0, 5.0,
                   xtol=1e-10)
    isov_res = abs(Lsym_of_arho(a_rho) - nmp["L_sym"])

    par = Parameters.from_microscopic(
        n_sat=n_sat, gamma_sigma=Gs, b_sigma=bS, c_sigma=cS,
        gamma_omega=Gw, b_omega=bW, c_omega=cW,
        gamma_rho=Grho, a_rho=a_rho, m_sigma=m_sigma)

    # --- report what the closure predicts, with the forward map's stencils --
    from eos.dd2.nmp import esym
    q_final = _isoscalar_quantities(par, n_sat, want_Q=True)
    h = 1e-4
    d2Es = (esym(par, n_sat + h) - 2 * esym(par, n_sat)
            + esym(par, n_sat - h)) / h ** 2
    predictions = {"Q_sat": q_final["Q_sat"],
                   "K_sym": 9.0 * n_sat ** 2 * d2Es}

    status = InversionStatus(
        ok=(isov_res < 1e-3),                # isoscalar gate already passed
        message="converged" if isov_res < 1e-3 else
        f"isovector residual {isov_res:.2e} above 1e-3",
        isoscalar_residual=iso_res, isovector_residual=float(isov_res),
        predictions=predictions)
    return par, status


def from_nmp(nmp, m_sigma=546.212459, return_status=False):
    """Nuclear-matter parameters -> a `Parameters` carrying those couplings.

    `nmp` is a dict with {n_sat, E_sat, m_eff_ratio, K_sat, E_sym, L_sym}, and
    optionally Q_sat. Without Q_sat the inversion uses the structural closure
    (the cross-constraint plus the pinned shape coefficient) and reports Q_sat
    and K_sym as predictions in the status; with Q_sat present it is imposed
    instead of the pin. Returns the `Parameters`, or (Parameters,
    InversionStatus) when `return_status`.

    The hyperon and Delta sectors attach on top of the result through
    `Parameters.from_hyperon_potentials` / `from_delta_potential`, once the
    nucleon sector is set; they are not folded in here.
    """
    par, status = invert_nmp(nmp, m_sigma=m_sigma)
    return (par, status) if return_status else par
