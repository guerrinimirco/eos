"""The map between DD2's couplings and the nuclear-matter parameters they
produce, in both directions.

`compute_nmp` extracts {n_sat, E_sat, m*/m, K_sat, Q_sat, K_sym, E_sym, L_sym}
from a `Parameters`; `invert_nmp` and `from_nmp` recover the couplings from a
subset of them, and `build_parametrization` composes that inverse with the
hyperon and Delta sector constructors, so one sample dict of nuclear-matter
parameters and single-particle potentials becomes one `Parameters`. The two share this module because they share the stencils: the
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
the omega shape. Pinning c_omega anchors it, while pinning a sigma-side
coefficient lets the cross-constraint drag the omega shape and moves the
predicted Q_sat by 20-30 MeV.

Imposing Q_sat instead of the pin remains available (impose_Q_sat=True): the
isoscalar system is then the 6x6 {P, E_sat, m*/m, K_sat, Q_sat, cross} over
all six couplings. When the caller does not say, the presence of "Q_sat" in
the target dict decides — so existing callers that always supplied it keep
the closure they were written against.

The published couplings are NOT a root of the 5x5 closure
------------------------------------------------------
The isoscalar cross-constraint is DD2's own, and the published table obeys it
to 2.200718e-3, not exactly. Four of the five 5x5 rows vanish at the published
couplings and the cross row does not, so the published point is a stationary
point of the residual norm rather than a zero of it, and driving the cross row
to zero costs a move in the sigma shape: b_sigma -0.025, c_sigma -0.027. Those
are the coefficients Q_sat rides on, so the converged 5x5 solution given DD2's
own NMPs predicts Q_sat = 117.5, not the 169.0 the forward map returns for the
published table.

That is the closure answering honestly, not a solver defect, and it is the
price of closing the isoscalar sector with a constraint the fitted table only
approximately obeys. Do NOT read a 5x5 round trip as a test that recovers
published couplings: no seed recovers them, because they are not a root. The
6x6 path (impose_Q_sat=True) does reproduce the imposed NMPs, its shape
coefficients to ~2e-3.

What "converged" means here, and why the residual alone cannot say. A Powell
hybrid can give up on its first step and return the starting point bit for
bit, reporting the seed's own 2.2e-3 cross-row violation as its residual --
and whether it does so at any given target is decided in that target's last
bits, not by the SciPy version. ISO_GATE (2e-2) admits that residual, and it
cannot be tightened to reject it: a moved and ACCURATE solve was measured at
1.944e-3 (K_sat recovered to 0.01 MeV), so stalled and converged residuals
overlap and no threshold on the residual alone separates them.

What separates them is that the stall has not moved. `InversionStatus`
therefore carries `coupling_shift`, the max relative distance from the seed,
and a solve that returns the seed unmoved on a residual above STALL_RES is
reported as ok=False rather than certified. The same condition drives the
restart loop, which is the substantive half: the stall used to keep the
restarts from ever running, since its residual sat under the gate. It is not
an unreachable target -- at DD2's own nuclear-matter parameters the FIRST
jittered restart drives the 5x5 to 6.8e-08 and recovers K_sat to 1e-4 MeV.

`coupling_shift` also answers a second question the residual never could:
"converged" and "recovered the published couplings" are different statements.
The converged 5x5 branch at DD2's own NMPs sits 3.9% from the published set
and reproduces the same six NMPs -- see the section below on why the published
couplings are not a root of this closure.

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
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq, root

from eos.general.physics_constants import hc3
from eos.dd2.couplings import (
    rational_d2f, derived_a, derived_d,
    SU6_HYPERON, DD2Y_HYPERON, _POTENTIAL_KEY,
    scalar_ratio_from_potential,
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
#: Gate on the isoscalar residual. Two scales meet here, both below it: the
#: finite-difference third derivative behind Q_sat in the 6x6 path (~0.1 MeV,
#: scaled by 1e-2 in the residual), and the published table's own 2.2e-3
#: violation of the cross-constraint in the 5x5 path. A tighter gate would
#: reject converged solutions for reasons that have nothing to do with
#: whether the NMPs are representable. Note the cost of the second scale: at
#: 2e-2 the gate also admits a solve that stalled at the published seed
#: without reducing the cross row, so ok=True is a statement about the
#: residual and not a promise that the solver moved (module docstring).
ISO_GATE = 2e-2

#: Perturbed restarts attempted when the first isoscalar solve misses the
#: gate. They run ONLY on a miss, so an NMP set that inverts from the DD2 seed
#: costs exactly what it did before. What they buy is large and does not
#: saturate — see the module docstring — so this default is a compromise with
#: scan cost (a miss costs ~n_restarts x 40 ms), not a converged answer.
#: Raise it when mapping a boundary matters more than the wall clock.
N_RESTARTS = 32

#: Residual above which an UNMOVED seed is a stall rather than an answer.
#: `root(method="hybr")` can return its starting point bit for bit, reporting
#: the seed's own residual, and ISO_GATE is too coarse to notice: at DD2's own
#: nuclear-matter parameters the 5x5 stalls at the published couplings' 2.201e-3
#: cross-row violation, which sits UNDER the 2e-2 gate. The gate cannot be
#: tightened to catch it either -- a moved and accurate solve was measured at
#: 1.944e-3 (K_sat recovered to 0.01 MeV), so stalled and converged residuals
#: overlap and no threshold on the residual alone separates them. What does
#: separate them is that the stall has not moved at all. An unmoved seed is
#: legitimate only when the seed was already the root, which lands at <= 2e-8
#: (measured by re-seeding a solve at its own answer); five orders separate
#: that from the stall and this floor sits in the middle. Measured on
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
    #: How far the isoscalar solve left its seed, max relative over the free
    #: couplings. Exactly 0.0 means the solver never moved, which `ok` reads
    #: as a failure unless the seed was already the root (STALL_RES). Reported
    #: because "converged" and "recovered the published couplings" are
    #: different statements: at DD2's own NMPs the converged 5x5 branch sits
    #: 3.9% from the published set and reproduces the same six NMPs.
    coupling_shift: float = float("nan")


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
    shift = _relative_shift(best_x, seed)

    if _stalled(best_x, seed, iso_res):
        # The solver never left the seed, and the seed is not a root: this is
        # the published couplings handed straight back with their own residual.
        # ISO_GATE admits it (2.201e-3 < 2e-2) and cannot be tightened to
        # reject it without also rejecting moved, accurate solves at 1.944e-3,
        # so the verdict is made here instead. Section 6: a non-convergence is
        # a reported return value, never a silent wrong answer.
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
        predictions=predictions, coupling_shift=shift)
    return par, status


def from_nmp(nmp, m_sigma=546.212459, return_status=False):
    """Nuclear-matter parameters -> a `Parameters` carrying those couplings.

    `nmp` is a dict with {n_sat, E_sat, m_eff_ratio, K_sat, E_sym, L_sym}, and
    optionally Q_sat. Without Q_sat the inversion uses the structural closure
    (the cross-constraint plus the pinned shape coefficient) and reports Q_sat
    and K_sym as predictions in the status; with Q_sat present it is imposed
    instead of the pin. Returns the `Parameters`, or (Parameters,
    InversionStatus) when `return_status`.

    **`Parameters` is None when the inversion did not converge**, since a soft
    failure is a return value and not an exception (CLAUDE.md section 6). Pass
    `return_status=True` and test `status.ok` before using the result: without
    it a failed inversion is indistinguishable from a successful one until the
    None reaches a solver. `build_parametrization` below does that test for
    the caller and reports the stage instead.

    The hyperon and Delta sectors attach on top of the result through
    `from_hyperon_potentials` / `from_delta_potential` below, once the
    nucleon sector is set; they are not folded in here.
    """
    par, status = invert_nmp(nmp, m_sigma=m_sigma)
    return (par, status) if return_status else par


# ==========================================================================
# THE HYPERON AND DELTA SECTORS FROM THEIR SINGLE-PARTICLE POTENTIALS
# ==========================================================================
# Free functions rather than classmethods on `Parameters`, and here rather
# than in `parameters.py`, for the reason stated at the top of this module:
# both invert a potential by re-solving symmetric nuclear matter at
# saturation, so both sit ABOVE `solver.py` in the layer order, while
# `parameters.py` is its bottom (CLAUDE.md section 5).

def from_hyperon_potentials(U_Lambda=-30.0, U_Sigma=30.0, U_Xi=-18.0,
                            base=None, x_phi=None):
    """
    Nucleon + hyperon octet with SU(6) vector couplings and scalar couplings
    *inverted* from the hyperon potentials U_Y in SNM at saturation (report
    §2.4b). This is the mechanism that regenerates the DD2Y R_sigma table
    (U_Xi = -18) and the route for non-DD2Y potentials. Hyperon masses
    default to the DD2Y (Marques) values.

    base: an existing Parameters to attach the hyperon sector to (e.g.
    an NMP-inverted nucleon par, so NMP + hyperons compose); defaults to
    nucleonic DD2. The scalar inversion re-solves SNM on ``base``, so it
    adapts to that par's nucleon couplings automatically.

    x_phi: override the SU(6) hidden-strange column x_phiY = g_phiY/g_omegaN.
    None keeps the SU(6) value per hyperon; a float replaces it in every row.
    `x_phi = 0.0` is how a hyperonic set is built with no phi sector at all --
    the coupling carries that statement, there is no flag for it.
    """
    base = replace(base if base is not None else Parameters.default(),
                   U_Lambda=U_Lambda, U_Sigma=U_Sigma, U_Xi=U_Xi)
    sat = solve_snm(base, base.n_sat)
    Gs_sat, Gw_sat, _, _, _, _ = base.couplings_at(base.n_sat)
    U_map = {"U_Lambda": U_Lambda, "U_Sigma": U_Sigma, "U_Xi": U_Xi}

    rows = []
    for name, su6 in SU6_HYPERON.items():
        x_sigma = scalar_ratio_from_potential(
            U_map[_POTENTIAL_KEY[name]], su6["x_omega"], Gs_sat, Gw_sat,
            sat.matter.fields["sigma"], sat.matter.fields["omega0"],
            sat.matter.Sigma_R)
        rows.append((name, DD2Y_HYPERON[name]["mass"], x_sigma,
                     su6["x_omega"], su6["x_rho"],
                     su6["phi_over_omegaN"] if x_phi is None else float(x_phi)))
    return replace(base, hyperon_couplings=tuple(rows))


def from_delta_potential(U_Delta=-50.0, x_wD=1.0, x_rD=1.0, base=None):
    """
    Δ-isobar couplings from the Δ single-particle potential in SNM at
    saturation (report v11 §2.4). There is no canonical DD2Δ coupling
    table, so the default is universal coupling (x_Δσ = x_Δω = x_Δρ = 1);
    this constructor instead fixes x_Δσ by inverting

        U_Δ = -x_Δσ Γ_σN σ̄ + x_Δω Γ_ωN ω0 + Σ^R      (all at n_sat)

    for a chosen Δ potential (literature U_Δ ∈ [-100, -50] MeV, default -50)
    and vector ratios x_wD, x_rD. base: an existing Parameters to attach
    the Δ sector to (e.g. a DD2Y octet); defaults to nucleonic DD2.
    """
    if not (-100.0 <= U_Delta <= -50.0):
        raise ValueError(
            f"U_Delta = {U_Delta} MeV outside the literature range "
            f"[-100, -50]; pass an explicit value in range or widen it")
    base = base or Parameters.default()
    sat = solve_snm(base, base.n_sat)
    Gs_sat, Gw_sat, _, _, _, _ = base.couplings_at(base.n_sat)
    x_Delta_sigma = scalar_ratio_from_potential(
        U_Delta, x_wD, Gs_sat, Gw_sat, sat.matter.fields["sigma"],
        sat.matter.fields["omega0"], sat.matter.Sigma_R)
    return replace(base, x_Delta_sigma=x_Delta_sigma,
                   x_Delta_omega=x_wD, x_Delta_rho=x_rD)


# ==========================================================================
# NMPs + SECTOR POTENTIALS -> ONE PARAMETRIZATION
# ==========================================================================

#: Hadronic-sector coupling knobs that may be carried *inside* an NMP sample
#: dict, alongside the nuclear-matter parameters themselves, so that one
#: sample describes the whole hadronic parametrization. `x_wD`/`x_rD` are the
#: Delta vector coupling ratios x_omegaDelta / x_rhoDelta; x_sigmaDelta is not
#: free, being fixed by inverting U_Delta.
SECTOR_KEYS = ("U_Lambda", "U_Sigma", "U_Xi", "U_Delta", "x_wD", "x_rD")


def _split_sample(sample, hyperon_potentials=None, U_Delta=-50.0):
    """Separate a sample dict into (nmp, sector kwargs).

    A sample may carry any of the `SECTOR_KEYS` next to the nuclear-matter
    parameters; those override the corresponding keyword arguments. This is
    what lets one dict put L_sym, U_Xi and U_Delta on axes together -- they
    are all "hadronic parameters" to the caller even though the inversion
    treats them in separate stages. Keys absent from both the sample and the
    keyword arguments are left out, so the sector constructors below apply
    their own published defaults.
    """
    nmp = {k: v for k, v in sample.items() if k not in SECTOR_KEYS}
    pots = dict(hyperon_potentials or {})
    pots.update({k: float(sample[k]) for k in ("U_Lambda", "U_Sigma", "U_Xi")
                 if k in sample})
    sector = {"hyperon_potentials": pots,
              "U_Delta": float(sample.get("U_Delta", U_Delta)),
              "x_wD": float(sample.get("x_wD", 1.0)),
              "x_rD": float(sample.get("x_rD", 1.0))}
    return nmp, sector


def build_parametrization(nmp, flags, hyperon_potentials=None,
                          U_Delta=-50.0):
    """Nuclear-matter parameters to a `Parameters` with the strange and
    resonant sectors attached, as `flags` requires.

    `from_nmp` inverts the NUCLEON sector only -- it carries no hyperon
    couplings, so `SpeciesFlags(hyperons=True)` on its output would fail deep
    in a coupling lookup. The hyperon and Delta sectors are attached on top
    here, each by inverting its single-particle potential in symmetric matter
    at saturation *on the inverted base*, so they adapt to that base's nucleon
    couplings rather than assuming DD2's.

    `nmp` may also carry any of the `SECTOR_KEYS` (U_Lambda, U_Sigma, U_Xi,
    U_Delta, x_wD, x_rD); those take precedence over the keyword arguments, so
    a single dict can put nuclear-matter parameters and sector potentials on
    axes together.

    Returns `(par, stage, message)`. `stage` is 'ok', 'inversion_failed' when
    the NMPs have no DD-RMF realisation at all, or 'sectors_failed' when they
    do but the hyperon/Delta scalar inversion does not converge on them -- the
    second can happen even when the first succeeded, which is why they are
    reported separately. `par` is None unless `stage` is 'ok'.
    """
    nmp, sector = _split_sample(dict(nmp), hyperon_potentials, U_Delta)
    par, status = from_nmp(nmp, return_status=True)
    if not status.ok:
        return None, "inversion_failed", status.message
    try:
        if flags.hyperons:
            par = from_hyperon_potentials(
                base=par, **sector["hyperon_potentials"])
        if flags.deltas:
            par = from_delta_potential(
                U_Delta=sector["U_Delta"], x_wD=sector["x_wD"],
                x_rD=sector["x_rD"], base=par)
    except Exception as exc:
        return None, "sectors_failed", f"{type(exc).__name__}: {exc}"
    return par, "ok", ""
