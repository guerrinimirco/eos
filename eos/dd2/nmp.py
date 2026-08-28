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
pinned. It is NOT the default and it is not currently a usable closure --
see "Q_sat cannot be imposed while it is a third difference" below. The
selector is the argument only: the presence of "Q_sat" in the target dict
decides nothing, because a whole compute_nmp() dict carries Q_sat and would
otherwise route the natural round trip into the worse closure.

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
must be held. Measured at the published DD2 point (isoscalar Jacobian,
central differences, rows scaled by a physical size and columns by the
coupling), the condition number over the six choices is

    b_sigma + c_omega    128      <- pinned here
    c_sigma + c_omega    165
    b_sigma + b_omega    185
    b_sigma + c_sigma    305
    c_sigma + b_omega    323
    b_omega + c_omega    354

One coefficient from each meson beats holding either shape whole, because
what is left free should be the least collinear surviving pair, and c_sigma
against b_omega at |cos| = 0.974 is the least collinear pair in the matrix.
The same measurement over the five-row closure ranks its single pin
c_omega 259, b_omega 354, c_sigma 703, b_sigma 4191.

Q_sat cannot be imposed while it is a third difference
------------------------------------------------------
The five-row closure conditions at 259, and Q_sat is a third finite
difference of a solved quantity carrying a relative floor near 1.5e-3, so a
solve inherits 259 x 1.5e-3 = 0.39 of relative coupling error. No choice of
pin rescues it: 259 is the best of the four and the arithmetic still closes
on a nonsense answer. The default closure escapes this entirely -- P, E_sat,
m*/m and K_sat are all h-exact to the stencil optimum (K_sat moves 5.2e-04
MeV over h in [5e-5, 5e-4], against 2.48 MeV for Q_sat) -- which is why it,
and not the wider system, is what ships. Imposing Q_sat becomes legitimate
when the derivative is taken analytically rather than by stencil, and not
before.

What "converged" means here, and why the residual alone cannot say. A Powell
hybrid can give up on its first step and return its starting point bit for
bit, reporting the seed's own residual as though it were an answer, and
whether it does so at any given target is decided in that target's last bits
rather than by the SciPy version. That is a property of the solver, not of
the closure, and it survived the closure change: on a 105-cell
(K_sat 180-320) x (m*/m 0.45-0.75) grid at zero restarts, 18 targets miss and
12 of those misses are stalls.

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
seed could not find it"; on the grid above they recover 14 of the 18 misses.
Do NOT infer a feasibility boundary from a scan run at low `n_restarts`.

(The 187-cell (K_sat, Q_sat) scan this section used to quote -- 7/187 at zero
restarts against 115/187 at sixty-four -- was measured with the retired
closure that imposed the cross-constraint and Q_sat together. Those numbers
do not carry over and are not restated here; the conclusion they supported
does, and is re-measured above.)

The remaining numerical limit is the stencil
--------------------------------------------
Q_sat is a third finite difference of E/A, which is itself the output of a
nonlinear solve, and the h = 1e-4 step used here and in the forward map is
just past the truncation/roundoff optimum (~3e-4 to 1e-3): Q_sat spans 2.48
MeV over h in [5e-5, 5e-4] and diverges outright by h = 1e-6. K_sat, a second
difference, spans 5.2e-04 MeV over the same range, which is the whole reason
the default closure stops at K_sat.
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
#: Gate on the isoscalar residual.
#:
#: 2e-2 was set to clear two scales, and RETIRING THE CROSS ROW REMOVED BOTH
#: of the reasons it had to be this wide: the published table's own 2.2e-3
#: violation of that constraint, and the third-difference noise behind Q_sat
#: in a closure that no longer imposes it by default. Measured on the
#: (K_sat, m*/m) grid in the module docstring with restarts on, the 101 cells
#: that pass split 95 below 1e-5 and 6 in [1e-3, 2e-2] with NOTHING in
#: between, so the six sitting under this gate are certified without being
#: roots. A tighter gate is therefore now both possible and wanted -- it was
#: not, while the cross row made accurate solves land at 1.9e-3 -- but
#: choosing the value moves `ok` for real targets and belongs to the ticket
#: that measures it on more than two axes, not here.
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
                q = _isoscalar_quantities(par, n_sat, want_Q=False)
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

    `nmp` is a dict with {n_sat, E_sat, m_eff_ratio, K_sat, E_sym, L_sym}.
    The inversion always uses the default closure -- four rows over four
    couplings with b_sigma and c_omega pinned -- and reports Q_sat and K_sym
    as predictions in the status. A "Q_sat" key is ignored here: imposing it
    is `invert_nmp(..., impose_Q_sat=True)` and is not a closure to reach by
    accident. Returns the `Parameters`, or (Parameters, InversionStatus) when
    `return_status`.

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
