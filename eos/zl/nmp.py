"""
nmp.py
====================
The nuclear-matter parameters of a ZL parameter set, both directions.

ZL is a nucleonic energy-density functional whose six parameters exist to set
these numbers, so the map from couplings to (n_sat, E_sat, K_sat, E_sym, L_sym)
is the natural statement of what a parameter set *is*. Everything here is a
PREDICTION of the couplings: ZL imposes no saturation condition internally, so
even n_sat is found rather than declared.

The inverse map is CLOSED FORM. ZL's interaction contributes a0 u + b0 u^gamma
to the energy per baryon of symmetric matter and (a1 - a0) u + b1 u^gamma1
- b0 u^gamma to the symmetry energy, so imposing {n_sat, E_sat, K_sat} and
{E_sym, L_sym} inverts by algebra rather than by a root find -- no seed, no
basin, no restart count. Six couplings against five nuclear-matter parameters
still leaves one free choice, and `invert_nmp` makes the caller name it: either
gamma1, or a sixth datum K_sym, which ZL alone among the models here can impose
because it alone has three isovector knobs. Q_sat is NOT available as that
datum -- in ZL it is rigidly 3 (gamma - 2) K_sat in the interaction part, so
{n_sat, E_sat, K_sat} already determine it.

Nucleons only, no leptons: nuclear matter is the isolated strong sector, and
every quantity below is a property of that sector alone. `thermo_from_n` is
already leptonless, so no flag is needed to say so.
"""
from dataclasses import dataclass, field as dataclass_field, replace

from scipy.optimize import brentq

from eos.zl.parameters import Parameters
from eos.zl.thermodynamics import thermo_from_n

#: Bracket for the saturation root, in fm^-3. Wide enough for any parameter
#: set a sampler is likely to propose, narrow enough that the bracket does not
#: reach the spinodal region where P = 0 has a second root.
N_SAT_BRACKET = (0.10, 0.25)


def nuclear_matter(par, n_B, beta=0.0, T=0.0):
    """Nuclear matter at density n_B and asymmetry beta, leptonless.

    beta = 0 is symmetric matter and beta = -1 pure neutron matter; the charge
    fraction that means is Y_C = (1 - beta)/2, since for nucleons
    beta = (n_n - n_p)/n_B = 1 - 2 Y_C.
    """
    Y_C = 0.5 * (1.0 - beta)
    return thermo_from_n((1.0 - Y_C) * n_B, Y_C * n_B, T, par)


def energy_per_baryon(par, n_B, beta=0.0, T=0.0):
    """E/A = eps/n_B - m [MeV], the binding energy per baryon.

    The rest mass subtracted follows the COMPOSITION, m_p n_p + m_n n_n, not an
    average over one nucleon mass: with the physical m_n - m_p = 1.29 MeV an
    average leaves a term linear in beta that swamps the quadratic symmetry
    term at small asymmetry. In symmetric matter the two coincide.
    """
    st = nuclear_matter(par, n_B, beta, T)
    rest = par.m_p * st.n_p + par.m_n * st.n_n
    return (st.e - rest) / st.n_B


def pressure(par, n_B, beta=0.0, T=0.0):
    """P [MeV/fm^3] of nuclear matter at (n_B, beta)."""
    return nuclear_matter(par, n_B, beta, T).P


def saturation_density(par, T=0.0, bracket=N_SAT_BRACKET):
    """n_sat [fm^-3]: the density where symmetric matter has P = 0.

    Solved as a root of P rather than as a minimum of E/A. The two are the same
    condition -- P = n^2 d(E/A)/dn -- but P is returned exactly by the
    thermodynamics, so rooting it avoids differencing E/A and costs one
    derivative less of accuracy in everything computed at n_sat.
    """
    return brentq(lambda n: pressure(par, n, 0.0, T), *bracket, xtol=1e-13)


def symmetry_energy(par, n_B, T=0.0, d=2e-3):
    """S(n) [MeV], the coefficient of beta^2 in E/A(beta) = E/A(0) + S beta^2.

    Taken as the curvature at beta = 0,

        S(n) = (1/2) d^2 (E/A) / d beta^2 |_{beta=0} ,

    by a symmetric second difference. This is the standard definition and it is
    the one Constantinou et al. quote: it returns E_sym = 30.848 MeV against
    their 30.85 and L_sym = 41.27 against their 41.26.

    A note for anyone comparing with `eos.did.nmp`, which uses a full step to
    pure neutron matter with a Richardson correction instead. That estimator
    measures the same coefficient only when the quartic term is negligible;
    here it returns 30.776 and 41.124, which is a real difference rather than
    numerical noise -- it carries beta^4 contamination that the published
    numbers do not include. DID needs the full step because its E/A difference
    at small asymmetry sits in numerical noise; ZL's does not, so ZL can take
    the definition directly. The step is stable to the fifth digit across
    d = 1e-3 to 1e-2.
    """
    e = lambda beta: energy_per_baryon(par, n_B, beta, T)
    return 0.5 * (e(d) - 2.0 * e(0.0) + e(-d)) / d ** 2


def _derivative_in_x(f, par, n_sat, order, h=0.02):
    """d^k f/dx^k at x = 1, with x = n_B/n_sat, by central differences.

    `f(par, n_B)` is any quantity of nuclear matter. The step is in x, so h is
    a fraction of saturation density; 0.02 keeps the truncation error below the
    third digit of K_sat while staying far above the solver's residual.
    """
    def at(k):
        return f(par, n_sat * (1.0 + k * h))

    if order == 1:
        return (at(1) - at(-1)) / (2.0 * h)
    if order == 2:
        return (at(1) - 2.0 * at(0) + at(-1)) / h ** 2
    if order == 3:
        return (at(2) - 2.0 * at(1) + 2.0 * at(-1) - at(-2)) / (2.0 * h ** 3)
    raise ValueError(f"derivative order {order} is not implemented")


def compute_nmp(par, T=0.0):
    """The nuclear-matter parameters of a ZL parameter set.

    Returns a dict, all in MeV except n_sat [fm^-3]:

        n_sat       saturation density, where P = 0 in symmetric matter
        E_sat       binding energy per baryon there
        K_sat       incompressibility,  9 n^2 d^2(E/A)/dn^2
        Q_sat       skewness,          27 n^3 d^3(E/A)/dn^3
        E_sym       symmetry energy at saturation
        L_sym       its slope,          3 n dS/dn
        K_sym       its curvature,      9 n^2 d^2S/dn^2

    Every one is a prediction: ZL imposes none of them. The published set of
    Constantinou et al. is pinned in `verify/run_full_check.py`.

    Derivatives in x = n_B/n_sat rather than in n_B directly, so that
    K_sat = 9 d^2(E/A)/dx^2 and L_sym = 3 dS/dx with no density factors left
    to get wrong.
    """
    n_sat = saturation_density(par, T)
    e_of_n = lambda p, n: energy_per_baryon(p, n, 0.0, T)
    s_of_n = lambda p, n: symmetry_energy(p, n, T)

    return {
        "n_sat": n_sat,
        "E_sat": energy_per_baryon(par, n_sat, 0.0, T),
        "K_sat": 9.0 * _derivative_in_x(e_of_n, par, n_sat, 2),
        "Q_sat": 27.0 * _derivative_in_x(e_of_n, par, n_sat, 3),
        "E_sym": symmetry_energy(par, n_sat, T),
        "L_sym": 3.0 * _derivative_in_x(s_of_n, par, n_sat, 1),
        "K_sym": 9.0 * _derivative_in_x(s_of_n, par, n_sat, 2),
    }


# =============================================================================
# THE INVERSE MAP
# =============================================================================
# Closed form, no seed and no iteration: ZL's interaction enters the
# nuclear-matter parameters linearly in (a0, b0, a1, b1) once gamma and gamma1
# are known, and the two exponents themselves come out of ratios of the
# isoscalar data. Every other model in this repository inverts by a multi-
# dimensional root find with a seed, a basin and a restart count; ZL does not,
# which is why an inference run over ZL couplings meets no inversion lottery.
#
# TWO CONVENTIONS, both of which give silently wrong answers if mixed:
#
#   1. REST MASS. Every "b" quantity below is a difference between a total and
#      its free-Fermi-gas part -- Eb = E_sat - E_sat_K and so on -- and that
#      subtraction is only the interaction piece when BOTH sides carry the rest
#      mass or NEITHER does. Here neither does: `energy_per_baryon` subtracts
#      m_p n_p + m_n n_n from eps, so E_sat and E_sat_K are both binding
#      energies and the masses cancel out of Eb. A derivation carrying an
#      explicit "+ m_H" is using the other reading of the same symbol.
#
#   2. WHICH E_sym. `symmetry_energy` above is the QUADRATIC COEFFICIENT,
#      (1/2) d^2(E/A)/dbeta^2 at beta = 0. Constantinou et al. use the full
#      difference E(n_sat, x=0) - E(n_sat, x=0.5) instead, which carries the
#      beta^4 and higher terms with it. On the SHIPPED SET the two read
#
#          quadratic coefficient   E_sym = 30.848   L_sym = 41.270
#          full PNM - SNM step     E_sym = 31.561   L_sym = 42.718
#
#      -- one functional, two conventions, 0.7 and 1.4 MeV apart. So the
#      familiar target {E_sym = 31.6, L_sym = 43} is the shipped set stated
#      in the OTHER convention, and putting it through this inversion returns
#      a1 = -25.20, b1 = 7.19 where the shipped set carries -26.06 and 7.34.
#      That is the convention, not an error: neither number is wrong, they
#      answer different questions. This function is the exact inverse of
#      `compute_nmp`, so it uses `compute_nmp`'s convention throughout, and a
#      round-trip test written against the other one fails for that reason
#      rather than for an error in the algebra.
#
# The kinetic reference is taken from the model's own thermodynamics with the
# interaction switched off, never from a second transcription of the Fermi
# integrals (CLAUDE.md section 7): there is exactly one Fermi gas in this
# repository and this is it, evaluated at a0 = b0 = a1 = b1 = 0.

#: The inversion is exact in the interaction algebra, so the round trip is
#: limited by the stencils `compute_nmp` differentiates with -- K_sat and
#: L_sym return to about 1e-2 MeV, four orders below their own values. A
#: relative mismatch above this on any imposed key means the recovered
#: couplings do not realise the target, not that the difference was coarse.
ROUND_TRIP_GATE = 1e-3


@dataclass
class InversionStatus:
    """What the inversion achieved, as a return value rather than a raise."""
    ok: bool
    message: str
    #: Largest relative mismatch over the imposed keys, measured by putting
    #: the recovered couplings back through `compute_nmp`.
    residual: float
    #: What the closure did NOT impose, computed forward with `compute_nmp`'s
    #: own stencils: {"Q_sat": MeV, "K_sym": MeV} (K_sym only when gamma1 was
    #: the free choice rather than K_sym the imposed datum).
    predictions: dict = dataclass_field(default_factory=dict)


def _kinetic_reference(par, n0):
    """The free-Fermi-gas part of the nuclear-matter parameters at n0.

    Not at saturation: the kinetic gas has no P = 0 root, so there is no
    saturation of its own to evaluate at. Everything is taken at the density
    the inversion is imposing saturation AT, which is what makes the
    interaction pieces below the exact remainder.

    The curvatures use the same operator `compute_nmp` uses -- 9 d^2/dx^2 with
    x = n_B/n0, i.e. 9 n0^2 d^2/dn^2 -- and NOT 9 dP/dn. The two agree only
    where P = 0, which is true of the total and false of the kinetic part
    alone; the missing 18 n P term is carried explicitly by P_K below.
    """
    kin = replace(par, n0=n0, a0=0.0, b0=0.0, a1=0.0, b1=0.0)
    e_of_n = lambda p, n: energy_per_baryon(p, n, 0.0)
    s_of_n = lambda p, n: symmetry_energy(p, n)
    return {
        "E": energy_per_baryon(kin, n0, 0.0),
        "P": pressure(kin, n0, 0.0),
        "K": 9.0 * _derivative_in_x(e_of_n, kin, n0, 2),
        "E_sym": symmetry_energy(kin, n0),
        "L_sym": 3.0 * _derivative_in_x(s_of_n, kin, n0, 1),
        "K_sym": 9.0 * _derivative_in_x(s_of_n, kin, n0, 2),
    }


def invert_nmp(nmp, gamma1=None, par_base=None):
    """Recover ZL couplings from a target nuclear-matter-parameter set.

    The exact inverse of `compute_nmp`, in closed form. The functional's
    reference density n0 is set EQUAL to the requested n_sat, which is what
    makes the isoscalar sector solvable in one line: saturation is then
    imposed at u = 1 rather than found. The shipped set does not have that
    property -- it saturates 0.3 % below its own n0 -- so inverting the
    published nuclear-matter parameters returns a set that agrees with the
    published couplings to a few tenths of a percent rather than exactly
    (gamma to 3e-5, a0 and b0 to 0.3 %).

    Args:
        nmp: the targets, named as `compute_nmp`'s keys. {n_sat [fm^-3],
            E_sat, K_sat, E_sym, L_sym [MeV]} are required. "K_sym" is
            consumed only when it is imposed (below); "Q_sat" is ignored, and
            cannot be imposed -- see the rigidity note below.
        gamma1: the isovector exponent, held fixed rather than fitted. Six
            couplings against five nuclear-matter parameters leaves a
            one-parameter family, and gamma1 is the member of it no
            saturation-density observable constrains. There is deliberately
            NO default: a hidden one would pick the high-density isovector
            behaviour of the returned functional on the caller's behalf.
        par_base: the set everything the closure does not free is inherited
            from -- the nucleon masses and the set name. Defaults to
            published ZL.

    The isovector sector may be closed the other way instead: pass "K_sym" in
    `nmp` and leave `gamma1` unset, and gamma1 is solved for. ZL is the only
    model here that can impose K_sym, because it is the only one with three
    isovector knobs (a1, b1, gamma1) against the three isovector data
    (E_sym, L_sym, K_sym). Passing both, or neither, is a ValueError: they are
    two names for the same freedom.

    **Q_sat cannot be imposed.** In ZL the interaction skewness is rigidly
    Q_sat_pot = 3 (gamma - 2) K_sat_pot, since both come from the single term
    b0 u^gamma; once {n_sat, E_sat, K_sat} fix gamma and b0, Q_sat follows. A
    prior over (K_sat, Q_sat) in ZL lives on a curve, not in a plane.

    Returns:
        (Parameters, InversionStatus). Non-convergence is a RETURN VALUE
        (CLAUDE.md section 6): a sampler walks into targets ZL cannot realise
        -- ones that drive the recovered functional to saturate outside
        `N_SAT_BRACKET`, or that make gamma singular -- and must be able to
        score one and move on. `status.ok` is judged by putting the recovered
        couplings back through `compute_nmp`; it is False, with `parameters`
        still returned, when any imposed key misses by more than
        ROUND_TRIP_GATE.

    Reference: the closed form is the inversion of the potential-energy
    expressions in `zl.tex` section "Nuclear-matter parameters".
    """
    par_base = par_base or Parameters.default()
    impose_K_sym = "K_sym" in nmp
    if impose_K_sym == (gamma1 is not None):
        raise ValueError(
            "the ZL isovector sector has one free choice and needs exactly "
            "one of them named: pass gamma1, or put K_sym in the nmp dict "
            "and leave gamma1 unset. "
            f"Got gamma1={gamma1!r} and K_sym "
            f"{'present' if impose_K_sym else 'absent'}.")

    n0 = nmp["n_sat"]
    kin = _kinetic_reference(par_base, n0)

    # Interaction remainders: what the couplings have to supply on top of the
    # free gas. Both sides of every difference are rest-mass-subtracted.
    Eb = nmp["E_sat"] - kin["E"]
    Kb = nmp["K_sat"] - kin["K"]
    Sb = nmp["E_sym"] - kin["E_sym"]
    Lb = nmp["L_sym"] - kin["L_sym"]

    # Isoscalar sector. E_pot/A = a0 u + b0 u^gamma gives three conditions at
    # u = 1 -- a0 + b0 = Eb, n0 (a0 + gamma b0) = -P_K (total P vanishes), and
    # 9 gamma (gamma - 1) b0 = Kb -- which solve in this order.
    X = Eb * n0 + kin["P"]
    D = (9.0 * Eb + Kb) * n0 ** 2 + 9.0 * kin["P"] * n0
    if X == 0.0 or D == 0.0:
        return None, InversionStatus(
            ok=False, residual=float("inf"),
            message=("degenerate isoscalar target: the interaction pressure "
                     "at n_sat vanishes, so no (b0, gamma) realises it"))
    gamma = -Kb * n0 / (9.0 * X)
    b0 = 9.0 * X * X / D
    a0 = (Kb * Eb * n0 ** 2 - 9.0 * kin["P"] * (Eb * n0 + kin["P"])) / D

    # Isovector sector. E_sym_pot(u) = (a1 - a0) u + b1 u^gamma1 - b0 u^gamma,
    # so (a1, b1) are linear once gamma1 is known; with K_sym imposed instead,
    # gamma1 comes first from the curvature condition
    # K_sym_pot = 9 [gamma1 (gamma1 - 1) b1 - gamma (gamma - 1) b0].
    if impose_K_sym:
        slope = Lb / 3.0 - Sb - X / n0          # = (gamma1 - 1) b1
        if slope == 0.0:
            return None, InversionStatus(
                ok=False, residual=float("inf"),
                message=("degenerate isovector target: E_sym and L_sym leave "
                         "no b1 term, so K_sym cannot select a gamma1"))
        gamma1 = ((nmp["K_sym"] - kin["K_sym"]) / 9.0
                  + gamma * (gamma - 1.0) * b0) / slope
    if gamma1 == 1.0:
        return None, InversionStatus(
            ok=False, residual=float("inf"),
            message="gamma1 = 1 is degenerate: b1 u^gamma1 collapses onto a1 u")
    a1 = (3.0 * gamma1 * Sb - Lb + 3.0 * a0 * (gamma1 - 1.0)
          + 3.0 * b0 * (gamma1 - gamma)) / (3.0 * (gamma1 - 1.0))
    b1 = (3.0 * Sb - Lb + 3.0 * b0 * (1.0 - gamma)) / (3.0 * (1.0 - gamma1))

    par = replace(par_base, n0=n0, a0=a0, b0=b0, gamma=gamma,
                  a1=a1, b1=b1, gamma1=gamma1)
    return par, _round_trip_status(par, nmp, impose_K_sym)


def _round_trip_status(par, nmp, impose_K_sym):
    """Score the recovered couplings by putting them back through the forward
    map, and report the derivatives the closure left free.

    The forward map brackets saturation in `N_SAT_BRACKET` and raises if the
    recovered functional saturates outside it, which a target n_sat near the
    edge of that bracket can produce. That is a failed inversion, not an
    exception at a public boundary (CLAUDE.md section 6).
    """
    imposed = ["n_sat", "E_sat", "K_sat", "E_sym", "L_sym"]
    if impose_K_sym:
        imposed.append("K_sym")
    try:
        got = compute_nmp(par)
    except (ValueError, RuntimeError) as exc:
        return InversionStatus(
            ok=False, residual=float("inf"),
            message=f"the recovered couplings do not saturate in "
                    f"{N_SAT_BRACKET} ({exc})")

    residual = max(abs(got[k] - nmp[k]) / max(abs(nmp[k]), 1.0)
                   for k in imposed)
    predictions = {"Q_sat": got["Q_sat"]}
    if not impose_K_sym:
        predictions["K_sym"] = got["K_sym"]
    ok = residual <= ROUND_TRIP_GATE
    return InversionStatus(
        ok=ok, residual=residual, predictions=predictions,
        message="converged" if ok else
                f"round trip misses by {residual:.2e}, above "
                f"ROUND_TRIP_GATE = {ROUND_TRIP_GATE:g}")


def from_nmp(nmp, gamma1=None, par_base=None, return_status=False):
    """Nuclear-matter parameters -> a `Parameters` carrying those couplings.

    The convenience face of `invert_nmp`: same arguments, returns the
    parameters alone unless `return_status`. Raises when the inversion did not
    converge, since a caller asking only for parameters has nowhere to put a
    failure -- use `invert_nmp` directly to score a target instead of raising
    on it.

        par = from_nmp({"n_sat": 0.16, "E_sat": -16.0, "K_sat": 250.0,
                        "E_sym": 31.6, "L_sym": 43.0}, gamma1=2.45)
    """
    par, status = invert_nmp(nmp, gamma1=gamma1, par_base=par_base)
    if not status.ok:
        raise RuntimeError(f"NMP inversion failed: {status.message}")
    return (par, status) if return_status else par


if __name__ == "__main__":
    par = Parameters.default()
    for name, value in compute_nmp(par).items():
        print(f"  {name:<7} {value:12.5f}")
