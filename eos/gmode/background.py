"""
gmode/background.py
===================
The stellar background a non-radial mode solver needs: the TOV structure with
its *radial profiles retained*, both metric functions, the local gravity, and
the Brunt-Vaisala frequency.

Why this module exists
----------------------
`eos.tov.solver` integrates the TOV equations and returns scalars (M, R, k2,
Lambda); the profiles are discarded and the metric function `nu(r)` is never
formed, because a static structure calculation does not need it. A g-mode does:
the Cowling system is an ODE *in* `r` whose coefficients are the background
fields. This module therefore re-integrates the same equations keeping the
profiles, and adds `nu`. It imports the equation-of-state plumbing from
`eos.tov.solver` rather than duplicating it, and does not modify it.

Metric and units
----------------
Interior Schwarzschild metric with signature (-, +, +, +),

    ds^2 = -e^{nu(r)} dt^2 + e^{lambda(r)} dr^2 + r^2 (dtheta^2 + sin^2 theta dphi^2)

    e^{lambda} = (1 - 2 m(r) / r)^{-1}

Everything inside is in **geometric units**: `r` and `m` in km, `P` and `eps` in
km^-2, `nu` and `lambda` dimensionless, the local gravity `g` and the
Brunt-Vaisala frequency `N` in km^-1. This is the natural system for the
oscillation equations, in which G = c = 1 and an angular frequency is a
wavenumber; `omega_to_hz` converts back. Public arguments and the `n_B` column
stay fm-based as everywhere else in `eos` (n_B in fm^-3, table P and eps in
MeV/fm^3).

`nu` needs no ODE of its own. Relativistic hydrostatic equilibrium,

    dP/dr = -(eps + P) * (1/2) dnu/dr ,

fixes it up to a constant from the pressure profile, and the constant follows
from matching to the exterior Schwarzschild solution at the surface,
e^{nu(R)} = 1 - 2M/R. The same relation defines the local gravitational
acceleration used by the mode equations,

    g = -(dP/dr) / (eps + P) = (1/2) dnu/dr .

Brunt-Vaisala frequency
-----------------------
    N^2 = g^2 (1/c_eq^2 - 1/c_ad^2) e^{nu - lambda}

with `c_eq^2 = dP/deps` taken along the equilibrium sequence and `c_ad^2 =
(dP/deps)_x` at frozen composition. Buoyancy exists only where the two differ,
so `N^2` is a direct measure of the composition gradient; a star with a single
sound speed supports no composition g-mode at all. Stable stratification means
`c_ad > c_eq`, hence `N^2 > 0`.

Reference: Jaikumar, Semposki, Prakash and Constantinou, "g-mode oscillations
in hybrid stars: A tale of two sounds", Phys. Rev. D 103, 123009 (2021),
Eqs. (2)-(5). The equilibrium structure follows Tolman (1939) and Oppenheimer
and Volkoff (1939).
"""
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from eos.general.physics_constants import MEV_FM3_TO_KM2_INV, r_sun_km

# Half the solar Schwarzschild radius: G M_sun / c^2, i.e. 1 M_sun expressed in
# km. Derived from the same constant `eos.tov` uses so the two agree by
# construction rather than by a second hard-coded literal.
KM_PER_MSUN = 0.5 * r_sun_km

# Speed of light in km/s, for turning an angular frequency in km^-1 into Hz.
C_KM_S = 2.99792458e5


def omega_to_hz(omega):
    """Angular frequency in km^-1 -> ordinary frequency in Hz."""
    return np.asarray(omega) * C_KM_S / (2.0 * np.pi)


def hz_to_omega(nu_hz):
    """Ordinary frequency in Hz -> angular frequency in km^-1."""
    return np.asarray(nu_hz) * 2.0 * np.pi / C_KM_S


@dataclass
class StellarBackground:
    """Equilibrium star sampled on a radial grid, in geometric units.

    Every array is defined on `r` and has the same length. `M` and `R` are the
    gravitational mass and circumferential radius; `M_msun` and `R_km` are the
    same numbers in the units the rest of `eos` reports.

    r        : radial coordinate [km], ascending, r[0] > 0
    m        : enclosed gravitational mass [km]
    P, eps   : pressure and energy density [km^-2]
    n_B      : baryon number density [fm^-3] (diagnostic; not used by the modes)
    e_lam    : e^{lambda} = (1 - 2m/r)^{-1}
    e_nu     : e^{nu}, normalised so e^{nu(R)} = 1 - 2M/R
    g        : local gravity -(dP/dr)/(eps + P) [km^-1]
    cs2_eq   : equilibrium sound speed squared, dimensionless (units of c)
    cs2_ad   : frozen/adiabatic sound speed squared; may be complex if a finite
               reaction rate was folded in (see `eos.gmode.sound_speeds`)
    N2       : Brunt-Vaisala frequency squared [km^-2]; complex when cs2_ad is
    gamma    : chemical equilibration rate [s^-1] carried onto the star, or
               None. Only used to rebuild N^2 at a trial frequency.
    """
    r: np.ndarray
    m: np.ndarray
    P: np.ndarray
    eps: np.ndarray
    n_B: np.ndarray
    e_lam: np.ndarray
    e_nu: np.ndarray
    g: np.ndarray
    cs2_eq: np.ndarray
    cs2_ad: np.ndarray
    N2: np.ndarray
    gamma: np.ndarray = None

    def at_frequency(self, omega_s):
        """A copy whose buoyancy uses the dynamical sound speed at `omega_s`.

        `omega_s` is an angular frequency in s^-1. The stellar structure --
        m, P, eps, the metric, g -- does not depend on the sound speeds at all,
        so a finite reaction rate changes only `cs2_ad` and `N2`. Rebuilding
        just those avoids re-solving the TOV equations (and re-searching for
        the target mass) every time the trial frequency moves.

        Returns `self` unchanged when no rate was supplied.
        """
        from dataclasses import replace
        from eos.gmode.sound_speeds import cs2_dynamical

        if self.gamma is None:
            return self
        cs2_dy = cs2_dynamical(self.cs2_eq, self.cs2_ad, self.gamma, omega_s)
        return replace(self, cs2_ad=cs2_dy,
                       N2=brunt_vaisala(self.g, self.cs2_eq, cs2_dy,
                                        self.e_nu, self.e_lam))

    @property
    def R(self):
        """Circumferential radius [km]."""
        return float(self.r[-1])

    @property
    def M(self):
        """Gravitational mass [km]."""
        return float(self.m[-1])

    @property
    def R_km(self):
        return self.R

    @property
    def M_msun(self):
        """Gravitational mass [M_sun]."""
        return self.M / KM_PER_MSUN

    @property
    def is_complex(self):
        """True when a finite reaction rate has made the buoyancy complex."""
        return np.iscomplexobj(self.N2) or np.iscomplexobj(self.cs2_ad)

    def interpolators(self):
        """Monotone interpolants of every field needed by the mode equations.

        Returns a dict of callables of `r`. Complex fields are interpolated as
        two real splines and recombined, since `PchipInterpolator` is real-only.
        """
        def build(y):
            y = np.asarray(y)
            if np.iscomplexobj(y):
                re = PchipInterpolator(self.r, y.real, extrapolate=True)
                im = PchipInterpolator(self.r, y.imag, extrapolate=True)
                return lambda x: re(x) + 1j * im(x)
            return PchipInterpolator(self.r, y, extrapolate=True)

        return {name: build(getattr(self, name)) for name in
                ("m", "P", "eps", "e_lam", "e_nu", "g", "cs2_eq", "cs2_ad", "N2")}


def brunt_vaisala(g, cs2_eq, cs2_ad, e_nu, e_lam):
    """N^2 = g^2 (1/c_eq^2 - 1/c_ad^2) e^{nu - lambda}, in km^-2.

    Arguments are arrays in the geometric units of `StellarBackground`. Where
    the two sound speeds are equal the result is exactly zero: no composition
    gradient, no buoyancy. `cs2_ad` may be complex, in which case so is N^2.
    """
    g = np.asarray(g)
    cs2_eq = np.asarray(cs2_eq)
    cs2_ad = np.asarray(cs2_ad)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 1.0 / cs2_eq - 1.0 / cs2_ad
    delta = np.where(np.isfinite(delta), delta, 0.0)
    return g**2 * delta * e_nu / e_lam


def with_crust(eos, cs2_eq, cs2_ad, crust="BPS", n_transition=0.08,
               custom_path=None):
    """Prepend a tabulated crust to a core equation of state, with N^2 = 0.

    Returns `(eos_full, cs2_eq_full, cs2_ad_full)` ready for
    `build_background`. The crust rows get `cs2_ad = cs2_eq`, so the
    Brunt-Vaisala frequency vanishes there: a tabulated crust carries no
    composition information, and the standard treatment in the g-mode
    literature is to model it as a homogeneous fluid, which supports no
    composition buoyancy. The core g-mode is insensitive to this, being
    confined well inside the crust.

    eos          : core `EOSTable_for_TOV`
    cs2_eq/cs2_ad: sound speeds on the core rows
    crust        : name understood by `eos.tov.solver.load_crust_table`, or
                   "No" to skip and return the input unchanged
    n_transition : baryon density [fm^-3] below which crust rows are kept
    """
    from eos.tov.solver import EOSTable_for_TOV, load_crust_table

    cs2_eq = np.asarray(cs2_eq)
    cs2_ad = np.asarray(cs2_ad)
    if crust in (None, "No", "no", False):
        return eos, cs2_eq, cs2_ad

    ct = load_crust_table(crust, custom_path=custom_path)
    keep = ct.nB < n_transition
    core = np.asarray(eos.nB) >= n_transition
    if not np.any(keep):
        raise ValueError(f"crust {crust!r} has no rows below "
                         f"n_transition = {n_transition} fm^-3")

    # The crust's own dP/deps: equilibrium and frozen coincide by construction.
    cs2_crust = np.gradient(ct.P[keep], ct.epsilon[keep])
    cs2_crust = np.clip(cs2_crust, 1e-8, 1.0)

    full = EOSTable_for_TOV(
        P=np.concatenate([ct.P[keep], np.asarray(eos.P)[core]]),
        epsilon=np.concatenate([ct.epsilon[keep],
                                np.asarray(eos.epsilon)[core]]),
        nB=np.concatenate([ct.nB[keep], np.asarray(eos.nB)[core]]))
    return (full,
            np.concatenate([cs2_crust, cs2_eq[core]]),
            np.concatenate([cs2_crust, cs2_ad[core]]))


def _tov_rhs(r, y, eps_grid, P_grid):
    """RHS of [dm/dr, dP/dr, dnu/dr] in geometric units.

    dm/dr  = 4 pi r^2 eps
    dP/dr  = -(eps + P) (m + 4 pi r^3 P) / (r (r - 2m))
    dnu/dr = -2 (dP/dr) / (eps + P) = 2 (m + 4 pi r^3 P) / (r (r - 2m))
    """
    m, P, _nu = y
    if P <= 0.0:
        return np.array([0.0, 0.0, 0.0])
    eps = np.interp(P, P_grid, eps_grid)
    denom = r * (r - 2.0 * m)
    if denom <= 0.0:
        return np.array([0.0, 0.0, 0.0])
    dnu_dr = 2.0 * (m + 4.0 * np.pi * r**3 * P) / denom
    return np.array([
        4.0 * np.pi * r**2 * eps,
        -0.5 * (eps + P) * dnu_dr,
        dnu_dr,
    ])


def build_background(eos, cs2_eq, cs2_ad, e_c=None, M_target=None,
                     n_points=800, r_max=40.0, P_surf_rel=1e-9,
                     e_c_bracket=None, gamma=None):
    """Integrate the TOV equations keeping the profiles, and form N^2.

    eos       : `EOSTable_for_TOV` (P, epsilon in MeV/fm^3, nB in fm^-3), or any
                object with those three attributes. Must be ascending in P.
    cs2_eq    : equilibrium c^2 = dP/deps, one value per row of `eos`,
                dimensionless. `eos.mixed.coefficients.sound_speed_eq` produces
                exactly this from the table's own P and eps columns.
    cs2_ad    : frozen c^2 = (dP/deps)_x on the same grid. Set it equal to
                `cs2_eq` wherever composition is unavailable (a tabulated crust,
                say); N^2 then vanishes there, which is the standard treatment
                of the crust as a homogeneous fluid. May be complex.
    e_c       : central energy density [MeV/fm^3]. Give this or `M_target`.
    M_target  : gravitational mass [M_sun] to solve for instead, by bisection on
                the central density. Slower, but it is how published g-mode
                results are quoted.
    n_points  : radial samples in the returned profiles.
    r_max     : integration cut-off [km]; the surface event fires long before.
    P_surf_rel: surface pressure as a fraction of the central pressure.
    e_c_bracket: (lo, hi) in MeV/fm^3 for the `M_target` bisection. Defaults to
                the span of the table.
    gamma     : optional chemical equilibration rate [s^-1], a scalar or one
                value per row of `eos`. It is carried onto the star so that
                `StellarBackground.at_frequency` can rebuild the buoyancy at a
                trial mode frequency without re-solving the structure.

    Returns a `StellarBackground`.
    """
    P_tab = np.asarray(eos.P, dtype=float) * MEV_FM3_TO_KM2_INV
    e_tab = np.asarray(eos.epsilon, dtype=float) * MEV_FM3_TO_KM2_INV
    n_tab = np.asarray(eos.nB, dtype=float)
    cs2_eq = np.asarray(cs2_eq)
    cs2_ad = np.asarray(cs2_ad)
    if not (len(P_tab) == len(e_tab) == len(n_tab) == len(cs2_eq) == len(cs2_ad)):
        raise ValueError(
            "eos columns and sound-speed arrays must have equal length "
            f"(got P:{len(P_tab)} eps:{len(e_tab)} nB:{len(n_tab)} "
            f"cs2_eq:{len(cs2_eq)} cs2_ad:{len(cs2_ad)})")

    if gamma is not None:
        gamma = np.broadcast_to(np.asarray(gamma, dtype=float),
                                P_tab.shape).copy()

    order = np.argsort(P_tab)
    P_tab, e_tab, n_tab = P_tab[order], e_tab[order], n_tab[order]
    cs2_eq, cs2_ad = cs2_eq[order], cs2_ad[order]
    if gamma is not None:
        gamma = gamma[order]
    # A Maxwell plateau repeats P; keep the last (high-density) entry so the
    # inverse map eps(P) stays single-valued, matching `eos.tov`'s convention.
    keep = np.concatenate([np.diff(P_tab) > 0, [True]])
    P_tab, e_tab, n_tab = P_tab[keep], e_tab[keep], n_tab[keep]
    cs2_eq, cs2_ad = cs2_eq[keep], cs2_ad[keep]
    if gamma is not None:
        gamma = gamma[keep]

    if M_target is not None:
        return _background_at_mass(
            eos, cs2_eq, cs2_ad, M_target, n_points, r_max, P_surf_rel,
            e_c_bracket, P_tab, e_tab, n_tab, gamma)
    if e_c is None:
        raise ValueError("give either e_c or M_target")

    return _integrate(P_tab, e_tab, n_tab, cs2_eq, cs2_ad,
                      e_c * MEV_FM3_TO_KM2_INV, n_points, r_max, P_surf_rel,
                      gamma)


def _integrate(P_tab, e_tab, n_tab, cs2_eq, cs2_ad, e_c_geo,
               n_points, r_max, P_surf_rel, gamma=None):
    """The actual integration, everything already in geometric units."""
    P_c = float(np.interp(e_c_geo, e_tab, P_tab))
    if P_c <= 0.0:
        raise ValueError(f"central pressure <= 0 for e_c = {e_c_geo}")
    P_surf = P_c * P_surf_rel

    # Regular series start: m = (4/3) pi eps_c r^3 at small r. nu is integrated
    # from an arbitrary zero and shifted onto the exterior solution afterwards.
    r0 = 1e-4
    y0 = np.array([4.0 / 3.0 * np.pi * e_c_geo * r0**3, P_c, 0.0])

    # scipy passes `args` to the events as well as to the RHS.
    def surface(r, y, *_args):
        return y[1] - P_surf
    surface.terminal = True
    surface.direction = -1

    sol = solve_ivp(_tov_rhs, [r0, r_max], y0, args=(e_tab, P_tab),
                    method="DOP853", events=surface, dense_output=True,
                    rtol=1e-10, atol=1e-14)
    if sol.t_events[0].size == 0:
        raise RuntimeError(
            f"star did not terminate below r_max = {r_max} km; the table "
            "probably does not extend low enough in pressure")

    R = float(sol.t_events[0][0])
    r = np.linspace(r0, R, n_points)
    m, P, nu_raw = sol.sol(r)
    M = float(m[-1])

    # Match to the exterior: e^{nu(R)} = 1 - 2M/R.
    nu = nu_raw + np.log(1.0 - 2.0 * M / R) - nu_raw[-1]
    e_nu = np.exp(nu)
    e_lam = 1.0 / (1.0 - 2.0 * m / r)

    eps = np.interp(P, P_tab, e_tab)
    n_B = np.interp(P, P_tab, n_tab)
    g = (m + 4.0 * np.pi * r**3 * P) / (r * (r - 2.0 * m))

    # Sound speeds live on the table's P grid; carry them onto the star. The
    # equilibrium one is bounded away from zero so that 1/c_eq^2 stays finite
    # on a Maxwell plateau, where dP/deps genuinely vanishes.
    cs2_eq_r = np.clip(np.interp(P, P_tab, cs2_eq), 1e-8, 1.0)
    if np.iscomplexobj(cs2_ad):
        cs2_ad_r = (np.interp(P, P_tab, cs2_ad.real)
                    + 1j * np.interp(P, P_tab, cs2_ad.imag))
    else:
        cs2_ad_r = np.clip(np.interp(P, P_tab, cs2_ad), 1e-8, 1.0)

    N2 = brunt_vaisala(g, cs2_eq_r, cs2_ad_r, e_nu, e_lam)
    gamma_r = None if gamma is None else np.interp(P, P_tab, gamma)

    return StellarBackground(
        r=r, m=m, P=P, eps=eps, n_B=n_B, e_lam=e_lam, e_nu=e_nu, g=g,
        cs2_eq=cs2_eq_r, cs2_ad=cs2_ad_r, N2=N2, gamma=gamma_r)


def _background_at_mass(eos, cs2_eq, cs2_ad, M_target, n_points, r_max,
                        P_surf_rel, e_c_bracket, P_tab, e_tab, n_tab,
                        gamma=None):
    """Bisect on central density until M(e_c) = M_target.

    Bisection rather than a Newton step, and preceded by a coarse scan: M(e_c)
    turns over at the maximum mass, so a target mass below M_max is reached on
    *two* branches and only the low-density one is stable. The scan takes the
    first crossing, which is the stable one.
    """
    def mass_of(e_c):
        bg = _integrate(P_tab, e_tab, n_tab, cs2_eq, cs2_ad,
                        e_c * MEV_FM3_TO_KM2_INV, 32, r_max, P_surf_rel)
        return bg.M_msun

    if e_c_bracket is not None:
        lo, hi = e_c_bracket
        f_lo, f_hi = mass_of(lo) - M_target, mass_of(hi) - M_target
        if f_lo * f_hi > 0:
            raise ValueError(
                f"M = {M_target} M_sun not bracketed by the requested e_c "
                f"[{lo:.1f}, {hi:.1f}] MeV/fm^3, which spans M in "
                f"[{f_lo + M_target:.3f}, {f_hi + M_target:.3f}] M_sun")
    else:
        e_min = float(np.min(eos.epsilon))
        e_max = float(np.max(eos.epsilon))
        scan = np.logspace(np.log10(max(e_min * 2.0, 100.0)),
                           np.log10(e_max * 0.98), 24)
        masses = []
        for e in scan:
            try:
                masses.append(mass_of(e))
            except (RuntimeError, ValueError):
                masses.append(np.nan)
        masses = np.asarray(masses)
        f = masses - M_target
        ok = np.isfinite(f)
        idx = [i for i in range(len(f) - 1)
               if ok[i] and ok[i + 1] and f[i] * f[i + 1] <= 0]
        if not idx:
            reached = np.nanmax(masses) if ok.any() else float("nan")
            raise ValueError(
                f"M = {M_target} M_sun is not reached by this equation of "
                f"state: scanning e_c over [{scan[0]:.1f}, {scan[-1]:.1f}] "
                f"MeV/fm^3 gives M_max = {reached:.3f} M_sun. Pass e_c_bracket "
                "to search a different range.")
        i = idx[0]                                  # first = stable branch
        lo, hi, f_lo = scan[i], scan[i + 1], f[i]

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = mass_of(mid) - M_target
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1e-4 * mid:
            break
    e_c = 0.5 * (lo + hi)
    return _integrate(P_tab, e_tab, n_tab, cs2_eq, cs2_ad,
                      e_c * MEV_FM3_TO_KM2_INV, n_points, r_max, P_surf_rel,
                      gamma)
