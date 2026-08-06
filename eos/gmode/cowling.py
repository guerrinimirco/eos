"""
gmode/cowling.py
================
Non-radial fluid oscillations of a non-rotating star in the relativistic
Cowling approximation: the eigenvalue problem whose g-mode branch this package
exists to solve.

The approximation
-----------------
The Cowling approximation holds the spacetime fixed and perturbs only the
fluid, dropping the metric perturbations from the problem. That reduces a
fourth-order complex system with outgoing-wave conditions at infinity to a
second-order real one confined to the star, at the cost of losing the
gravitational-wave damping time (the eigenfrequency comes out real). For
g-modes the frequency error is at the few-per-cent level, because a g-mode's
density perturbation is concentrated in the core and nearly divergence-free on
large scales; it is a much worse approximation for the f-mode.

The equations
-------------
With the displacement expanded in spherical harmonics of degree `l` and time
dependence e^{+i omega t}, the perturbation equations reduce to two first-order
equations for

    U = r^2 e^{lambda/2} xi_r        (radial displacement)
    V = delta p / (eps + p)          (Eulerian pressure perturbation)

namely

    dU/dr = (g / c_ad^2) U + e^{lambda/2} [ l(l+1) e^nu / omega^2
                                            - r^2 / c_ad^2 ] V
    dV/dr = e^{lambda/2 - nu} (omega^2 - N^2) / r^2 * U
            + g * (1/c_eq^2 - 1/c_ad^2) * V

Note which sound speed goes where: the *adiabatic* one appears in the equations
of motion, because a fluid element responds to a compression at frozen
composition; the *equilibrium* one enters only through the buoyancy terms,
which is why a star with a single sound speed has N^2 = 0 and no g-mode branch
at all. Reference: Jaikumar, Semposki, Prakash and Constantinou,
Phys. Rev. D 103, 123009 (2021), Eq. (79); see also Thorne and Campolattaro,
Astrophys. J. 149, 591 (1967) and McDermott, Van Horn and Scholl,
Astrophys. J. 268, 837 (1983).

Boundary conditions
-------------------
Regularity at the centre gives, up to an arbitrary overall normalisation,

    U -> r^{l+1} ,      V -> (omega^2 / l) r^l e^{-nu(0)}

(the same reference, Eqs. (77)-(78); the e^{-nu(0)} is what makes the two
series consistent at leading order). At the surface the Lagrangian pressure
perturbation must vanish, Delta p = delta p + xi_r dp/dr = 0, i.e.

    V(R) = g(R) U(R) e^{-lambda(R)/2} / R^2 .

Both conditions hold only for discrete `omega`: that is the eigenvalue problem.

Mode classification
-------------------
Integrating from the centre with the regular series always succeeds; the
surface condition is what selects the spectrum. The textbook labelling counts
nodes of the radial eigenfunction `xi_r`, which shares its zeros with `U`: the
f-mode has none, and the g_k and p_k overtones have k each, the g-branch lying
below the f-mode and the p-branch above. So the *fundamental* g-mode is the
highest-frequency root below the f-mode and has exactly one node. It is the one
that matters observationally, being both the highest in frequency and the most
strongly tidally coupled.

In practice the branches are separated here by the buoyancy scale instead. A
g-mode is trapped where buoyancy can restore the displacement, which needs
omega < N, so the whole g-branch lies below max(N), while the f-mode sits near
the dynamical frequency sqrt(M/R^3) -- typically several times higher. That
separation is wide and numerically stable, whereas a node count can gain or
lose an apparent zero near the surface depending on where the star is
truncated. The node count is still reported on every `Mode` as a diagnostic,
and for a well-resolved g_k it agrees.

Finite reaction rates
---------------------
When the background carries a complex dynamical sound speed (see
`eos.gmode.sound_speeds.cs2_dynamical`), N^2 and the coefficients are complex
and so is the eigenfrequency. Its imaginary part is the bulk-viscous damping
rate of the mode. The complex root is found by seeding from the real solution,
which also fixes the mode's identity: node counting is not meaningful for a
complex eigenfunction, so the mode is labelled by the real problem it
continues from.

**Sign convention.** The equations above contain omega only as omega^2, so they
are blind to the sign convention in the time dependence; the dynamical sound
speed is not, entering linearly through gamma/(i omega). We follow its source
and use e^{+i omega t} throughout, in which an amplitude behaves as
e^{-Im(omega) t} and a *damped* mode therefore has **Im(omega) > 0**. Mixing
this with the opposite convention silently flips the sign of the damping while
leaving the frequency untouched, so it is worth stating explicitly.

Units follow `eos.gmode.background`: geometric, with `r` in km and `omega` in
km^-1. Public results are also reported in Hz.
"""
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from eos.gmode.background import omega_to_hz, hz_to_omega


@dataclass
class Mode:
    """One solved oscillation mode.

    omega     : angular eigenfrequency [km^-1]; complex when the background
                carries a finite reaction rate. With the e^{+i omega t}
                convention used here a damped mode has Im(omega) > 0.
    nu_hz     : ordinary frequency [Hz]; the real part if complex
    tau_s     : e-folding damping time [s], or None when the background is real
                (the Cowling approximation carries no gravitational-wave
                damping, so a finite tau here is bulk-viscous only)
    l         : spherical-harmonic degree
    n_nodes   : nodes of xi_r inside the star
    label     : 'f', 'g1', 'g2', ..., 'p1', ...
    r         : radial grid [km]
    xi_r      : radial displacement eigenfunction (arbitrary normalisation)
    dp_over_h : delta p / (eps + p) on the same grid
    """
    omega: complex
    nu_hz: float
    tau_s: float
    l: int
    n_nodes: int
    label: str
    r: np.ndarray
    xi_r: np.ndarray
    dp_over_h: np.ndarray

    @property
    def is_gmode(self):
        return self.label.startswith("g")


def _rhs(r, y, l, itp, omega2):
    """Right-hand side of the Cowling pair; `y` is [U, V]."""
    U, V = y[0], y[1]
    e_lam = itp["e_lam"](r)
    e_nu = itp["e_nu"](r)
    g = itp["g"](r)
    cs2_ad = itp["cs2_ad"](r)
    cs2_eq = itp["cs2_eq"](r)
    N2 = itp["N2"](r)
    sqrt_lam = np.sqrt(e_lam)

    dU = (g / cs2_ad) * U + sqrt_lam * (l * (l + 1) * e_nu / omega2
                                        - r**2 / cs2_ad) * V
    dV = (sqrt_lam / e_nu) * (omega2 - N2) / r**2 * U \
        + g * (1.0 / cs2_eq - 1.0 / cs2_ad) * V
    return [dU, dV]


def integrate_mode(bg, omega, l=2, itp=None, dense=True, rtol=1e-8,
                   atol=1e-12):
    """Integrate the Cowling system at a trial `omega`, centre to surface.

    Returns (r, U, V). `omega` may be complex, in which case so are U and V.
    The normalisation is the one fixed by the central series; the equations are
    linear, so the overall scale is arbitrary and carries no physics.

    `dense=False` samples only the surface, which is all the eigenvalue search
    needs and is several times cheaper; the eigenfunction is worth materialising
    only once a root has been accepted.
    """
    itp = itp if itp is not None else bg.interpolators()
    r = bg.r
    r0, R = float(r[0]), float(r[-1])

    complex_run = bool(np.iscomplexobj(bg.N2) or np.iscomplexobj(omega)
                       or np.iscomplexobj(bg.cs2_ad))
    omega2 = complex(omega)**2 if complex_run else float(np.real(omega))**2
    t_eval = r if dense else None

    U0 = r0**(l + 1)
    V0 = omega2 / l * r0**l / itp["e_nu"](r0)

    if complex_run:
        # scipy integrates real systems only; split into real and imaginary
        # parts and integrate the doubled real system.
        def rhs(rr, y):
            dU, dV = _rhs(rr, [y[0] + 1j * y[2], y[1] + 1j * y[3]],
                          l, itp, omega2)
            return [dU.real, dV.real, dU.imag, dV.imag]

        y0 = [np.real(U0), np.real(V0), np.imag(U0), np.imag(V0)]
        sol = solve_ivp(rhs, [r0, R], y0, t_eval=t_eval, method="DOP853",
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Cowling integration failed: {sol.message}")
        U = sol.y[0] + 1j * sol.y[2]
        V = sol.y[1] + 1j * sol.y[3]
    else:
        def rhs(rr, y):
            return _rhs(rr, y, l, itp, omega2)

        sol = solve_ivp(rhs, [r0, R], [float(U0), float(V0)], t_eval=t_eval,
                        method="DOP853", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Cowling integration failed: {sol.message}")
        U, V = sol.y[0], sol.y[1]
    return sol.t, U, V


def surface_discriminant(bg, omega, l=2, itp=None, dense=True,
                         normalize=True, **kw):
    """Surface condition; its zeros are the eigenfrequencies.

        D_raw = V(R) - g(R) U(R) e^{-lambda(R)/2} / R^2

    With `normalize=True` this is divided by the sum of the magnitudes of the
    two terms, which keeps D of order unity whatever the eigenfunction's
    normalisation and makes it safe to bracket sign changes across many decades
    in omega. That is what the real-axis scan wants.

    With `normalize=False` the raw difference is returned. The normalisation
    involves absolute values and so is *not* an analytic function of omega;
    the complex root finder needs the raw form, which is analytic, or its
    secant steps are meaningless.
    """
    itp = itp if itp is not None else bg.interpolators()
    r, U, V = integrate_mode(bg, omega, l=l, itp=itp, dense=dense, **kw)
    R = float(r[-1])
    target = itp["g"](R) * U[-1] / np.sqrt(itp["e_lam"](R)) / R**2
    raw = V[-1] - target
    if not normalize:
        return raw, r, U, V
    scale = abs(V[-1]) + abs(target)
    if scale == 0.0:
        return 0.0, r, U, V
    return raw / scale, r, U, V


def _nodes(U, rel_floor=1e-6):
    """Number of interior sign changes of the (real) eigenfunction.

    Samples below `rel_floor` times the peak amplitude are dropped first. Near
    the surface the eigenfunction is many orders of magnitude smaller than in
    the core, and counting its sign there measures rounding, not physics.
    """
    u = np.real(U)[1:-1]
    if u.size < 2:
        return 0
    peak = np.max(np.abs(u))
    if peak <= 0.0:
        return 0
    nz = u[np.abs(u) > rel_floor * peak]
    if nz.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(np.sign(nz)) != 0))


def mode_spectrum(bg, l=2, nu_min=30.0, nu_max=3500.0, n_scan=220,
                  rtol=1e-8, atol=1e-12):
    """All modes of the background between `nu_min` and `nu_max` [Hz].

    Scans the surface discriminant on a log-spaced frequency grid, brackets its
    sign changes and refines each with Brent's method, then labels the roots by
    node count relative to the f-mode. Returns a list of `Mode`, ascending in
    frequency.

    Widen the window if a mode is missing: g-modes of a cold nucleonic star sit
    near 100-300 Hz, of a hybrid star with a mixed phase near 400-700 Hz, and
    the f-mode near 1.5-2.5 kHz.
    """
    if bg.is_complex:
        raise ValueError(
            "mode_spectrum needs a real background; a complex dynamical sound "
            "speed makes the eigenfrequency complex, so use solve_gmode, which "
            "seeds the complex root from the real problem")

    itp = bg.interpolators()
    grid = np.logspace(np.log10(hz_to_omega(nu_min)),
                       np.log10(hz_to_omega(nu_max)), n_scan)

    def disc(w):
        return surface_discriminant(bg, w, l=l, itp=itp, dense=False,
                                    rtol=rtol, atol=atol)[0]

    vals = np.array([disc(w) for w in grid])
    roots = []
    for i in range(len(grid) - 1):
        a, b, fa, fb = grid[i], grid[i + 1], vals[i], vals[i + 1]
        if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
            continue
        # Every sign change is a genuine root: D is a ratio of continuous
        # quantities whose denominator is the sum of the two terms' magnitudes,
        # so it has no poles. It does saturate near +-1 between roots, because
        # away from an eigenfrequency one term dominates the other by orders of
        # magnitude; a bracket showing +0.99 -> -0.99 is a root the scan grid
        # merely stepped over, not an artefact.
        try:
            w = brentq(disc, a, b, xtol=1e-14, rtol=1e-12)
        except (ValueError, RuntimeError):
            continue
        _d, r, U, V = surface_discriminant(bg, w, l=l, itp=itp,
                                           rtol=rtol, atol=atol)
        roots.append((w, _nodes(U), r, U, V))

    if not roots:
        return []

    # Classify by the buoyancy scale rather than by node count. A g-mode is
    # trapped in the region where the buoyancy can restore the displacement,
    # which requires omega < N, so the whole g-branch lies below max(N) while
    # the f-mode sits at the dynamical frequency ~sqrt(M/R^3), typically a
    # factor of several higher. That separation is wide and numerically stable.
    #
    # Node counting is the textbook classifier (f has none, g_k and p_k have k)
    # but it is delicate: a g-mode's eigenfunction can pick up or lose an
    # apparent zero near the surface depending on where the star is truncated.
    # It is therefore reported on each `Mode` as a diagnostic and not used to
    # label. Within each branch the labels come from ordering, so they are
    # always distinct.
    roots.sort(key=lambda x: x[0])
    N_max = float(np.sqrt(max(np.max(np.real(bg.N2)), 0.0)))

    g_idx = [i for i, (w, *_) in enumerate(roots) if w < N_max]
    rest = [i for i, (w, *_) in enumerate(roots) if w >= N_max]

    labels = {}
    for k, i in enumerate(reversed(g_idx), start=1):     # descending in omega
        labels[i] = f"g{k}"
    for k, i in enumerate(rest):
        labels[i] = "f" if k == 0 else f"p{k}"

    modes = []
    for i, (w, n, r, U, V) in enumerate(roots):
        xi_r = U / np.sqrt(itp["e_lam"](r)) / r**2
        modes.append(Mode(omega=w, nu_hz=float(omega_to_hz(w)), tau_s=None,
                          l=l, n_nodes=n, label=labels[i],
                          r=r, xi_r=xi_r, dp_over_h=V))
    return modes


def solve_gmode(bg, l=2, order=1, nu_min=30.0, nu_max=3500.0, n_scan=220,
                bg_complex=None, rtol=1e-8, atol=1e-12):
    """The g-mode of radial order `order` (1 = fundamental, highest frequency).

    bg         : real `StellarBackground`
    order      : 1 for the fundamental g-mode, 2 for the first overtone, ...
    bg_complex : optional second background, identical to `bg` except that its
                 `cs2_ad`/`N2` carry a complex dynamical sound speed. When
                 given, the real root is used as a seed and the eigenvalue is
                 recomputed in the complex plane, so the returned `Mode` has a
                 finite bulk-viscous damping time.

    Raises `RuntimeError` if no g-mode of that order is found in the window,
    which for a star with no composition gradient (c_ad == c_eq everywhere, so
    N^2 == 0) is the correct answer rather than a failure.
    """
    modes = mode_spectrum(bg, l=l, nu_min=nu_min, nu_max=nu_max,
                          n_scan=n_scan, rtol=rtol, atol=atol)
    want = f"g{order}"
    found = [m for m in modes if m.label == want]
    if not found:
        raise RuntimeError(
            f"no {want} mode between {nu_min} and {nu_max} Hz. Found: "
            f"{[(m.label, round(m.nu_hz, 1)) for m in modes] or 'nothing'}. "
            "A star whose two sound speeds coincide has N^2 = 0 and supports "
            "no composition g-mode at all.")
    mode = found[0]
    if bg_complex is None:
        return mode
    return _refine_complex(bg_complex, mode, l=l, rtol=rtol, atol=atol)


def gmode_frequency(eos, cs2_eq, cs2_ad, e_c=None, M_target=1.4, l=2, order=1,
                    n_points=600, gamma=None, nu_min=30.0, nu_max=3500.0,
                    n_scan=220, **bg_kw):
    """Equation of state in, g-mode frequency out. The package's front door.

    eos      : `EOSTable_for_TOV` (P, epsilon in MeV/fm^3, nB in fm^-3), crust
               already attached — see `eos.gmode.background.with_crust`
    cs2_eq   : equilibrium c^2 = dP/deps, one per row of `eos`
    cs2_ad   : frozen c^2 = (dP/deps)_x, one per row of `eos`
    e_c      : central energy density [MeV/fm^3], or
    M_target : gravitational mass [M_sun] to solve for instead (default 1.4)
    order    : 1 for the fundamental g-mode
    gamma    : optional chemical equilibration rate [s^-1], scalar or one per
               row of `eos`. When given, the frozen sound speed is replaced by
               the complex dynamical one at the mode's own frequency and the
               result carries a bulk-viscous damping time. See
               `eos.gmode.rates.equilibration_rate`.

    Returns a `Mode`. The two sound speeds are the entire physics input: their
    difference is the buoyancy, and where they coincide there is no g-mode.
    """
    from eos.gmode.background import build_background

    bg = build_background(eos, cs2_eq, cs2_ad, e_c=e_c, M_target=M_target,
                          n_points=n_points, gamma=gamma, **bg_kw)
    mode = solve_gmode(bg, l=l, order=order, nu_min=nu_min, nu_max=nu_max,
                       n_scan=n_scan)
    if gamma is None:
        return mode

    # The dynamical sound speed depends on the frequency being solved for, so
    # iterate to self-consistency. Only the buoyancy has to be rebuilt each
    # pass -- the stellar structure does not depend on the sound speeds -- so
    # `at_frequency` is nearly free and the TOV equations are solved once.
    omega_s = 2.0 * np.pi * mode.nu_hz
    for _ in range(12):
        mode = _refine_complex(bg.at_frequency(omega_s), mode, l=l)
        new = 2.0 * np.pi * mode.nu_hz
        if abs(new - omega_s) < 1e-8 * abs(new):
            break
        omega_s = new
    return mode


def _refine_complex(bg, seed, l=2, rtol=1e-8, atol=1e-12):
    """Continue a real eigenfrequency into the complex plane.

    The surface discriminant is an analytic function of omega, so a secant
    iteration in the complex plane converges without ever needing a derivative
    or a two-dimensional solve -- and unlike a real 2-D root finder started on
    the real axis, it has no trouble stepping off it. The seed comes from the
    real problem, which is what pins down *which* mode is being continued:
    node counting does not survive complexification.
    """
    itp = bg.interpolators()

    def disc(w):
        # Raw, unnormalised: the secant needs an analytic function.
        return surface_discriminant(bg, w, l=l, itp=itp, dense=False,
                                    normalize=False, rtol=rtol, atol=atol)[0]

    seed_w = complex(np.real(seed.omega), 0.0)

    # Try progressively deeper starting offsets into the upper half plane: with
    # the e^{+i omega t} convention an amplitude goes as e^{-Im(omega) t}, so a
    # damped mode has Im(omega) > 0, and how far up depends on how strongly it
    # is damped.
    w = None
    for offset in (1e-3, 1e-2, 5e-2, 0.2, 0.5):
        w0 = seed_w
        w1 = seed_w * (1.0 + 1j * offset)
        d0, d1 = disc(w0), disc(w1)
        for _ in range(80):
            if d1 == d0:
                break
            w2 = w1 - d1 * (w1 - w0) / (d1 - d0)
            if not np.isfinite(w2):
                break
            w0, d0, w1 = w1, d1, w2
            d1 = disc(w1)
            if abs(w1 - w0) <= 1e-11 * abs(w1):
                w = w1
                break
        if w is not None:
            break

    if w is None or np.real(w) <= 0.0:
        raise RuntimeError(
            "no damped g-mode found near the undamped one at "
            f"{seed.nu_hz:.1f} Hz. This is a physical outcome as well as a "
            "possible numerical one: once the equilibration rate reaches the "
            "mode frequency, the composition re-equilibrates within a cycle, "
            "the buoyancy that supports the mode is destroyed and the g-mode "
            "ceases to exist. Check gamma/omega -- the mode survives while "
            "gamma is below roughly half of omega.")
    _d, r, U, V = surface_discriminant(bg, w, l=l, itp=itp,
                                       rtol=rtol, atol=atol)
    # With e^{+i omega t} an amplitude goes as e^{-Im(omega) t}, so a damped
    # mode has Im(omega) > 0 and its e-folding time is 1/Im(omega). Bulk
    # viscosity can only remove energy, so a root on the other side is not a
    # physical answer -- it means the search left the branch it was continuing.
    from eos.gmode.background import C_KM_S
    omega_i = float(np.imag(w)) * C_KM_S
    if omega_i < 0.0:
        raise RuntimeError(
            f"complex refinement returned a growing mode, Im(omega) = "
            f"{np.imag(w):.3e} km^-1. Dissipation cannot amplify, so this is a "
            "spurious branch rather than a physical instability.")
    tau = 1.0 / omega_i if omega_i > 0 else float("inf")
    xi_r = U / np.sqrt(itp["e_lam"](r)) / r**2
    return Mode(omega=w, nu_hz=float(omega_to_hz(np.real(w))), tau_s=tau,
                l=l, n_nodes=seed.n_nodes, label=seed.label,
                r=r, xi_r=xi_r, dp_over_h=V)


__all__ = ["Mode", "mode_spectrum", "solve_gmode", "integrate_mode",
           "surface_discriminant"]
