"""g-mode eigenfrequencies: the null test, the spectrum, and the numbers.

The literature values quoted below are the l = 2 core g-mode frequencies
collected in Table I of Jaikumar, Semposki, Prakash and Constantinou,
Phys. Rev. D 103, 123009 (2021), for cold non-superfluid nucleonic stars of
1.4 M_sun:

    Reisenegger and Goldreich (1992)   npe      0.215 kHz  (at 1.405 M_sun)
    Kantor and Gusakov (2014)          npe      0.13  kHz
    Kantor and Gusakov (2014)          npemu    0.19  kHz
    Yu and Weinberg (2017)             npemu    0.13  kHz
    Jaikumar et al. (2021), ZL         npe      0.24  kHz
    Jaikumar et al. (2021), ZL         npemu    0.27  kHz

The spread is a factor of two and is genuine: the g-mode measures the
composition gradient, which is exactly where these equations of state differ.
The test therefore asserts the band, not a single value -- a solver bug would
put the answer orders of magnitude out, not tens of per cent.
"""
import numpy as np
import pytest

from eos.gmode.background import build_background
from eos.gmode.cowling import mode_spectrum, solve_gmode, gmode_frequency


def test_no_composition_gradient_means_no_gmode(polytrope):
    """The null test: one sound speed, so N^2 = 0 and the g-branch is empty.

    This is the sharpest check that the solver is finding buoyancy modes rather
    than numerical artefacts. Only the f-mode should survive.
    """
    eos, cs2 = polytrope
    bg = build_background(eos, cs2, cs2, M_target=1.4, n_points=400)
    modes = mode_spectrum(bg, nu_min=60.0, nu_max=3000.0, n_scan=90)

    assert [m.label for m in modes] == ["f"]
    assert modes[0].n_nodes == 0
    with pytest.raises(RuntimeError, match="no g1 mode"):
        solve_gmode(bg, nu_min=60.0, nu_max=3000.0, n_scan=90)


def test_fmode_of_a_14_msun_star_is_around_2_khz(polytrope):
    """A sanity anchor on the whole eigenvalue machinery.

    The l = 2 f-mode of a 1.4 M_sun neutron star lies near 2 kHz, and the
    Cowling approximation is known to overestimate it by 10-20 per cent, so a
    result in the 1.8-2.7 kHz band is what a correct solver gives.
    """
    eos, cs2 = polytrope
    bg = build_background(eos, cs2, cs2, M_target=1.4, n_points=400)
    f = mode_spectrum(bg, nu_min=60.0, nu_max=3000.0, n_scan=90)[0]
    assert f.label == "f"
    assert 1800.0 < f.nu_hz < 2700.0


def test_dd2_gmode_is_in_the_published_band(dd2_eos):
    """DD2 npemu at 1.4 M_sun against the Table I spread of 0.13-0.27 kHz."""
    eos, ceq, cad, _n, _y = dd2_eos
    mode = gmode_frequency(eos, ceq, cad, M_target=1.4, n_points=500,
                           nu_min=60.0, nu_max=3000.0, n_scan=90)

    assert mode.label == "g1"
    assert mode.n_nodes == 1, "the fundamental g-mode has one node"
    assert 100.0 < mode.nu_hz < 350.0, f"g1 = {mode.nu_hz:.1f} Hz"
    assert mode.tau_s is None, "a real background carries no damping time"


def test_gmode_spectrum_is_ordered_and_below_the_fmode(dd2_eos):
    """g1 > g2 > g3 ... and the whole g-branch sits below the f-mode."""
    eos, ceq, cad, _n, _y = dd2_eos
    bg = build_background(eos, ceq, cad, M_target=1.4, n_points=500)
    modes = mode_spectrum(bg, nu_min=60.0, nu_max=3000.0, n_scan=90)

    gmodes = [m for m in modes if m.is_gmode]
    fmode = [m for m in modes if m.label == "f"]
    assert len(gmodes) >= 2 and len(fmode) == 1

    by_order = sorted(gmodes, key=lambda m: int(m.label[1:]))
    freqs = [m.nu_hz for m in by_order]
    assert freqs == sorted(freqs, reverse=True), "g1 must be the highest"
    assert max(freqs) < fmode[0].nu_hz


def test_eigenfunction_is_regular_and_has_the_right_node_count(dd2_eos):
    """xi_r ~ r^{l-1} at the centre and has exactly one node for g1."""
    eos, ceq, cad, _n, _y = dd2_eos
    bg = build_background(eos, ceq, cad, M_target=1.4, n_points=500)
    g1 = solve_gmode(bg, nu_min=60.0, nu_max=3000.0, n_scan=90)

    assert np.all(np.isfinite(g1.xi_r))
    assert abs(g1.xi_r[0]) < 1e-3 * np.max(np.abs(g1.xi_r))
    interior = g1.xi_r[1:-1]
    big = interior[np.abs(interior) > 1e-6 * np.max(np.abs(interior))]
    assert np.count_nonzero(np.diff(np.sign(big)) != 0) == 1


def test_gmode_frequency_scales_with_buoyancy(polytrope):
    """Doubling c_ad^2 - c_eq^2 raises the g-mode, roughly as sqrt.

    The g-mode frequency is bounded by and proportional to N, which scales as
    the square root of the sound-speed difference. The check is loose because
    the eigenvalue is a weighted average of N over the star, not its peak.
    """
    eos, cs2 = polytrope
    out = []
    for delta in (0.001, 0.004):
        bg = build_background(eos, cs2, cs2 + delta, M_target=1.4,
                              n_points=400)
        out.append(solve_gmode(bg, nu_min=40.0, nu_max=2000.0,
                               n_scan=90).nu_hz)
    ratio = out[1] / out[0]
    assert 1.5 < ratio < 2.5, f"ratio {ratio:.2f}, expected near sqrt(4) = 2"


@pytest.mark.slow
def test_finite_reaction_rate_damps_and_lowers_the_mode(dd2_eos):
    """A finite gamma makes the eigenfrequency complex.

    When the composition partly re-equilibrates within a cycle the buoyancy is
    reduced, so the real frequency drops, and the reaction lagging the
    compression dissipates energy, so a finite damping time appears.
    """
    eos, ceq, cad, _n, _y = dd2_eos
    kw = dict(M_target=1.4, n_points=400, nu_min=40.0, nu_max=3000.0,
              n_scan=120)
    cold = gmode_frequency(eos, ceq, cad, **kw)
    warm = gmode_frequency(eos, ceq, cad, gamma=0.3 * 2 * np.pi * cold.nu_hz,
                           **kw)

    assert cold.tau_s is None, "a real background carries no damping time"
    assert warm.tau_s is not None and np.isfinite(warm.tau_s)
    assert warm.tau_s > 0.0
    assert np.imag(warm.omega) > 0.0, "with e^{+i omega t}, decay means Im > 0"
    assert warm.nu_hz < cold.nu_hz, "partial equilibration weakens buoyancy"


@pytest.mark.slow
def test_damping_time_scales_inversely_with_the_reaction_rate(dd2_eos):
    """For gamma << omega the dissipation is proportional to gamma.

    In that regime Im[c_dy^2] -> (c_ad^2 - c_eq^2) gamma/omega, linear in
    gamma, so the mode's damping rate is too and tau ~ 1/gamma. A decade in
    gamma must buy a decade in tau; the frozen frequency is untouched.
    """
    eos, ceq, cad, _n, _y = dd2_eos
    kw = dict(M_target=1.4, n_points=400, nu_min=40.0, nu_max=3000.0,
              n_scan=120)
    cold = gmode_frequency(eos, ceq, cad, **kw)
    omega = 2.0 * np.pi * cold.nu_hz

    slow_g = gmode_frequency(eos, ceq, cad, gamma=1e-3 * omega, **kw)
    fast_g = gmode_frequency(eos, ceq, cad, gamma=1e-2 * omega, **kw)

    assert slow_g.tau_s / fast_g.tau_s == pytest.approx(10.0, rel=0.05)
    # deep in the frozen limit the real frequency is the undamped one
    assert slow_g.nu_hz == pytest.approx(cold.nu_hz, rel=1e-4)


@pytest.mark.slow
def test_fast_equilibration_suppresses_the_mode(dd2_eos):
    """Faster equilibration destroys the buoyancy, and with it the mode.

    The suppression is progressive rather than abrupt. As gamma/omega rises the
    quality factor Q = Re(omega) / 2 Im(omega) falls monotonically, passing
    through Q ~ 1 -- critical damping, where the mode is no longer an
    oscillation -- at around gamma = omega, which is also where the dissipation
    peaks. Well beyond that the root ceases to exist at all and the solver says
    so. This is the suppression of composition g-modes in warm neutron stars.
    """
    eos, ceq, cad, _n, _y = dd2_eos
    kw = dict(M_target=1.4, n_points=400, nu_min=60.0, nu_max=3000.0,
              n_scan=90)
    cold = gmode_frequency(eos, ceq, cad, **kw)
    omega = 2.0 * np.pi * cold.nu_hz

    def quality(frac):
        m = gmode_frequency(eos, ceq, cad, gamma=frac * omega, **kw)
        return np.real(m.omega) / (2.0 * np.imag(m.omega)), m.nu_hz

    Q_weak, nu_weak = quality(0.1)
    Q_res, nu_res = quality(1.0)

    assert Q_weak > 3.0, "a slow reaction should barely damp the mode"
    assert Q_res < 1.5, "at gamma ~ omega the mode is near critically damped"
    assert Q_res < Q_weak
    assert nu_res < nu_weak < cold.nu_hz, "buoyancy weakens as gamma grows"

    # Far into the fast-equilibration regime there is no g-mode left to find.
    with pytest.raises(RuntimeError, match="no damped g-mode"):
        gmode_frequency(eos, ceq, cad, gamma=10.0 * omega, **kw)
