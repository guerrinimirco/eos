"""
M7 gate: thermal meson gas (pseudoscalar + vector nonets).

Gate: T->0 contribution -> 0; HVH still holds with the meson
terms. The mesons are an additive ideal Bose gas on top of the mean field with
effective chemical potentials tied to the DD2 density-dependent couplings; the
Bose-gas Euler relation e + P = T s + mu* n makes mu*_j n_j the right HVH term.

Note: the thermal mesons are added post-baryonic-solve and NOT fed back into
charge neutrality (valid in the dilute thermal regime; off by default for cold
NS). Their net charge/strangeness densities are exposed for later
self-consistent coupling.
"""
import pytest

from eos.dd2 import Parametrization, SpeciesFlags, solve_octet
from eos.dd2.physics.mesons import thermal_meson_thermo


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


def _flags(ps=False, tv=False):
    return SpeciesFlags(hyperons=False, phi_field=False,
                        include_pseudoscalars=ps, include_thermal_vectors=tv)


def test_flags_no_longer_raise():
    SpeciesFlags(include_pseudoscalars=True)      # wired at M7 -> no raise
    SpeciesFlags(include_thermal_vectors=True)


def test_t0_contribution_vanishes(par):
    base = solve_octet(par, 0.16, _flags())
    for T in (0.0, 1.0):
        m = solve_octet(par, 0.16, _flags(ps=True, tv=True), T=T)
        # cold: massive bosons exponentially suppressed -> identical thermo
        assert m.P == pytest.approx(base.P if T == 0.0
                                    else solve_octet(par, 0.16, _flags(), T=T).P,
                                    rel=1e-9)


def test_hvh_holds_with_mesons(par):
    for T in (5.0, 20.0, 50.0):
        m = solve_octet(par, 0.16, _flags(ps=True, tv=True), T=T)
        assert abs(m.hvh_rel) < 1e-11


def test_contribution_grows_with_T(par):
    dPs = []
    for T in (10.0, 30.0, 50.0):
        base = solve_octet(par, 0.16, _flags(), T=T)
        both = solve_octet(par, 0.16, _flags(ps=True, tv=True), T=T)
        dPs.append(both.P - base.P)
    assert all(d > 0 for d in dPs)
    assert dPs[0] < dPs[1] < dPs[2]


def test_pseudoscalar_and_vector_additive(par):
    # Measure contributions on the meson function directly (a P-difference on
    # the full solve loses the tiny heavy-vector term to float cancellation).
    T = 60.0
    args = dict(mu_Q=20.0, mu_S=0.0, omega0=25.0, rho0=-5.0, T=T)
    ps = thermal_meson_thermo(par, 0.16, include_pseudoscalars=True, **args)
    tv = thermal_meson_thermo(par, 0.16, include_thermal_vectors=True, **args)
    both = thermal_meson_thermo(par, 0.16, include_pseudoscalars=True,
                                include_thermal_vectors=True, **args)
    assert both["P"] == pytest.approx(ps["P"] + tv["P"], rel=1e-12)
    assert ps["P"] > 0 and tv["P"] > 0
    # vectors (heavier: m_rho=771 vs m_pi=140) contribute less than pseudoscalars
    assert tv["P"] < ps["P"]


def test_meson_charge_strangeness(par):
    # In symmetric-ish matter mu_Q small, meson charge is near zero; give it a
    # nonzero mu_Q and check the net charge tracks pi+/pi- imbalance.
    mg = thermal_meson_thermo(par, 0.16, mu_Q=40.0, mu_S=0.0,
                              omega0=25.0, rho0=-5.0, T=40.0,
                              include_pseudoscalars=True)
    assert mg["n_C"] > 0.0          # mu_Q>0 favors positive mesons
    # Euler relation bookkeeping is finite and used by HVH
    assert mg["mu_dot_n"] != 0.0


def test_meson_gas_off_at_T0_direct(par):
    mg = thermal_meson_thermo(par, 0.16, 40.0, 0.0, 25.0, -5.0, T=0.0,
                              include_pseudoscalars=True,
                              include_thermal_vectors=True)
    assert (mg["P"], mg["e"], mg["s"], mg["n_C"]) == (0.0, 0.0, 0.0, 0.0)


if __name__ == "__main__":
    par = Parametrization.from_dd2_defaults()
    for T in (1.0, 10.0, 30.0, 50.0):
        base = solve_octet(par, 0.16, _flags(), T=T)
        both = solve_octet(par, 0.16, _flags(ps=True, tv=True), T=T)
        print(f"T={T:5.1f}: dP={both.P - base.P:.3e} de={both.eps - base.eps:.3e}"
              f"  HVH={both.hvh_rel:.1e}")
