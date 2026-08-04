"""
Species flags that were previously accepted and ignored: muons and the thermal
meson gas.

The engine must never silently drop a degree of freedom the caller switched on
(CLAUDE.md §5). These tests pin down that:

  - muons join every neutrality condition the eta split creates — locally in
    each phase and globally on the average — rather than only the electrons;
  - muons sit at mu_mu = mu_e - mu_L, the same relation eos/dd2 uses, so the
    two engines describe the same matter;
  - switching muons on genuinely changes the state (a test that only checked
    "it still runs" would have passed against the old, silently-electron-only
    code);
  - switching muons off reproduces the electron-only answer exactly, so the
    wiring costs nothing when it is not wanted;
  - the thermal meson gas contributes to the hadronic phase pressure, which is
    what mechanical equilibrium balances, and so shifts the transition;
  - Euler / Hugenholtz-Van Hove still holds with either sector on, which is the
    real check that the new terms were added to *every* place they belong and
    not just to the residual.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.general.thermodynamics_leptons import electron_thermo, muon_thermo
from eos.vmit.parameters import get_vmit_custom
from eos.mixed import beta_eq_neutrinoless, beta_eq_neutrino_trapped, solve_mixed
from eos.mixed.equilibrium.residual import charged_leptons, mixed_slots


@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


NO_MUONS = SpeciesFlags(hyperons=False, muons=False)
WITH_MUONS = SpeciesFlags(hyperons=False, muons=True)
N_B = 0.7                                    # inside the window at every eta


# =============================================================================
# 1. muons enter the neutrality conditions
# =============================================================================
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_neutrality_counts_muons(par, eta):
    """Every active neutrality domain balances the non-leptonic charge against
    electrons AND muons, not electrons alone."""
    r = solve_mixed(par, WITH_MUONS, N_B, eta, beta_eq_neutrinoless(), T=0.0)
    L_H, L_Q, G = r.extras["L_H"], r.extras["L_Q"], r.extras["G"]
    if eta > 0.0:                            # local neutrality, per phase
        assert r.th_H.n_C - L_H.n == pytest.approx(0.0, abs=1e-10)
        assert r.th_Q.n_C - L_Q.n == pytest.approx(0.0, abs=1e-10)
    if eta < 1.0:                            # global neutrality, on the average
        avg = (1 - r.chi) * r.th_H.n_C + r.chi * r.th_Q.n_C
        assert avg - G.n == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("eta", [0.5, 1.0])
def test_muons_are_actually_populated(par, eta):
    """The muon density is nonzero at this density — otherwise the neutrality
    test above would pass vacuously."""
    r = solve_mixed(par, WITH_MUONS, N_B, eta, beta_eq_neutrinoless(), T=0.0)
    assert r.extras["L_H"].n_mu > 1e-4
    assert r.extras["L_H"].n_mu < r.extras["L_H"].n_e     # and subdominant


# =============================================================================
# 2. mu_mu = mu_e - mu_L, matching eos/dd2
# =============================================================================
def test_muon_potential_follows_the_electron():
    """Neutrino-transparent matter puts the muon at the electron potential;
    trapped neutrinos shift it by mu_L, exactly as eos/dd2/physics/octet.py
    does (mu_e = mu_L - mu_Q while mu_mu = -mu_Q)."""
    mu_e, T = 180.0, 0.0
    dom = charged_leptons(mu_e, T, muons=True, mu_L=0.0)
    assert dom.n_e == pytest.approx(electron_thermo(mu_e, T).n)
    assert dom.n_mu == pytest.approx(muon_thermo(mu_e, T).n)

    mu_L = 40.0
    shifted = charged_leptons(mu_e, T, muons=True, mu_L=mu_L)
    assert shifted.n_mu == pytest.approx(muon_thermo(mu_e - mu_L, T).n)
    assert shifted.n_e == pytest.approx(electron_thermo(mu_e, T).n)


def test_muons_add_no_unknowns(par):
    """Because mu_mu is tied to mu_e, muons cost no extra slots — the unknown
    vector is the same size with and without them."""
    for eta in (0.0, 0.5, 1.0):
        spec = beta_eq_neutrinoless()
        assert (mixed_slots(spec, eta, NO_MUONS)
                == mixed_slots(spec, eta, WITH_MUONS))


# =============================================================================
# 3. switching the sector on changes the answer; off reproduces it exactly
# =============================================================================
@pytest.mark.parametrize("eta", [0.5, 1.0])
def test_muons_shift_the_state(par, eta):
    """Muons take some of the neutralizing charge from the electrons, softening
    the lepton pressure and moving the quark fraction. If this ever passes with
    equality, the muon sector has been silently dropped again."""
    without = solve_mixed(par, NO_MUONS, N_B, eta, beta_eq_neutrinoless(), T=0.0)
    with_mu = solve_mixed(par, WITH_MUONS, N_B, eta, beta_eq_neutrinoless(), T=0.0)
    assert abs(with_mu.chi - without.chi) > 1e-3
    assert with_mu.extras["L_H"].n_mu > 0.0


def test_muons_off_is_electron_only(par):
    """With muons off the lepton block is exactly the electron block."""
    r = solve_mixed(par, NO_MUONS, N_B, 0.5, beta_eq_neutrinoless(), T=0.0)
    for key in ("L_H", "L_Q", "G"):
        dom = r.extras[key]
        assert dom.n_mu == 0.0
        assert dom.n == dom.n_e


# =============================================================================
# 4. the thermal meson gas reaches the phase pressure
# =============================================================================
def test_meson_gas_enters_the_hadronic_pressure(par):
    """The meson gas is an additive Bose gas on the hadronic phase, so it raises
    P_H — which is what mechanical equilibrium balances against P_Q, and so it
    moves the transition. At T=0 it must contribute nothing at all."""
    plain = SpeciesFlags(hyperons=False, muons=False)
    mesons = SpeciesFlags(hyperons=False, muons=False,
                          include_pseudoscalars=True)
    hot_plain = solve_mixed(par, plain, N_B, 0.0, beta_eq_neutrinoless(), T=30.0)
    hot_meson = solve_mixed(par, mesons, N_B, 0.0, beta_eq_neutrinoless(), T=30.0)
    assert hot_meson.th_H.P > hot_plain.th_H.P
    assert abs(hot_meson.chi - hot_plain.chi) > 1e-6

    cold_plain = solve_mixed(par, plain, N_B, 0.0, beta_eq_neutrinoless(), T=0.0)
    cold_meson = solve_mixed(par, mesons, N_B, 0.0, beta_eq_neutrinoless(), T=0.0)
    assert cold_meson.th_H.P == pytest.approx(cold_plain.th_H.P, rel=1e-12)


# =============================================================================
# 5. thermodynamic consistency survives both sectors
# =============================================================================
@pytest.mark.parametrize("flags", [
    WITH_MUONS,
    SpeciesFlags(hyperons=False, muons=True, include_pseudoscalars=True),
])
@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_euler_holds_with_the_new_sectors(par, flags, eta):
    """eps + P = T s + sum_i mu_i n_i. solve_mixed asserts this itself, so a
    missing muon or meson term anywhere in the totals raises here."""
    r = solve_mixed(par, flags, N_B, eta, beta_eq_neutrinoless(), T=20.0,
                    check_consistency=True)
    assert r.in_mixed_phase                  # otherwise the check is vacuous
    assert np.isfinite(r.P) and np.isfinite(r.eps) and r.s > 0.0


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_euler_holds_with_hyperons_and_muons(eta):
    """Same check with the strange sector open. The transition sits elsewhere
    for the softer hyperonic parametrization, so the density is located rather
    than assumed — a point outside the window would make the consistency gate
    (which only fires on a genuine mixed phase) pass vacuously.
    """
    from eos.mixed import locate_window
    par_y = Parametrization.from_dd2y_defaults()
    flags_y = SpeciesFlags(hyperons=True, muons=True, phi_field=True)
    vp = get_vmit_custom(B4=160.0)           # puts the transition in reach
    grid = np.linspace(0.3, 1.3, 40)
    win = locate_window(par_y, flags_y, grid, eta, beta_eq_neutrinoless(),
                        vmit_params=vp, T=20.0)
    assert win.exists, "no quark transition found for these parameters"
    n_mid = 0.5 * (win.n_onset + win.n_offset)
    r = solve_mixed(par_y, flags_y, n_mid, eta, beta_eq_neutrinoless(),
                    vmit_params=vp, T=20.0, check_consistency=True)
    assert r.in_mixed_phase
    assert r.th_H.n_S != 0.0                 # hyperons really are populated
    assert r.extras["L_H"].n_mu > 0.0 or eta == 0.0


def test_euler_holds_with_trapped_neutrinos_and_muons(par):
    """The trapped-neutrino mode is where the muon relation stops being
    mu_mu = mu_e, so it is the case most likely to break the energy budget."""
    r = solve_mixed(par, WITH_MUONS, N_B, 0.0, beta_eq_neutrino_trapped(0.3),
                    T=20.0, check_consistency=True)
    assert r.extras["nu"].n > 0.0
    assert r.potentials["mu_L"] > 0.0
