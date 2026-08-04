"""
The two TOV backends must agree on a hybrid equation of state.

`mass_radius_mixed` defaults to the Numba backend, so this pins that default to
the readable scipy reference it is validated against. Both eta = 0 (a Gibbs
mixed phase, pressure rising through the window) and eta = 1 (a Maxwell
plateau with a genuine density jump) are checked: the plateau is the case where
the fast backend has to apply the Takatsy & Kovacs tidal correction across the
discontinuity explicitly, so it is the one that could plausibly break.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.mixed import (
    beta_eq_neutrinoless, build_mixed_eos_table, mass_radius_mixed,
)
from eos.vmit.parameters import get_vmit_default


@pytest.fixture(scope="module")
def setup():
    par = Parametrization.from_dd2_defaults()
    flags = SpeciesFlags(hyperons=False, phi_field=False, muons=False)
    grid = np.linspace(0.1 * par.n_sat, 12.0 * par.n_sat, 220)
    return par, flags, grid, get_vmit_default()


# Per-eta M_max tolerance [Msun]. These are the MEASURED differences, not
# round numbers: eta = 0 agrees to 1.4e-4, eta = 1 to 4.1e-3. The eta = 1 gap
# is a systematic in how the two backends treat the density discontinuity — it
# is flat in the central-density resolution (unchanged from n_ec = 100 to 400),
# so it is a real 0.2% difference across a Maxwell jump and not a grid artifact.
# The eta = 0 bound is kept tight deliberately: letting the smooth case inherit
# the plateau's tolerance would stop it detecting anything.
MMAX_TOL = {0.0: 1e-3, 1.0: 6e-3}


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_fast_matches_scipy(setup, eta):
    par, flags, grid, vmit = setup
    spec = beta_eq_neutrinoless()
    # Build the core EoS once; both backends must integrate the SAME table, or
    # the comparison would fold in solver noise that has nothing to do with the
    # integrator.
    core = build_mixed_eos_table(par, flags, grid, eta, spec,
                                 vmit_params=vmit, T=0.0)
    assert core.has_transition, "no transition: nothing to compare"

    kw = dict(vmit_params=vmit, T=0.0, table=core, n_ec=100, crust="No")
    fast = mass_radius_mixed(par, flags, grid, eta, spec, backend="fast", **kw)
    ref = mass_radius_mixed(par, flags, grid, eta, spec, backend="scipy", **kw)

    assert fast["M_max"] == pytest.approx(ref["M_max"], abs=MMAX_TOL[eta])
    assert fast["R_Mmax"] == pytest.approx(ref["R_Mmax"], abs=0.01)
    if np.isfinite(ref["R_1p4"]):
        assert fast["R_1p4"] == pytest.approx(ref["R_1p4"], abs=0.01)

    # Tidal deformability over the whole stable branch. Lambda spans orders of
    # magnitude, so it is compared relatively and only where both are positive.
    lam_f, lam_r = fast["results"][:, -1], ref["results"][:, -1]
    n = min(lam_f.size, lam_r.size)
    good = np.isfinite(lam_f[:n]) & np.isfinite(lam_r[:n]) & (lam_r[:n] > 0)
    assert good.any()
    rel = np.abs(lam_f[:n][good] - lam_r[:n][good]) / lam_r[:n][good]
    assert rel.max() < 0.05, f"Lambda differs by {rel.max():.1%}"


def test_default_backend_is_fast(setup):
    """The default must be the fast path — that is the point of the flip."""
    par, flags, grid, vmit = setup
    core = build_mixed_eos_table(par, flags, grid, 0.0, beta_eq_neutrinoless(),
                                 vmit_params=vmit, T=0.0)
    default = mass_radius_mixed(par, flags, grid, 0.0, beta_eq_neutrinoless(),
                                vmit_params=vmit, T=0.0, table=core, n_ec=60,
                                crust="No")
    explicit = mass_radius_mixed(par, flags, grid, 0.0, beta_eq_neutrinoless(),
                                 vmit_params=vmit, T=0.0, table=core, n_ec=60,
                                 crust="No", backend="fast")
    assert default["M_max"] == pytest.approx(explicit["M_max"], rel=1e-12)
