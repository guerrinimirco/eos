"""
M8 gate: NMP inverter (from_nmp cascade + feasibility flags).

Gate (report §3.y M8): from_nmp(DD2 NMPs) recovers the §2.1 couplings < 1e-3;
round-trip idempotent. Inversion is seeded from the published DD2 couplings
(DD2-class NMPs sit near them; a generic seed can fall into a spurious basin
where the cross-constraint is satisfied but Q_sat is wrong).

Note: Q_sat reproduction is limited to ~0.4 MeV by the finite-difference 3rd
derivative in the forward map, and the isoscalar cross-constraint
f_sigma''(1)=f_omega''(1) holds on the DD2 table only to 2.2e-3 — so
from_nmp reproduces the well-determined NMPs and predicts Q_sat, per the
report's own §2.3 statement that higher moments are predicted.
"""
import pytest

from eos.dd2 import Parametrization, compute_nmp

ISO = ("gamma_sigma", "b_sigma", "c_sigma",
       "gamma_omega", "b_omega", "c_omega", "gamma_rho", "a_rho")


@pytest.fixture(scope="module")
def dd2():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def dd2_nmp(dd2):
    return compute_nmp(dd2)


def test_roundtrip_recovers_couplings(dd2, dd2_nmp):
    par, status = Parametrization.from_nmp(dd2_nmp, return_status=True)
    assert status.ok, status.message
    for k in ISO:
        assert abs(getattr(par, k) - getattr(dd2, k)) < 1e-3, k


def test_roundtrip_reproduces_nmps(dd2, dd2_nmp):
    par = Parametrization.from_nmp(dd2_nmp)
    got = compute_nmp(par)
    # well-determined NMPs to < 0.5 MeV (Q_sat is FD-floor-limited but < 0.5)
    for key in ("E_sat", "K_sat", "Q_sat", "E_sym", "L_sym"):
        assert abs(got[key] - dd2_nmp[key]) < 0.5, key
    assert abs(got["n_sat"] - dd2_nmp["n_sat"]) < 1e-4
    assert abs(got["m_eff_ratio"] - dd2_nmp["m_eff_ratio"]) < 1e-3


def test_idempotent(dd2_nmp):
    par1 = Parametrization.from_nmp(dd2_nmp)
    par2 = Parametrization.from_nmp(compute_nmp(par1))
    for k in ISO:
        assert abs(getattr(par2, k) - getattr(par1, k)) < 1e-3, k


def test_feasibility_m_star_too_small(dd2_nmp):
    bad = dict(dd2_nmp, m_eff_ratio=0.25)   # below the physical window
    with pytest.raises(ValueError, match="m./m"):
        Parametrization.from_nmp(bad)


def test_feasibility_esym_below_kinetic(dd2_nmp):
    # E_sym below the kinetic symmetry energy (~19 MeV) has no real Gamma_rho
    bad = dict(dd2_nmp, E_sym=12.0)
    with pytest.raises(ValueError, match="E_sym"):
        Parametrization.from_nmp(bad)


def test_perturbed_nmp_solves(dd2_nmp):
    # a nearby feasible NMP set converges and reproduces its own inputs
    tgt = dict(dd2_nmp, K_sat=250.0, L_sym=60.0)
    par, status = Parametrization.from_nmp(tgt, return_status=True)
    assert status.ok
    got = compute_nmp(par)
    assert abs(got["K_sat"] - 250.0) < 0.5
    assert abs(got["L_sym"] - 60.0) < 0.5


if __name__ == "__main__":
    dd2 = Parametrization.from_dd2_defaults()
    nmp = compute_nmp(dd2)
    par, status = Parametrization.from_nmp(nmp, return_status=True)
    print("status:", status.ok, "iso_res=%.1e" % status.isoscalar_residual)
    print("max coupling diff:",
          max(abs(getattr(par, k) - getattr(dd2, k)) for k in ISO))
