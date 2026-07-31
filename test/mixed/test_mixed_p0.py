"""
P0 gate (scaffolding): eos/mixed regime data structures + the two engine
adapters (docs/phase2/SPECIFICATION_AND_PLAN.md §3, milestone P0).

The gate: the adapters reproduce, to round-off, the values obtained by calling
eos/dd2 and eos/vmit directly at a set of golden points. Plus:
  - the four mode factories carry the spec §2 regime assignment;
  - ChargeSpec enforces its invariants (B GLOBAL; targets iff fixed);
  - MixedState pack/unpack round-trips for every mode's charge-potential layout;
  - adapter output is fm-based (no natural-unit leak, CLAUDE.md §3).

No equilibrium/eta physics is exercised here — that is P1+.
"""
import numpy as np
import pytest

from eos.dd2 import Parametrization, SpeciesFlags
from eos.dd2.solver import solve_octet
from eos.vmit.eos import solve_vmit_beta_eq
from eos.vmit.thermodynamics_quarks import compute_quark_matter_thermo_from_mu

from eos.mixed import (
    ChargeSpec, Regime, mode_A, mode_B, mode_C, mode_D,
    MixedState, charge_potential_slots, quark_charges, hadronic_charges,
    quark_phase, hadronic_phase,
)


# =============================================================================
# fixtures (module-scoped, built once — test/dd2 style)
# =============================================================================
@pytest.fixture(scope="module")
def par():
    return Parametrization.from_dd2_defaults()


@pytest.fixture(scope="module")
def par_y():
    return Parametrization.from_dd2y_defaults()


@pytest.fixture(scope="module")
def flags():
    return SpeciesFlags(hyperons=False)


@pytest.fixture(scope="module")
def flags_y():
    return SpeciesFlags(hyperons=True, phi_field=True)


# =============================================================================
# 1. QUARK adapter reproduces vMIT at golden points  (the gate, quark side)
# =============================================================================
@pytest.mark.parametrize("n_B, T", [(0.32, 50.0), (0.60, 0.0)])
def test_quark_adapter_reproduces_vmit(n_B, T):
    res = solve_vmit_beta_eq(n_B=n_B, T=T)          # realistic mu_u,mu_d,mu_s
    n_u, n_d, n_s, P, e, s, nB = compute_quark_matter_thermo_from_mu(
        res.mu_u, res.mu_d, res.mu_s, T)            # eos/vmit called directly
    th = quark_phase(res.mu_u, res.mu_d, res.mu_s, T=T)

    assert th.densities["u"] == pytest.approx(n_u, rel=1e-9)
    assert th.densities["d"] == pytest.approx(n_d, rel=1e-9)
    assert th.densities["s"] == pytest.approx(n_s, rel=1e-9)
    assert th.P == pytest.approx(P, rel=1e-9)
    assert th.eps == pytest.approx(e, rel=1e-9)
    assert th.s == pytest.approx(s, rel=1e-9, abs=1e-12)
    assert th.n_B == pytest.approx(nB, rel=1e-9)
    # conserved-charge densities from the QN table match the flavor densities
    qB, qC, qS = quark_charges(n_u, n_d, n_s)
    assert (th.n_B, th.n_C, th.n_S) == pytest.approx((qB, qC, qS), rel=1e-9)
    # decomposition mu_B = mu_u + 2 mu_d, mu_C = mu_u - mu_d, mu_S = mu_s - mu_d
    assert th.mu_B == pytest.approx(res.mu_u + 2 * res.mu_d, rel=1e-12)
    assert th.mu_C == pytest.approx(res.mu_u - res.mu_d, rel=1e-12)
    assert th.mu_S == pytest.approx(res.mu_s - res.mu_d, rel=1e-12)


# =============================================================================
# 2. HADRONIC adapter reproduces DD2 at golden points  (the gate, hadron side)
# =============================================================================
@pytest.mark.parametrize("n_B, T, Y_C", [(0.32, 0.0, 0.3), (0.32, 0.0, 0.1),
                                         (0.32, 50.0, 0.3)])
def test_hadronic_adapter_reproduces_dd2_nucleons(par, flags, n_B, T, Y_C):
    # eos/dd2 called directly: leptonless fixed-Y_C octet solve (pure hadronic).
    # include_photons=False: the adapter is matter-only, so is this reference.
    pt = solve_octet(par, n_B, flags, T=T, charge_mode="fixed", Y_C=Y_C,
                     include_photons=False)
    mu_tilde_B = pt.mu_n - pt.Sigma_R                # octet kinetic baryon pot.
    mu_Q = pt.mu_p - pt.mu_n
    th = hadronic_phase(par, flags, mu_tilde_B, mu_Q, 0.0, T=T,
                        n_B_guess=0.9 * n_B)         # generic guess, not the answer

    assert th.n_B == pytest.approx(n_B, rel=1e-9)
    assert th.P == pytest.approx(pt.P, rel=1e-9)
    assert th.eps == pytest.approx(pt.eps, rel=1e-9)
    assert th.s == pytest.approx(pt.s, rel=1e-9, abs=1e-12)
    assert th.densities["p"] == pytest.approx(pt.n_p, rel=1e-9)
    assert th.densities["n"] == pytest.approx(pt.n_n, rel=1e-9)
    assert th.mu_B == pytest.approx(pt.mu_n, rel=1e-9)          # mu_B = mu_n
    assert th.mu_i["p"] == pytest.approx(pt.mu_p, rel=1e-9)     # decomposition
    # non-leptonic charge fraction reproduced
    assert th.n_C / th.n_B == pytest.approx(Y_C, rel=1e-9)


def test_hadronic_adapter_reproduces_dd2_hyperons(par_y, flags_y):
    n_B, Y_C = 0.6, 0.1                              # above Lambda onset
    pt = solve_octet(par_y, n_B, flags_y, T=0.0, charge_mode="fixed", Y_C=Y_C,
                     include_photons=False)
    assert pt.Y("Lambda") > 0.0                      # golden point is genuinely strange
    mu_tilde_B = pt.mu_n - pt.Sigma_R
    mu_Q = pt.mu_p - pt.mu_n
    th = hadronic_phase(par_y, flags_y, mu_tilde_B, mu_Q, pt.mu_S, T=0.0,
                        n_B_guess=0.5)
    _, _, n_S_ref = hadronic_charges(flags_y, pt.composition_map)

    assert th.n_B == pytest.approx(n_B, rel=1e-9)
    assert th.P == pytest.approx(pt.P, rel=1e-9)
    assert th.eps == pytest.approx(pt.eps, rel=1e-9)
    assert th.densities["Lambda"] == pytest.approx(pt.Y("Lambda") * n_B, rel=1e-9)
    assert th.n_S == pytest.approx(n_S_ref, rel=1e-9)


# =============================================================================
# 3. Regime assignment: the four modes carry the spec §2 table
# =============================================================================
def test_modes_regime_assignment():
    a = mode_A()
    assert (a.B, a.C, a.S, a.L_e) == (Regime.GLOBAL, Regime.NOT_CONSERVED,
                                      Regime.NOT_CONSERVED, Regime.NOT_CONSERVED)
    b = mode_B(Y_L=0.4)
    assert b.L_e is Regime.GLOBAL and b.targets["Y_L"] == 0.4
    assert b.C is Regime.NOT_CONSERVED and b.S is Regime.NOT_CONSERVED
    c = mode_C(Y_C=0.3)
    assert c.C is Regime.GLOBAL and c.targets["Y_C"] == 0.3
    assert c.S is Regime.NOT_CONSERVED and c.L_e is Regime.NOT_CONSERVED
    d = mode_D(Y_C=0.3, Y_S=0.1)
    assert d.C is Regime.GLOBAL and d.S is Regime.GLOBAL      # D-global default
    assert d.targets["Y_C"] == 0.3 and d.targets["Y_S"] == 0.1


def test_chargespec_invariants():
    with pytest.raises(ValueError):                 # B must be GLOBAL
        ChargeSpec(B=Regime.NOT_CONSERVED)
    with pytest.raises(ValueError):                 # GLOBAL C needs a Y_C target
        ChargeSpec(C=Regime.GLOBAL)
    with pytest.raises(ValueError):                 # NOT_CONSERVED must not carry one
        ChargeSpec(targets={"Y_C": 0.3})
    with pytest.raises(ValueError):                 # yc_leptons needs C GLOBAL
        ChargeSpec(yc_leptons=True)
    # an unnamed combination is constructible directly (spec §1.5)
    spec = ChargeSpec(S=Regime.LOCAL, targets={"Y_S": 0.1})
    assert spec.S is Regime.LOCAL


# =============================================================================
# 4. MixedState owns the layout: pack/unpack round-trips
# =============================================================================
def test_charge_potential_slots():
    assert charge_potential_slots(mode_A()) == ("chi", "mu_B")
    assert charge_potential_slots(mode_C(0.3)) == ("chi", "mu_B", "mu_C")
    assert charge_potential_slots(mode_D(0.3, 0.1)) == (
        "chi", "mu_B", "mu_S", "mu_C")
    # LOCAL charge -> per-phase pair
    assert charge_potential_slots(
        ChargeSpec(S=Regime.LOCAL, targets={"Y_S": 0.1})) == (
        "chi", "mu_B", "mu_S_H", "mu_S_Q")


@pytest.mark.parametrize("spec", [mode_A(), mode_B(0.4), mode_C(0.3),
                                  mode_D(0.3, 0.1)])
def test_mixedstate_roundtrip(spec):
    st = MixedState.for_charges(spec)
    x = np.arange(len(st), dtype=float) + 1.0
    assert st.pack(st.unpack(x)) == pytest.approx(x)
    d = {name: float(i) for i, name in enumerate(st.slots)}
    assert st.unpack(st.pack(d)) == d
    with pytest.raises(ValueError):                 # length mismatch is caught
        st.unpack(np.zeros(len(st) + 1))


# =============================================================================
# 5. QN helpers + unit-boundary sanity
# =============================================================================
def test_hadronic_charge_helper(par, flags):
    pt = solve_octet(par, 0.32, flags, T=0.0, charge_mode="fixed", Y_C=0.25)
    dens = {"n": pt.n_n, "p": pt.n_p}
    n_B, n_C, n_S = hadronic_charges(flags, dens)
    assert n_B == pytest.approx(0.32, rel=1e-9)
    assert n_C == pytest.approx(pt.n_p, rel=1e-9)    # C = proton charge (nucleons)
    assert n_S == pytest.approx(0.0, abs=1e-15)


def test_adapter_output_is_fm_based(par, flags):
    # fm-based densities are O(0.1-1), not natural units O(1e6): catches a hc3 slip
    pt = solve_octet(par, 0.32, flags, T=0.0, charge_mode="fixed", Y_C=0.3)
    th = hadronic_phase(par, flags, pt.mu_n - pt.Sigma_R, pt.mu_p - pt.mu_n,
                        0.0, T=0.0, n_B_guess=0.28)
    assert 0.05 < th.n_B < 5.0
    assert 0.0 < th.P < 500.0
    q = quark_phase(*(solve_vmit_beta_eq(0.5, 0.0).__dict__[k]
                      for k in ("mu_u", "mu_d", "mu_s")), T=0.0)
    assert 0.05 < q.n_B < 5.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
