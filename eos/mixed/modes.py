"""
mixed/modes.py
==============
The four named equilibrium modes as `ChargeSpec` factories
(docs/phase2/SPECIFICATION_AND_PLAN.md §2; STEP0_AUDIT.md §2).

| mode | fixed inputs        | B      | C            | S            | L_e    |
|------|---------------------|--------|--------------|--------------|--------|
| A    | (n_B, T, eta)       | global | not-cons.    | not-cons.    | not-cons.|
| B    | (n_B, Y_L, T, eta)  | global | not-cons.    | not-cons.    | global |
| C    | (n_B, Y_C, T, eta)  | global | global       | not-cons.    | not-cons.|
| D    | (n_B, Y_C, Y_S, T)  | global | global       | global       | not-cons.|

These are conveniences: every one is just a regime choice per charge, and an
unnamed combination is constructible by instantiating `ChargeSpec` directly
(spec §1.5). The point of the engine is that all four are configurations of one
solver, not four solvers.
"""
from eos.mixed.charges import ChargeSpec, Regime


def mode_A():
    """Beta-equilibrium: C, S, L_e all not conserved (spec §2)."""
    return ChargeSpec()


def mode_B(Y_L):
    """Beta-equilibrium with trapped neutrinos at fixed total Y_L (spec §2).

    L_e is GLOBAL: trapped neutrinos are treated as purely global,
    mu_nu^H = mu_nu^Q, with NO local component and no eta weighting — a stated
    modelling assumption, not a thesis result (spec §1.7, audit §4). Callers
    should surface it in output metadata.
    """
    return ChargeSpec(L_e=Regime.GLOBAL, targets={"Y_L": Y_L})


def mode_C(Y_C, *, yc_leptons=False):
    """Fixed non-leptonic charge fraction Y_C (spec §2).

    `yc_leptons` picks the §1.6 flavor: False = leptonless (CompOSE
    (n_B, T, Y_q) convention), True = neutralizing leptons present.
    """
    return ChargeSpec(C=Regime.GLOBAL, targets={"Y_C": Y_C},
                      yc_leptons=yc_leptons)


def mode_D(Y_C, Y_S, *, yc_leptons=False):
    """Fixed Y_C and Y_S, with strangeness GLOBAL (D-global, spec §2).

    D-global is the default per audit §2: strangeness is globally conserved
    (mu_S^H = mu_S^Q), consistent with how Y_C is treated and with the
    tabulated-EoS convention. The per-phase reading (D-local, the combined
    (A.93) baryon relation) is deferred behind a future flag driven by the same
    regime mechanism — set S=Regime.LOCAL on a ChargeSpec once it is wired.
    `yc_leptons` selects the §1.6 charge flavor exactly as mode_C.
    """
    return ChargeSpec(C=Regime.GLOBAL, S=Regime.GLOBAL,
                      targets={"Y_C": Y_C, "Y_S": Y_S}, yc_leptons=yc_leptons)
