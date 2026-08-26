"""Parameters of the Zhao-Lattimer nucleonic density functional.

The functional carries its interaction in two terms, each with a linear and a
power-law piece (see `zl.tex` for the full description):

    V(n_p, n_n) = 4 n_p n_n [a0/n0 + b0/n0 u^(gamma-1)]
                + (n_n - n_p)^2 [a1/n0 + b1/n0 u^(gamma1-1)],   u = n_B/n0

so six numbers set the six lowest nuclear-matter parameters almost
independently. There is no scalar field and hence no effective mass: the
nucleon masses are parameters and enter the Fermi integrals unchanged.

`n0` is a reference density OF THE FUNCTIONAL, not the saturation density it
predicts -- the P = 0 root of cold symmetric matter is 0.15951 fm^-3, 0.3%
below it.

Every number here is an argument, never a module-level constant: inference
varies them across millions of calls.

Reference: T. Zhao and J. M. Lattimer, Phys. Rev. D 102, 023021 (2020); the
shipped set is the one used by C. Constantinou et al., Phys. Rev. D 104,
123032 (2021), Phys. Rev. D 107, 074013 (2023) and arXiv:2506.20418 (2025).
"""
from dataclasses import dataclass


@dataclass
class Parameters:
    """The eight numbers a ZL calculation needs.

    Attributes:
        name:   parameter-set identifier, used in generated file names
        m_p:    proton mass (MeV)
        m_n:    neutron mass (MeV) -- equal to m_p in the published set, since
                the kinetic term carries no isospin splitting
        n0:     reference density of the functional (fm^-3)
        a0:     proton-neutron cross term, linear in u (MeV)
        b0:     proton-neutron cross term, coefficient of u^gamma (MeV)
        gamma:  exponent of the cross term
        a1:     isovector term, linear in u (MeV)
        b1:     isovector term, coefficient of u^gamma1 (MeV)
        gamma1: exponent of the isovector term

    Both brackets of the functional enter the symmetry energy: the a0, b0 term
    is a proton-neutron cross interaction which switches off as matter becomes
    pure, so the potential part of E_sym is
    (a1 - a0) u + b1 u^gamma1 - b0 u^gamma, not a1 + b1.
    """
    name: str = "ZL_Constantinou"
    m_p: float = 939.5     # MeV
    m_n: float = 939.5     # MeV
    n0: float = 0.16       # fm^-3
    a0: float = -96.64     # MeV
    b0: float = 58.85      # MeV
    gamma: float = 1.40
    a1: float = -26.06     # MeV
    b1: float = 7.34       # MeV
    gamma1: float = 2.45

    @classmethod
    def default(cls) -> "Parameters":
        """The published set of Constantinou et al.

        n_sat = 0.15951 fm^-3, E_sat = -16.00, K_sat = 250.2, E_sym = 30.85,
        L_sym = 41.26 MeV. All five are predictions of the couplings -- ZL
        imposes no saturation condition -- and `nmp.compute_nmp` reproduces
        them at T = 0, pinned by `verify/run_full_check.py`.

        A NEW set is `Parameters(a0=..., gamma=...)` -- every field carries a
        default, so only the ones that change need naming -- or
        `dataclasses.replace` of one already in hand. FROM nuclear-matter
        parameters there is no route: `nmp.invert_nmp` raises, because six
        couplings against five NMPs leaves a one-parameter family with no
        published closure. `nmp.compute_nmp` is the forward direction.
        """
        return cls()

    @classmethod
    def named(cls, name: str) -> "Parameters":
        """A published set by name.

        ZL ships exactly one, the set of Constantinou et al. that `default()`
        returns, so the map has a single entry; it exists because CLAUDE.md
        section 13 makes `named` part of the vocabulary every model speaks,
        and a caller that sweeps parameter sets must not have to know which
        models happen to have more than one. The key is the set's own `name`
        field, so `Parameters.named(par.name)` round-trips.
        """
        known = {"ZL_Constantinou": cls.default}
        if name not in known:
            raise KeyError(f"unknown ZL parameter set {name!r}; "
                           f"available: {sorted(known)}")
        return known[name]()
