"""Parameters of the vMIT quark model.

Four numbers set the equation of state: the bag constant B, the vector
coupling a = g_V^2/m_V^2, and the three current quark masses. There is nothing
else to fit -- no scalar sector, no density dependence -- which is what makes
the model cheap enough to scan over, and why B and a are the two axes a hybrid
parameter study moves along.

    B     confines: it costs energy density to make a bag, so it enters eps
          with a plus and P with a minus, and it is what holds the pressure
          negative until deconfinement pays for itself.
    a     repels: it raises P at fixed density and so stiffens the quark
          branch, which is how a hybrid star reaches two solar masses.

B is carried as its fourth root B^(1/4) in MeV, the form it is quoted in;
`B` returns B itself in MeV^4. a is in fm^2. The masses are current masses in
MeV, not constituent masses: there is no chiral condensate here to dress them.

Parameters are ALWAYS arguments -- a `Parameters` instance passed to every
call -- never module-level constants. Inference varies B and a across millions
of evaluations, and a parameter that can only be changed by editing a source
file makes that impossible. The dataclass is frozen so two parametrizations
can coexist in one process without either being able to disturb the other, and
so it can key a read-only cache.

References: Chodos et al., Phys. Rev. D 9, 3471 (1974) for the bag; the vector
term as used by Gomes et al., Astrophys. J. 877, 139 (2019) and Constantinou
et al., Phys. Rev. D 104, 123032 (2021) and 107, 074013 (2023). See `vmit.tex`.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    """One vMIT parametrization.

    Attributes:
        name: label carried into table headers and figure legends
        m_u, m_d, m_s: current quark masses (MeV)
        a: vector coupling g_V^2/m_V^2 (fm^2)
        B4: bag constant B^(1/4) (MeV)
    """
    name: str = "vMIT_default"
    m_u: float = 5.0       # MeV (up quark mass)
    m_d: float = 7.0       # MeV (down quark mass)
    m_s: float = 150.0     # MeV (strange quark mass)
    a: float = 0.2         # fm^2 (vector coupling = g_V^2/m_V^2)
    B4: float = 180.0      # MeV (bag constant B^1/4)

    @property
    def B(self) -> float:
        """The bag constant itself, B = (B^(1/4))^4, in MeV^4.

        Divide by (hbar c)^3 to reach the MeV/fm^3 the rest of the code uses;
        `eos.vmit.bag_pressure` is the one place that does.
        """
        return self.B4**4

    @classmethod
    def default(cls) -> "Parameters":
        """The working parametrization of this repository: B^(1/4) = 180 MeV,
        a = 0.2 fm^2, m_s = 150 MeV.

        A starting point, not a published fit: vMIT's parameters are what a
        hybrid study scans, and which pair is right depends on the hadronic
        model it is paired with. `eos.mixed.scan` moves over (B4, a, m_s).
        """
        return cls(name="vMIT_default")


def get_vmit_custom(
    m_u: float = 5.0, m_d: float = 7.0, m_s: float = 150.0,
    a: float = 0.2, B4: float = 180.0, name: str = "vMIT_custom"
) -> Parameters:
    """A parametrization with any subset of the four numbers changed.

    Keyword-named so a scan can write `get_vmit_custom(B4=170.0, a=0.15)` and
    leave the rest at their defaults.
    """
    return Parameters(
        name=name,
        m_u=m_u, m_d=m_d, m_s=m_s,
        a=a, B4=B4
    )
