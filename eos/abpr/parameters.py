"""Parameters of the ABPR colour-flavour locked parametrization.

Four numbers: the bag constant B, the pQCD factor a4 that carries the leading
perturbative correction, the strange current mass, and the pairing gap.

    a4       softens: it multiplies the free-gas pressure, and is the same
             knob as the alpha_s of `eos.alphabag` through
             alpha_s = pi/2 (1 - a4). a4 = 1 is the free quark gas.
    m_s      costs: it enters as the leading expansion term
             -3 m_s^2 mu^2/(4 pi^2), not as an exact Fermi gas -- that is
             what distinguishes this model from `eos.alphabag`.
    Delta0   pays: the condensation energy +3 Delta0^2 mu^2/pi^2 is what
             makes the paired phase competitive, and Delta0 > m_s/2 is
             exactly the condition under which it wins against the mass term.
    B        confines: it costs energy density to make a bag, so it enters
             eps with a plus and P with a minus.

B is carried as its fourth root B^(1/4) in MeV, the form it is quoted in;
`B` returns B itself in MeV^4 -- the same unit, for the same attribute, as
`eos.alphabag.Parameters.B` and `eos.vmit.VMITParams.B` -- so a set cannot
carry a B and a B4 that disagree, and the single division by (hbar c)^3
happens where the pressure is assembled.

Unlike `eos.alphabag`, the gap belongs HERE rather than being passed per
call. In that model Delta0 selects between two phases one potential can be
in, so it is a per-call phase selector; in the ABPR parametrization it is
fitted alongside a4, m_s and B and is one of the four numbers that define the
set. Two homes for one number is what this avoids.

Parameters are ALWAYS arguments -- a `Parameters` instance passed to every
call -- never module-level constants. The dataclass is frozen so two sets can
coexist in one process without either being able to disturb the other.

References: M. Alford, M. Braby, M. Paris and S. Reddy, Astrophys. J. 629,
969 (2005) for the parametrization; Chodos et al., Phys. Rev. D 9, 3471
(1974) for the bag; B. Freedman and L. McLerran, Phys. Rev. D 16, 1169 (1977)
for the perturbative correction the factor a4 stands for. See `abpr.tex`.
"""
from dataclasses import dataclass

from eos.general.physics_constants import PI


@dataclass(frozen=True)
class Parameters:
    """One ABPR parameter set.

    Attributes:
        name: label carried into table headers and figure legends
        m_s: strange current quark mass (MeV). The light flavours are massless
             and have no parameter here.
        Delta0: the CFL pairing gap (MeV). Constant -- this model is defined
                at T = 0, so there is no Delta(T) and no critical temperature.
        a4: the pQCD factor, dimensionless, a4 = 1 - 2 alpha_s/pi. Typical
            range 0.6 - 1.0, with 1.0 the free gas.
        B4: bag constant B^(1/4) (MeV)
    """
    name: str = "abpr_default"
    m_s: float = 150.0      # MeV (strange quark mass)
    Delta0: float = 80.0    # MeV (CFL pairing gap)
    a4: float = 0.7         # dimensionless (pQCD factor)
    B4: float = 135.0       # MeV (bag constant B^1/4)

    @property
    def B(self) -> float:
        """The bag constant itself, B = (B^(1/4))^4, in MeV^4.

        Divide by (hbar c)^3 to reach the MeV/fm^3 the rest of the code uses;
        `eos.abpr.thermodynamics.pressure` is the one place that does.
        """
        return self.B4**4

    @property
    def alpha(self) -> float:
        """The QCD coupling the factor a4 stands for, alpha_s = pi/2 (1 - a4).

        This identity is what lets this model and `eos.alphabag` be driven as
        a matched pair: a4 and alpha_s are one knob written two ways.
        """
        return PI / 2.0 * (1.0 - self.a4)

    @classmethod
    def default(cls) -> "Parameters":
        """The working set of this repository: m_s = 150 MeV, Delta0 = 80 MeV,
        a4 = 0.7 (alpha_s = 0.4712), B^(1/4) = 135 MeV.

        There is no published single ABPR set. The four numbers span a range
        that a hybrid-star study scans -- a4 in [0.6, 1], Delta0 up to about
        200 MeV, B^(1/4) ~ 130-180 MeV, m_s ~ 90-250 MeV -- and which point in
        it is right depends on the hadronic model the quark phase is paired
        with. This is the set the repository's numerical baseline is frozen
        at, and at it the P = 0 surface has E/A = 831.6 MeV, below the 930 MeV
        of iron: absolutely stable strange quark matter.

        A set with some of them changed is `Parameters(a4=..., B4=...)`, or
        `dataclasses.replace` of one already in hand.
        """
        return cls()
