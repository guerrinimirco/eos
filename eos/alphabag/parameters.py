"""Parameters of the alphaBag quark model.

Two numbers and three masses: the bag constant B, the QCD coupling alpha_s
that carries the leading perturbative correction, and the current quark
masses. The coupling does not run -- it is a constant of the set, which is
what makes it something an inference run can vary.

    B        confines: it costs energy density to make a bag, so it enters
             eps with a plus and P with a minus, and it is what holds the
             pressure negative until deconfinement pays for itself.
    alpha_s  softens: it multiplies the free-gas pressure by factors below
             one. Where vMIT stiffens the quark branch with a vector field,
             this model has no repulsion at all -- the two are not
             reparametrisations of one another.

B is carried as its fourth root B^(1/4) in MeV, the form it is quoted in;
`B` returns B itself in MeV^4, so a set cannot carry a B and a B4 that
disagree. The masses are current masses: there is no chiral condensate here to
dress them.

The zero-temperature pairing gap of the colour-flavour locked phase is
deliberately NOT here. It selects a phase rather than tuning one, the way a
species flag does, and is passed per call.

Parameters are ALWAYS arguments -- a `Parameters` instance passed to every
call -- never module-level constants. The dataclass is frozen so two sets can
coexist in one process without either being able to disturb the other.

References: Chodos et al., Phys. Rev. D 9, 3471 (1974) for the bag;
T. Fischer et al., Astrophys. J. Suppl. Ser. 194, 39 (2011) for the
arrangement of the alpha_s correction; M. Alford and S. Reddy, Phys. Rev. D
67, 074024 (2003) for the parameter ranges. See `alphabag.tex`.
"""
from dataclasses import dataclass


#: The CFL critical-temperature coefficient of the shipped set,
#: T_c = 0.57 * 2^(1/3) Delta0 -- the weak-coupling BCS result with the
#: colour factor of D. T. Son, Phys. Rev. D 59, 094019 (1999). It is the
#: DEFAULT of `Parameters.tc_coeff`, not a constant: an inference run over
#: CFL pairing varies it like any other parameter (CLAUDE.md section 6).
TC_COEFF = 0.57 * 2**(1.0/3.0)


@dataclass(frozen=True)
class Parameters:
    """One alphaBag parameter set.

    Attributes:
        name: label carried into table headers and figure legends
        m_u, m_d: up and down current masses (MeV). Zero in the shipped set;
                  below 1e-5 MeV the thermodynamics takes its massless
                  branch, which closes in elementary functions.
        m_s: strange current mass (MeV), carried exactly through the Fermi
             integrals of `eos.general.fermi_integrals`
        alpha: the QCD coupling alpha_s, dimensionless and constant. It enters
               as the three factors 1 - 2a/pi, 1 - 50a/(21 pi) and
               1 - 15a/(4 pi) multiplying the quark chemical, quark thermal
               and gluon terms.
        B4: bag constant B^(1/4) (MeV)
        tc_coeff: the CFL critical temperature as a multiple of the
                  zero-temperature gap, T_c = tc_coeff * Delta0. Only the
                  paired (`cfl`) mode reads it.
    """
    name: str = "alphabag_default"
    m_u: float = 0.0       # MeV (up quark mass, treated as massless)
    m_d: float = 0.0       # MeV (down quark mass, treated as massless)
    m_s: float = 150.0     # MeV (strange quark mass)
    alpha: float = 0.3     # dimensionless (QCD coupling alpha_s)
    B4: float = 165.0      # MeV (bag constant B^1/4)
    tc_coeff: float = TC_COEFF   # dimensionless (T_c = tc_coeff * Delta0)

    @property
    def B(self) -> float:
        """The bag constant itself, B = (B^(1/4))^4, in MeV^4.

        Divide by (hbar c)^3 to reach the MeV/fm^3 the rest of the code uses;
        `eos.alphabag.bag_pressure` is the one place that does.
        """
        return self.B4**4

    @classmethod
    def default(cls) -> "Parameters":
        """The working set of this repository: B^(1/4) = 165 MeV,
        alpha_s = 0.3, m_s = 150 MeV.

        A central choice within the ranges used for compact-star quark matter
        (B^(1/4) ~ 145-180 MeV, alpha_s ~ 0.2-0.5, m_s ~ 90-150 MeV), not a
        published fit: these are the axes a hybrid study scans, and which
        triple is right depends on the hadronic model it is paired with. A set
        with some of them changed is `Parameters(alpha=..., B4=...)`, or
        `dataclasses.replace` of one already in hand.
        """
        return cls()
