"""What a mode IS: one choice per conserved charge, and the weak-equilibrium
relations that choice selects.

A solver needs three families of equation, and only the last two depend on the
mode:

    field equations         the model's own self-consistency. ALWAYS all of
                            them, whatever the mode -- a mean field does not
                            know what is being held fixed.
    conserved-charge        n_C = Y_C n_B, n_S = Y_S n_B, ...
    equilibrium relations   mu_S = 0, mu_C + mu_e = mu_nue, ...

and the second and third are *paired*: for each charge, either its fraction is
imposed and its potential is an unknown, or its potential is set by a relation
and the fraction comes out. That is one binary choice per charge, and it is the
whole content of a mode.

So the equations are written ONCE each, here and in the model, and a mode is a
declaration read by the residual assembly -- never a per-mode residual
function. `eos.mixed` has worked this way from the start (its `ChargeSpec`
drives the unknown vector, the residual list and the analytic Jacobian alike,
and its four named modes carry no per-mode code); this is the same idea for a
single phase, which is the case every model has.

A two-phase system needs one axis more: a charge conserved on the volume
average (Gibbs) versus inside each phase separately (Maxwell). `eos.mixed`
adds that on top; FIXED here is what it refines into GLOBAL and LOCAL, and
EQUILIBRATED here is its NOT_CONSERVED.

Sign conventions are the repository's (CLAUDE.md section 2): C is the charge
of strongly-interacting matter only, S = +1 per s quark, and
mu_C = mu_p - mu_n, so beta equilibrium reads mu_C + mu_e = 0.
"""
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class Conservation(Enum):
    """How one conserved charge is treated by a mode."""
    #: Its fraction is imposed; its conjugate potential is a solver unknown.
    FIXED = "fixed"
    #: Its potential is set by an equilibrium relation; the fraction comes out.
    EQUILIBRATED = "equilibrated"


#: charge name -> the `targets` key carrying its fraction, in the condition
#: names CLAUDE.md section 5 fixes.
TARGET_KEY = {"C": "Y_C", "S": "Y_S", "L_e": "Y_Le", "L_mu": "Y_Lmu"}

#: charge name -> the potential that becomes an unknown when it is FIXED.
POTENTIAL = {"C": "mu_C", "S": "mu_S", "L_e": "mu_nue", "L_mu": "mu_numu"}


@dataclass(frozen=True)
class ModeSpec:
    """One mode, as a choice per conserved charge.

    Baryon number is not listed: it is fixed in every mode, by n_B itself.
    Everything else defaults to EQUILIBRATED, which is plain neutrino-transparent
    beta equilibrium.

    `leptons` distinguishes the two flavours of a fixed-fraction mode: with
    leptons the neutralizing electrons (and muons, if the species flags enable
    them) are present so the total system is electrically neutral; without
    them the result is charged matter, which is what a mixed-phase construction
    needs per pure phase before imposing global neutrality. It has no meaning
    when C is EQUILIBRATED -- beta equilibrium is *about* the leptons.

    Prefer the named factories below; build a ModeSpec directly only for a
    combination they do not cover.
    """
    C: Conservation = Conservation.EQUILIBRATED
    S: Conservation = Conservation.EQUILIBRATED
    L_e: Conservation = Conservation.EQUILIBRATED
    L_mu: Conservation = Conservation.EQUILIBRATED
    targets: Mapping[str, float] = field(default_factory=dict)
    leptons: bool = True

    def __post_init__(self):
        for charge, key in TARGET_KEY.items():
            fixes = getattr(self, charge) is Conservation.FIXED
            given = key in self.targets
            if fixes and not given:
                raise ValueError(
                    f"{charge} is FIXED but no {key} target was given")
            if not fixes and given:
                raise ValueError(
                    f"{charge} is EQUILIBRATED; a {key} target is meaningless")
        if self.L_mu is Conservation.FIXED and self.L_e is not Conservation.FIXED:
            raise ValueError(
                "a trapped muon family (L_mu FIXED) with a free electron "
                "family is not a defined mode: the two are coupled by "
                "mu_mu = mu_e - mu_nue + mu_numu")
        if not self.leptons and self.C is not Conservation.FIXED:
            raise ValueError(
                "leptons=False only applies to a fixed-fraction mode; beta "
                "equilibrium is defined by the leptons")
        object.__setattr__(self, "targets",
                           MappingProxyType(dict(self.targets)))

    # A mappingproxy cannot be pickled, and CLAUDE.md section 6 requires model
    # objects to survive being sent to a worker process. Unwrap for the wire,
    # re-wrap on arrival, so immutability holds in every process.
    def __getstate__(self):
        state = dict(self.__dict__)
        state["targets"] = dict(state["targets"])
        return state

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "targets",
                           MappingProxyType(dict(state["targets"])))

    def is_fixed(self, charge):
        return getattr(self, charge) is Conservation.FIXED

    @property
    def name(self):
        """The mode's name, built from which charges it holds.

        The four names of CLAUDE.md section 3, plus the two combinations the
        named factories below do not have a word for: strangeness held with
        the charge equilibrated (`fixed_YS`), and that on top of a trapped
        lepton family. `leptons` is deliberately not in the name -- section 3
        makes it an orthogonal flag on the fixed-fraction modes, so
        `fixed_YC` asked with and without neutralizing leptons is one mode
        asked two ways, not two modes.
        """
        if self.is_fixed("C"):
            return "fixed_YC_YS" if self.is_fixed("S") else "fixed_YC"
        if self.is_fixed("L_e"):
            base = "beta_eq_neutrino_trapped"
        else:
            base = "beta_eq_neutrinoless"
        return f"{base}_fixed_YS" if self.is_fixed("S") else base


# =============================================================================
# THE NAMED MODES  (CLAUDE.md section 3)
# =============================================================================

def beta_eq_neutrinoless():
    """Beta equilibrium, free-streaming neutrinos. Variables (n_B, T).

    Nothing but baryon number is conserved: strangeness self-equilibrates
    (mu_S = 0), the neutrinos escape (mu_nue = mu_numu = 0), and the charge
    potential is fixed by mu_C + mu_e = 0 together with electric neutrality.
    """
    return ModeSpec()


def beta_eq_neutrino_trapped(Y_Le, Y_Lmu=None):
    """Beta equilibrium with trapped neutrinos. Variables (n_B, Y_Le, [Y_Lmu], T).

    The electron lepton family is conserved at Y_Le = (n_e + n_nue)/n_B, so
    mu_nue is an unknown rather than zero. Passing Y_Lmu traps the muon family
    too; without it the muon family stays transparent, mu_numu = 0.
    """
    targets = {"Y_Le": Y_Le}
    L_mu = Conservation.EQUILIBRATED
    if Y_Lmu is not None:
        targets["Y_Lmu"] = Y_Lmu
        L_mu = Conservation.FIXED
    return ModeSpec(L_e=Conservation.FIXED, L_mu=L_mu, targets=targets)


def fixed_YC(Y_C, leptons=True):
    """Fixed non-leptonic charge fraction. Variables (n_B, Y_C, T).

    The simulation-table mode. Strangeness still self-equilibrates.
    """
    return ModeSpec(C=Conservation.FIXED, targets={"Y_C": Y_C},
                    leptons=leptons)


def fixed_YC_YS(Y_C, Y_S, leptons=True):
    """Fixed charge and strangeness. Variables (n_B, Y_C, Y_S, T).

    Y_C = 0.5, Y_S = 0 is symmetric nuclear matter, for heavy-ion comparisons.
    mu_S becomes an output, determined by how much strangeness was demanded.
    """
    return ModeSpec(C=Conservation.FIXED, S=Conservation.FIXED,
                    targets={"Y_C": Y_C, "Y_S": Y_S}, leptons=leptons)


# =============================================================================
# WHAT A MODE IMPLIES
# =============================================================================

def charge_unknowns(spec):
    """The conserved-charge potentials this mode makes unknowns, in order.

    A model prepends its own unknowns -- its mean fields and its baryon
    potential -- since which of those it iterates on is a conditioning choice
    of the solver, not a property of the mode.
    """
    return tuple(POTENTIAL[charge]
                 for charge in ("C", "S", "L_e", "L_mu")
                 if spec.is_fixed(charge))


def charge_conditions(spec):
    """The conserved-charge rows this mode imposes, in the same order.

    One row per FIXED charge, plus electric neutrality wherever leptons are
    present -- which is every beta-equilibrium mode and the leptons=True
    flavour of a fixed-fraction one.
    """
    rows = [TARGET_KEY[charge] for charge in ("C", "S", "L_e", "L_mu")
            if spec.is_fixed(charge)]
    if has_leptons(spec):
        rows.append("neutrality")
    return tuple(rows)


def has_leptons(spec):
    """Are neutralizing leptons present?

    Beta equilibrium always needs them. A fixed-Y_C mode needs them only in
    its neutralizing flavour; leptonless fixed-Y_C is a charged slice of
    matter with no neutrality condition at all.
    """
    return spec.C is not Conservation.FIXED or spec.leptons


# =============================================================================
# THE WEAK-EQUILIBRIUM RELATIONS
# =============================================================================
# Written once for the whole repository. Both hold in the conserved-charge
# basis of CLAUDE.md section 2, where mu_C = mu_p - mu_n.

def electron_potential(mu_C, mu_nue=0.0):
    """mu_e from beta equilibrium, n + nu_e <-> p + e.

        mu_C + mu_e = mu_nue      =>      mu_e = mu_nue - mu_C

    With free-streaming neutrinos (mu_nue = 0) this is the repository's
    mu_C + mu_e = 0. Only meaningful where C is EQUILIBRATED: in a fixed-Y_C
    mode with leptons, mu_e is set by neutrality instead and is an unknown of
    its own.
    """
    return mu_nue - mu_C


def muon_potential(mu_e, mu_nue=0.0, mu_numu=0.0):
    """mu_mu from muon decay equilibrium, mu- <-> e- + nu_e_bar + nu_mu.

        mu_mu = mu_e - mu_nue + mu_numu

    Transparent matter gives the familiar mu_mu = mu_e; trapping only the
    electron family gives mu_mu = mu_e - mu_nue, since the muon neutrinos
    still escape.
    """
    return mu_e - mu_nue + mu_numu


def strangeness_potential(spec):
    """mu_S where strangeness is not conserved.

    Strangeness-changing weak reactions are fast compared with any
    astrophysical timescale, so mu_S = 0 unless the mode holds Y_S fixed --
    in which case mu_S is an unknown and this does not apply.
    """
    if spec.is_fixed("S"):
        raise ValueError("S is FIXED in this mode: mu_S is an unknown, not a "
                         "quantity derived from an equilibrium relation")
    return 0.0
