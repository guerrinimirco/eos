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
        model it is paired with. The library ships no scan driver: model
        parameters are arguments (CLAUDE.md section 6), so a sweep over
        (B4, a, m_s) is a loop in the caller that builds one `Parameters` per
        sample, wraps it in a `Phase` and passes the pairing to the mixed
        engine (`eos.mixed.eos_table((hadronic, vmit_phase(params)), ...)`),
        one call per sample.

        A NEW set is `Parameters(B4=..., a=..., m_s=...)` -- every field
        carries a default, so only the ones that change need naming -- or
        `dataclasses.replace(Parameters.default(), B4=...)` of one already in
        hand. The dataclass is frozen, so `replace` is how a set is modified;
        there is no setter and no mutating helper. FROM nuclear-matter
        parameters there is no route: vMIT has no nuclear sector, so there is
        no `nmp.py` and nothing to invert.
        """
        return cls(name="vMIT_default")

    @classmethod
    def named(cls, name: str) -> "Parameters":
        """A parameter set by name.

        vMIT ships exactly one, the working set that `default()` returns, so
        the map has a single entry; it exists because CLAUDE.md section 13
        makes `named` part of the vocabulary every model speaks, and a caller
        that sweeps parameter sets must not have to know which models happen
        to have more than one. The key is the set's own `name` field, so
        `Parameters.named(par.name)` round-trips.
        """
        known = {"vMIT_default": cls.default}
        if name not in known:
            raise KeyError(f"unknown vMIT parameter set {name!r}; "
                           f"available: {sorted(known)}")
        return known[name]()
