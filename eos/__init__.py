"""`eos` — equations of state for dense nuclear and quark matter.

What is one import deep. `import eos` costs milliseconds and pulls in no
model: the ten model packages, the composite engine and `astro` are
imported on first attribute access, so `eos.dd2` works after a bare
`import eos` without every other model's Numba kernels being compiled first.

    import eos
    par   = eos.dd2.Parameters.named("DD2Y")      # parameters are ARGUMENTS
    flags = eos.dd2.SpeciesFlags(hyperons=True)   # every d.o.f. is explicit
    res   = eos.dd2.eos_point(par, "beta_eq_neutrinoless", flags,
                              n_B=0.32, T=10.0)
    res.ok, res.point.P                            # convergence is a RETURN VALUE

Every model exposes the same three entry points with the same signatures
(CLAUDE.md section 5):

    eos_point(par, mode, species, **conditions)     quantities at one point
    eos_table(par, mode, species, axes)             a solved grid
    eos_response(par, mode, species, frozen=...)    second derivatives

`mode` is one of `eos.MODES`, `species` is the model's own `SpeciesFlags`
whose fields are named in `eos.SPECIES_FLAGS`, and the conditions are named
exactly n_B, T, Y_C, Y_S, Y_Le, Y_Lmu. Units at every public boundary are
fm-based: n in fm^-3, T and mu in MeV, eps and P in MeV/fm^3.

The mode FACTORIES re-exported here (`beta_eq_neutrinoless`, `fixed_YC`, ...)
build the `ModeSpec` declarations the solvers read; the uniform API above
takes the mode by name, as a string.
"""
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # points to eos/ repo root

from eos.general.modes import (
    ModeSpec, Conservation,
    beta_eq_neutrinoless, beta_eq_neutrino_trapped, fixed_YC, fixed_YC_YS,
)
from eos.general.state import EOSTable_for_TOV
from eos.general.table_io import save_table, load_table, export_csv

#: The model subpackages: one physics model each, importing only `general/`.
MODELS = ("dd2", "sfho", "zl", "did", "vmit",
          "alphabag", "abpr", "enjl", "njl", "ccdm")

#: The composite engines. `mixed` couples one hadronic and one quark phase
#: through the phase-adapter contract; `zlvmit` is the first-generation
#: hybrid, kept for its published results and exempt from the uniform API.
ENGINES = ("mixed", "zlvmit")

#: The equilibrium modes of CLAUDE.md section 3, as the strings `eos_point`,
#: `eos_table` and `eos_response` take. A mode fixes the independent
#: variables:
#:
#:     beta_eq_neutrinoless       (n_B, T)
#:     beta_eq_neutrino_trapped   (n_B, Y_Le, [Y_Lmu], T)
#:     fixed_YC                   (n_B, Y_C, T)
#:     fixed_YC_YS                (n_B, Y_C, Y_S, T)
#:     cfl                        (n_B, T)      colour-flavour-locked only
#:
#: `cfl` is a statement about which phase the model describes rather than a
#: choice of equilibrium condition, so only the locked-phase models
#: (`alphabag`, `abpr`) expose it. A model may offer further combinations
#: under its own `MODES`; these five are the shared vocabulary.
MODES = ("beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
         "fixed_YC", "fixed_YC_YS", "cfl")

#: The species flags of CLAUDE.md section 4: the fields every model's
#: `SpeciesFlags` carries under these names. Nucleons are always present;
#: everything else is an explicit boolean, and setting one a model does not
#: implement raises rather than being silently ignored. A model adds flags of
#: its own for physics only it has (`gluons`, `csc`, dd2's matter-composition
#: `neutrinos`); those default by the same rule as these six, below. A sector
#: a model ALREADY carries a coupling for gets no flag at all: setting that
#: coupling to zero is the same statement, made where every other model number
#: is made and continuously variable by a sampler, so a boolean beside it would
#: be a second way to say one thing. This is why the hidden-strange vector phi
#: has no flag in `dd2`, `sfho` or `did`.
#:
#: All ten models carry all six names: the six keywords construct a
#: `SpeciesFlags` anywhere, and no model answers them with a TypeError that
#: would read as the caller's bug. Carrying a name is not wiring the sector --
#: `dd2` (and `eos.mixed`, whose own flags carry the same six names) raises
#: NotImplementedError on `thermal_neutrinos=True`, the flavours a mode does
#: not track being unwired there; dd2's own `neutrinos` is the
#: matter-composition electron neutrino of the trapped modes, a different
#: sector with its own flag.
#:
#: All six DEFAULT TO FALSE in every model: off unless asked for, so
#: `SpeciesFlags()` means the same thing everywhere and no call inherits a
#: sector it did not name. This is what section 4's "if a sector is off, its
#: flag is False" costs at the default row, and it is a behaviour change from
#: the versions where `photons` defaulted True in six models and `muons` in
#: five -- a T > 0 call that relied on the default no longer carries the
#: photon gas, and must ask for it. A MODEL'S OWN FLAGS FOLLOW THE SAME RULE:
#: a flag with two legal values is a DEFAULT and is False, whatever its name;
#: a flag with only one legal value RAISES on the other and is a STATEMENT
#: about the model rather than a default. There is no third category. So
#: `alphabag.gluons` is False (both values are legal physics -- a bag model
#: without a thermal gluon gas is the standard MIT configuration), while
#: `dd2.sigma_star` raises, the model not having that field. `njl.csc` and
#: `ccdm.csc` were already False. `enjl` is the one exemption, and a different kind of
#: default -- it fixes every flag and RAISES on any move, so its
#: `hyperons=True` states which baryons the model has rather than a
#: convenience the caller inherited.
SPECIES_FLAGS = ("hyperons", "deltas", "muons", "thermal_mesons",
                 "thermal_neutrinos", "photons")

_LAZY = MODELS + ENGINES + ("general", "astro")


def __getattr__(name):
    """Import a subpackage on first attribute access (PEP 562).

    Eagerly importing the ten models would cost a second of Numba and SciPy
    for `import eos`, most of it for models the caller never touches.
    """
    if name in _LAZY:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module          # subsequent lookups skip this hook
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "MODELS", "ENGINES", "MODES", "SPECIES_FLAGS", "REPO_ROOT",
    "ModeSpec", "Conservation",
    "beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
    "fixed_YC", "fixed_YC_YS",
    "EOSTable_for_TOV", "save_table", "load_table", "export_csv",
    *MODELS, *ENGINES, "general", "astro",
]
