"""Stellar structure and oscillations: what an equation of state is FOR.

This layer consumes tables and arrays produced by the models and the composite
engines and turns them into observables. It never imports model internals, and
no model imports it (CLAUDE.md section 1):

    general/  ->  models  ->  composite engines  ->  astro/

    tov/      stellar structure -- TOV, tidal deformability, crust attachment,
              and uniformly rotating models through the RNS backend
    gmode/    composition g-modes of the resulting stars

Both subpackages are imported on first attribute access, so `eos.astro.tov`
resolves after a bare `import eos` without paying for the other one.
"""
import importlib

_LAZY = ("tov", "gmode")


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
