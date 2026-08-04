"""Import every module in the `eos` package.

Why this exists: `eos/sfho/compare_with_compose.py` sat un-importable for a long
time because it did `from eos.general.plotting_info import ... COLORS ...` and
`from eos.sfho.eos import SFHoEOS` -- two names that have never existed. Nothing
imported that module, so nothing noticed. A dangling name in a module no test
touches is invisible until someone opens it.

This sweep makes every module's import line a tested surface. It only imports;
it asserts nothing about behaviour.
"""
import importlib
import pkgutil

import matplotlib
matplotlib.use("Agg")          # no display in CI; some modules import pyplot

import pytest

import eos

# Modules that legitimately cannot be imported in a bare environment (optional
# heavy dependencies, __main__-only scripts). Keep this list SHORT and justified
# -- an entry here is a module the sweep no longer protects.
SKIP = {
    # (none at present)
}


def _all_modules():
    for mod in pkgutil.walk_packages(eos.__path__, prefix="eos."):
        if mod.name not in SKIP:
            yield mod.name


@pytest.mark.parametrize("name", sorted(_all_modules()))
def test_module_imports(name):
    importlib.import_module(name)


def test_eos_never_imports_nucleation():
    """`nucleation` depends on `eos`; the reverse would be a dependency cycle.

    `eos.dd2.notebook_api` used to import `nucleation.analysis.figure` for the
    M-R constraint overlay, which meant `pip install eos` alone could not draw
    that figure. The overlay now lives in `eos.general.observational_constraints`.
    """
    import sys

    blocked = "nucleation"

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"{name} is blocked: eos must not import it")
            return None

    guard = _Block()
    sys.meta_path.insert(0, guard)
    # Drop anything already imported so the guard actually gets consulted.
    cached = {k: sys.modules.pop(k) for k in list(sys.modules)
              if k == blocked or k.startswith(blocked + ".")}
    try:
        for name in _all_modules():
            importlib.import_module(name)
    finally:
        sys.meta_path.remove(guard)
        sys.modules.update(cached)
