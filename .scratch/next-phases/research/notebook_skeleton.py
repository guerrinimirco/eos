"""The shared spine of notebooks/{hadronic,quark,enjl,hybrid}_eos.

A throwaway prototype for ticket 04: three shapes to react to, not a library.
Whatever survives is COPIED into each notebook (CLAUDE.md section 11 forbids a
helper module beside them), except `standard_name`, which lands in
`eos/general/table_io.py` beside `save_table` -- section 7 makes general/ the
single home for table I/O, and four notebooks would otherwise carry four copies
of the same string-building.

Run it directly for the self-check: `python notebook_skeleton.py`.
"""
import os
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------- 1. the knobs
# The first executable cell. Everything the notebook does is selectable here and
# nowhere else; no cell below reaches past it for a number.

@dataclass
class Knobs:
    """Every choice the notebook makes, in one place.

    `mode` and the fractions are CLAUDE.md section 3; `leptons` is the
    orthogonal flag of that same section and is NOT a condition, so it is a
    field of its own rather than an entry in `conditions()`. The six species
    booleans are section 4, spelled with section 4's names in every model.
    """

    # --- which models ---------------------------------------------------
    models: tuple = ("zl", "sfho", "dd2", "did")

    # --- the equilibrium (section 3) ------------------------------------
    mode: str = "beta_eq_neutrinoless"
    Y_C: float = None                  # fixed_YC, fixed_YC_YS
    Y_S: float = None                  # fixed_YC_YS
    Y_Le: float = None                 # beta_eq_neutrino_trapped
    Y_Lmu: float = None                # beta_eq_neutrino_trapped, optional
    leptons: bool = True               # orthogonal to the mode

    # --- the grid -------------------------------------------------------
    n_B: tuple = (0.05, 1.2, 64)       # (lo, hi, count), fm^-3
    thermal: str = "T"                 # "T" or "SnB"
    thermal_grid: tuple = (0.0, 0.0, 1) # (lo, hi, count); MeV, or k_B per baryon

    # --- the sectors (section 4) ----------------------------------------
    species: dict = field(default_factory=lambda: dict(
        hyperons=False, deltas=False, muons=True,
        thermal_mesons=False, thermal_neutrinos=False, photons=True))

    # --- the parameters (section 6: parameters are arguments) -----------
    #   "default"          -> Parameters.default()
    #   ("named", "DD2Y")  -> Parameters.named("DD2Y")
    #   ("nmp", {...})     -> invert {n_sat, E_sat, m_eff, K_sat, E_sym, L_sym}
    #                         Q_sat and K_sym come back as PREDICTIONS.
    parameters: dict = field(default_factory=dict)   # per model; missing = default
    use_nmp_inversion: bool = False    # off by default: zl refuses by design and
                                       # dd2's inversion is ticket 47, unruled

    def axes(self):
        """The grid, as eos_table's `axes` argument."""
        lo, hi, n = self.n_B
        tlo, thi, tn = self.thermal_grid
        return {"nB": np.linspace(lo, hi, n),
                self.thermal: np.linspace(tlo, thi, tn)}

    def conditions(self):
        """Only the fractions THIS mode takes. Section 5 fixes these names."""
        taken = {"beta_eq_neutrinoless": (),
                 "beta_eq_neutrino_trapped": ("Y_Le", "Y_Lmu"),
                 "fixed_YC": ("Y_C",),
                 "fixed_YC_YS": ("Y_C", "Y_S"),
                 "cfl": ()}[self.mode]
        return {k: getattr(self, k) for k in taken if getattr(self, k) is not None}


# --------------------------------------------- 2. unsupported combinations
# Three distinct failure shapes, and the whole point is that they stay distinct.
# A gap is never presented as a result, and a result is never presented as a gap.

class Skipped(Exception):
    """Raised by nothing; the marker a section returns when a model refused."""


def run_section(name, call, **kwargs):
    """Call one model's public entry point, and report which of three happened.

    Returns (status, payload) with status in {"ok", "unsupported", "unconverged"}.

      "unsupported"  -- the model refused the mode, flag or parametrisation, and
                        said which (section 3). NotImplementedError and
                        ValueError only: those are the two a refusal uses.
      "unconverged"  -- the solve ran and did not converge. Section 6 makes this
                        a RETURN VALUE, so no except clause ever sees it; it is
                        found by testing `.ok`. Calling it "unsupported" would
                        be a lie about the physics.
      "ok"           -- payload is the result.

    TypeError is deliberately NOT caught. An unexpected keyword is the
    notebook's own bug, and a broad except would file it under "this model does
    not support that" in a way nobody would ever notice.
    """
    try:
        result = call(**kwargs)
    except (NotImplementedError, ValueError) as err:
        print(f"  [{name}] not supported: {err}")
        return "unsupported", None

    ok = getattr(result, "ok", True)
    if not ok:
        print(f"  [{name}] did not converge: {getattr(result, 'message', '')}")
        return "unconverged", result
    return "ok", result


def section(title, knobs):
    """One printed section header, so a skipped model is visible in the output."""
    print(f"\n=== {title} — mode={knobs.mode} "
          f"{knobs.conditions()} leptons={knobs.leptons} ===")


# ------------------------------------------------------- 3. saving a table
# `save_table(rows, path, meta=..., windows=...)` already exists in
# eos/general/table_io.py. What is missing is the NAME. This function goes
# there, beside it.

_FLAG_TOKENS = {"hyperons": "hyp", "deltas": "del", "muons": "mu",
                "thermal_mesons": "mes", "thermal_neutrinos": "nu",
                "photons": "ph"}


def standard_name(model, mode, conditions, axes, species, leptons=True,
                  eta=None, ext="h5"):
    """The automatic file name for a generated table.

    Every choice that changes a number is in the name, so two tables in one
    folder cannot collide silently and a name alone says how it was made:

        dd2_fixed_YC_YC0.100_T0.0-30.0x4_nB0.050-1.200x64_mu+ph.h5
        vmit_beta_eq_neutrinoless_T0.0x1_nB0.100-1.500x32_ph_nolep.h5
        dd2vmit_fixed_YC_YC0.100_eta0.30_T0.0x1_nB0.05...

    Order: model, mode, the mode's fractions, eta if a composite engine, the
    thermal axis, the density axis, the sectors that are ON, and `nolep` only
    when leptons are off (their presence is the common case).

    The full metadata still goes into the file through `save_table(meta=...)`;
    the name is for the human reading the folder.
    """
    parts = [model, mode]
    for key in ("Y_C", "Y_S", "Y_Le", "Y_Lmu"):
        if conditions.get(key) is not None:
            parts.append(f"{key.replace('_', '')}{conditions[key]:.3f}")
    if eta is not None:
        parts.append(f"eta{eta:.2f}")
    for key in ("T", "SnB"):
        if key in axes:
            parts.append(f"{key}{_span(axes[key])}")
    parts.append(f"nB{_span(axes['nB'])}")

    on = [tok for flag, tok in _FLAG_TOKENS.items() if species.get(flag)]
    parts.append("+".join(on) if on else "bare")
    if not leptons:
        parts.append("nolep")
    return "_".join(parts) + "." + ext


def _span(grid):
    """A grid as `lo-hix n` — or just `lo x1` when it is a single value."""
    grid = np.atleast_1d(np.asarray(grid, dtype=float))
    if grid.size == 1:
        return f"{grid[0]:.1f}x1"
    return f"{grid[0]:.1f}-{grid[-1]:.1f}x{grid.size}"


def table_path(model, name, root="output/tables"):
    """`output/tables/<model>/<name>` — section 11's per-model subfolder.

    Created on demand; the notebook prints the path so the reader can find it.
    """
    folder = os.path.join(root, model)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


# ---------------------------------------------------------------- self-check
if __name__ == "__main__":
    k = Knobs(mode="fixed_YC", Y_C=0.1, Y_S=0.4,
              thermal_grid=(0.0, 30.0, 4))

    # conditions() carries only what the MODE takes: Y_S is set but fixed_YC
    # does not take it, so it must not appear.
    assert k.conditions() == {"Y_C": 0.1}, k.conditions()
    assert "leptons" not in k.conditions()          # section 5: not a condition

    name = standard_name("dd2", k.mode, k.conditions(), k.axes(), k.species,
                         leptons=k.leptons)
    assert name == ("dd2_fixed_YC_YC0.100_T0.0-30.0x4"
                    "_nB0.1-1.2x64_mu+ph.h5"), name

    cold = Knobs(mode="beta_eq_neutrinoless", n_B=(0.1, 1.5, 32), leptons=False,
                 species=dict(hyperons=False, deltas=False, muons=False,
                              thermal_mesons=False, thermal_neutrinos=False,
                              photons=True))
    assert standard_name("vmit", cold.mode, cold.conditions(), cold.axes(),
                         cold.species, leptons=False).endswith("_ph_nolep.h5")

    # a composite engine's eta lands in the name, and nothing else moves
    assert "_eta0.30_" in standard_name("dd2vmit", k.mode, k.conditions(),
                                        k.axes(), k.species, eta=0.3)

    # the three failure shapes stay three
    class Bad:
        ok, message = False, "no root bracketed"

    assert run_section("m", lambda: (_ for _ in ()).throw(
        NotImplementedError("finite T not wired")))[0] == "unsupported"
    assert run_section("m", lambda: Bad())[0] == "unconverged"
    assert run_section("m", lambda: "table")[0] == "ok"
    try:
        run_section("m", lambda: None, thermal_mesons=True)
    except TypeError:
        pass                    # the notebook's own bug, never swallowed
    else:
        raise AssertionError("TypeError must escape run_section")

    print("\nself-check passed")
