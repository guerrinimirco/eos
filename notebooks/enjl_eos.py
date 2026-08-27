# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ENJL — baryons and quarks from one functional
#
# The extended Nambu-Jona-Lasinio model of
#
# > C.-J. Xia, *Extended NJL model for baryonic matter and quark matter*,
# > Phys. Rev. D **110**, 014022 (2024), arXiv:2405.02946
#
# A baryon here is a three-quark cluster whose mass is built from the same
# constituent masses the NJL gap equation determines, so the chiral, quarkyonic
# and deconfinement transitions come out of **one** mean field rather than out of
# two models joined at a boundary. That is why this model gets a notebook of its
# own rather than a column in the hadronic or the quark one: it is neither, and
# the object it has that neither of them has is a **branch pair** — two
# self-consistent states of the same thermodynamic potential at the same
# potentials, which is what a first-order transition in this model is a
# construction between.
#
# Driven through the public API and nothing else: `eos_point`, `eos_table` and
# the model's parameter and species objects. No solver internal is touched, and
# no helper module sits beside this notebook — everything it needs is either in
# the library or in the cells below.
#
# What is here:
#
# 1. **The knobs** — every choice this notebook makes, in one cell.
# 2. **Reporting a gap** — the three distinct things that can happen when the
#    model is asked for something, and why they must stay three.
# 3. **Saving a table** — the automatic name, and the one argument that has to
#    be passed for a notebook to write where it means to.
# 4. **A section per mode** — the four equilibrium conditions, at T = 0.
# 5. **The branch pair** — the two branches of the one functional.
# 6. **The author's tables** — reproduced, with the residual printed.
# 7. **Temperature** — what is closed above T = 0 and what is not.
# 8. **Figures** — into `output/enjl/`.
# 9. **The step-by-step treatment** — the five steps the quark notebook takes,
#    three of which this model has, one of which it answers with a different
#    object, and one of which it does not have at all.
# 10. **Benchmarks** — what a point, a line and a branch cost.
#
# Units are the ones every public boundary uses: densities in fm^-3,
# temperatures and chemical potentials in MeV, pressure and energy density in
# MeV/fm^3. Strangeness is S = +1 per s quark, the opposite of the PDG
# convention, and `Y_C` is the charge fraction of the strongly interacting
# matter alone — the leptons are not in it.

# %%
import sys
from pathlib import Path

import numpy as np

# `eos` is imported from this checkout rather than from site-packages: the
# package is not installed, so the repository root goes on the path first. This
# works whether the notebook is run from `notebooks/` or from the root.
ROOT = Path.cwd()
if not (ROOT / "eos").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import eos.enjl as enjl
from eos.enjl.table import TableSpec, build_constructed_table
from eos.general.table_io import save_table, standard_name, table_path

# %% [markdown]
# ## 1. The knobs
#
# Everything selectable is selectable here and nowhere else; no cell below
# reaches past this one for a number.
#
# The axis this notebook sweeps is **the parameter set**, not the model. The six
# published sets are the six `(f_q, B)` combinations of the study — `f_q`
# rescales the quark coupling to the vector fields, `B` is the Pauli-blocking
# strength of Eq. (4) — and five of them have an author reference table, which
# section 6 reproduces. Parameters are arguments: every call below takes a
# `Parameters` object, and `named()` returns a published one rather than
# reaching for module state.
#
# `conditions(mode)` returns only the fractions *that* mode takes: set `Y_S`
# while asking for `fixed_YC` and it is dropped rather than quietly accepted.
# `leptons` is orthogonal to the mode — it says whether neutralizing electrons
# and muons are added to a fixed-fraction solve — so it is a field of its own
# and never an entry in `conditions()`.
#
# `Y_Lmu` is `None` and stays `None`: this model's trapped mode holds `Y_Le`
# alone, and `conditions()` drops what is not set, so nothing has to know that
# here.

# %%
from dataclasses import dataclass, field


@dataclass
class Knobs:
    """Every choice this notebook makes, in one place."""

    # --- which published parameter sets ---------------------------------
    sets: tuple = ("fq0.5_B1", "fq0.7_B1", "fq1.0_B1")

    # --- the equilibria to exercise, and the fractions they take ---------
    modes: tuple = ("beta_eq_neutrinoless", "beta_eq_neutrino_trapped",
                    "fixed_YC", "fixed_YC_YS")
    Y_C: float = 0.3                   # fixed_YC, fixed_YC_YS
    Y_S: float = 0.0                   # fixed_YC_YS
    Y_Le: float = 0.3                  # beta_eq_neutrino_trapped
    Y_Lmu: float = None                # not a fraction this model's mode takes
    leptons: bool = True               # orthogonal to the mode

    # --- the grid -------------------------------------------------------
    n_B: tuple = (0.10, 1.60, 31)      # (lo, hi, count), fm^-3
    thermal: str = "T"                 # "T" or "SnB"
    thermal_grid: tuple = (0.0, 20.0)  # MeV, or k_B per baryon

    # --- the branch pair ------------------------------------------------
    # Which continuation `eos_table` follows: "up" from the low-density
    # chirally broken side, "down" from a deconfined guess at the top of the
    # grid. Section 5 is about the difference.
    directions: tuple = ("up", "down")

    # --- the sectors ----------------------------------------------------
    # Every one is explicit. Four of the six are FIXED by this model rather
    # than chosen — moving them raises, and section 2 prints the reason — and
    # two, the thermal sectors that carry no conserved charge, are the
    # caller's. Both are identically zero at T = 0.
    species: dict = field(default_factory=lambda: dict(
        hyperons=True, deltas=False, muons=True,
        thermal_mesons=False, thermal_neutrinos=False, photons=False))

    # --- the author tables ----------------------------------------------
    # Section 6 reads them from here if they are present and says so if they
    # are not. `test/` is gitignored, so a fresh clone has neither the tables
    # nor a loader for them (docs/DEFERRED.md records that as an open Phase 5
    # question); this notebook therefore parses the `.dat` itself in eight
    # lines and never imports anything out of `test/`.
    reference_dir: str = "test/enjl/reference"
    reference_window: tuple = (0.10, 0.45)   # fm^-3, where to compare

    def n_B_grid(self):
        lo, hi, count = self.n_B
        return np.linspace(lo, hi, count)

    def conditions(self, mode):
        """Only the fractions THIS mode takes, under the library's names."""
        taken = {"beta_eq_neutrinoless": (),
                 "beta_eq_neutrino_trapped": ("Y_Le", "Y_Lmu"),
                 "fixed_YC": ("Y_C",),
                 "fixed_YC_YS": ("Y_C", "Y_S"),
                 "cfl": ()}[mode]
        return {name: getattr(self, name) for name in taken
                if getattr(self, name) is not None}

    def axes(self, mode, thermal_value):
        """`eos_table`'s `axes` argument, at ONE thermal value.

        The hadronic and quark notebooks pass a thermal grid here. This model
        will not take one: its table is a density CONTINUATION, each point
        warm-started from its neighbour, and a second temperature or a second
        fraction restarts that rather than adding a column — so it raises and
        says to call once per value. This method therefore takes the value,
        and the cells below loop.
        """
        axes = {"nB": self.n_B_grid(), self.thermal: np.array([thermal_value])}
        for name, value in self.conditions(mode).items():
            axes[name] = np.array([value])
        return axes


KNOBS = Knobs()
KNOBS

# %% [markdown]
# ## 2. Reporting a gap without presenting it as a result
#
# Asking the model for something can end three different ways, and collapsing
# them would be the single most misleading thing this notebook could do.
#
# * **not supported** — the model refuses the mode, the flag, the temperature or
#   the construction, and says which. A refusal is the library's contract
#   working, not a defect, and it is never dressed up as a result.
# * **did not converge** — the solve ran and failed to converge. That is a
#   *return value*, not an exception, so no `except` clause ever sees it; it is
#   found by testing `.ok`. Calling it "not supported" would be a lie about the
#   physics, and in this model it is a common and meaningful answer: a branch
#   simply stops existing at some density, and the sweep says where.
# * **ok** — there is a number.
#
# `TypeError` is deliberately not caught. An unexpected keyword argument is this
# notebook's own bug, and a broad `except` would file it under "the model does
# not support that", where nobody would ever find it.

# %%
def run(name, call, *args, **kwargs):
    """Call one public entry point; report which of three happened.

    Returns `(status, payload)` with status in {"ok", "unsupported",
    "unconverged"}. `NotImplementedError` and `ValueError` are the two a
    refusal uses; anything else is left to propagate.
    """
    try:
        result = call(*args, **kwargs)
    except (NotImplementedError, ValueError) as err:
        print(f"  [{name}] not supported: {err}")
        return "unsupported", None

    if not getattr(result, "ok", True):
        print(f"  [{name}] did not converge: {getattr(result, 'message', '')}")
        return "unconverged", result
    return "ok", result


def header(title, mode=None):
    """One printed header, so a skipped set is visible in the output."""
    if mode is None:
        print(f"\n=== {title} ===")
    else:
        print(f"\n=== {title} — mode={mode} {KNOBS.conditions(mode)} "
              f"leptons={KNOBS.leptons} ===")


def parameters_for(name):
    """The published set the knobs ask for. Parameters are arguments."""
    return enjl.Parameters.named(name)


def flags_for():
    """The species flags the knobs ask for; raises where this model fixes one,
    which is the refusal `run` reports."""
    return enjl.SpeciesFlags(**KNOBS.species)


# %% [markdown]
# ### Which sectors are the caller's, and which the model fixes
#
# Each named sector, offered on its own. Four of the six are fixed by the model
# — the composition is the paper's `(p, n, Lambda, u, d, s, e, mu)` and is not
# configurable — and the refusal says which way each is fixed and why. That is
# section 4 of `CLAUDE.md` working: a sector that is off is off because its flag
# says so, never because the model happened not to look at it.
#
# The two that *are* the caller's are the thermal sectors that carry no
# conserved charge and so enter `eps`, `P` and `s` and no equation of the solve.

# %%
header("species flags")
for flag in KNOBS.species:
    for value in (True, False):
        one = dict(KNOBS.species)
        one[flag] = value
        if one == KNOBS.species:
            continue

        def probe(flag_values=one):
            return enjl.SpeciesFlags(**flag_values)

        status, _ = run(f"{flag}={value}", probe)
        if status == "ok":
            print(f"  [{flag}={value}] the caller's")

# %% [markdown]
# ## 3. Saving a table
#
# Every table generated below can be written to `output/tables/enjl/` under a
# name built from the run itself. Every choice that changes a number is in the
# name — here the parameter set and the branch as well as the mode, its
# fractions, the thermal axis, the density axis and the sectors that are on — so
# two runs cannot collide silently and a folder listing says months later how
# each file was made. The complete metadata still goes *inside* the file,
# through `save_table(meta=...)`.
#
# **`table_path` needs its `root` given here.** Its default is the relative
# string `"output/tables"`, which resolves against the *working directory*: run
# under `jupytext --execute`, that directory is `notebooks/`, and the tables
# land in `notebooks/output/` instead of the repository's `output/`. The
# bootstrap cell already found the root, so passing it is one argument and the
# ambiguity is gone.

# %%
TABLE_ROOT = str(ROOT / "output" / "tables")


def table_name(set_name, mode, axes, direction=None):
    """The automatic name. The parameter set and the branch are part of the
    model slot, because both change every number in the file."""
    slot = f"enjl-{set_name}" + (f"-{direction}" if direction else "")
    return standard_name(slot, mode, KNOBS.conditions(mode), axes,
                         KNOBS.species, leptons=KNOBS.leptons)


example = table_name("fq1.0_B1", "fixed_YC",
                     KNOBS.axes("fixed_YC", 0.0), direction="up")
print(example)
print(table_path("enjl", example, root=TABLE_ROOT))

# %% [markdown]
# ## 4. A section per mode
#
# The four equilibrium conditions of the library, each fixing which variables
# are independent:
#
# | mode | independent variables | meaning |
# |---|---|---|
# | `beta_eq_neutrinoless` | (n_B, T) | beta equilibrium, free-streaming neutrinos, charge neutral |
# | `beta_eq_neutrino_trapped` | (n_B, Y_Le, T) | beta equilibrium with trapped neutrinos |
# | `fixed_YC` | (n_B, Y_C, T) | fixed non-leptonic charge fraction — the simulation-table mode |
# | `fixed_YC_YS` | (n_B, Y_C, Y_S, T) | fixed charge and strangeness |
#
# `cfl` is not among them and asking for it refuses by name: a colour-flavour-
# locked phase is a statement about which phase a model describes, and this one
# describes matter in which the locking is not imposed.
#
# ### One point per set

# %%
POINT_N_B = 0.40
COLD = float(KNOBS.thermal_grid[0])

for mode in KNOBS.modes:
    header("one point", mode)
    conditions = KNOBS.conditions(mode)
    for set_name in KNOBS.sets:

        def solve(set_name=set_name, mode=mode, conditions=conditions):
            extra = dict(conditions)
            # The neutralizing-lepton flag applies to the fixed-fraction modes
            # and to those only: beta equilibrium is defined by the leptons, so
            # naming the flag there is a contradiction rather than a choice —
            # and this model raises on it, which is that rule with teeth.
            if mode.startswith("fixed_"):
                extra["leptons"] = KNOBS.leptons
            return enjl.eos_point(parameters_for(set_name), mode, flags_for(),
                                  n_B=POINT_N_B, T=COLD, **extra)

        status, result = run(set_name, solve)
        if status == "ok":
            p = result.point
            print(f"  [{set_name}] n_B={POINT_N_B:.3f}  T={COLD:.1f}  "
                  f"P={p.P:9.3f}  eps={p.eps:9.3f}  "
                  f"mu_B={p.mu_b:9.3f}  mu_e={p.mu_e:8.3f}")

# %% [markdown]
# Two sets printing the same numbers is not a bug: `f_q` rescales the coupling
# of the QUARKS to the vector fields, and at a density where the deconfined
# fraction `chi` is still zero there are no quarks for it to act on. The sets
# part company exactly where `chi` leaves zero, which is what the grids below
# and the branch pair of section 5 show.
#
# ### One mode refused, by name
#
# The fifth mode of the library, and the flag this model will not take in beta
# equilibrium. Both refusals are printed rather than avoided: they are the two
# places where a caller who copied a cell from the quark notebook would land.

# %%
header("refusals that are the contract working")
run("cfl", enjl.eos_point, parameters_for(KNOBS.sets[0]), "cfl", flags_for(),
    n_B=POINT_N_B, T=COLD)
run("beta_eq_neutrinoless with leptons=False", enjl.eos_point,
    parameters_for(KNOBS.sets[0]), "beta_eq_neutrinoless", flags_for(),
    n_B=POINT_N_B, T=COLD, leptons=False)

# %% [markdown]
# ### A table per set
#
# The same modes over the density grid of the knobs cell, at the cold end of the
# thermal grid. The density axis is warm-started inside the library — each
# solved point seeds the next — so a table is not a loop over `eos_point` that
# the caller could have written.
#
# `rows=True` returns the long format the table writer and the structure solver
# both read. The column names are the ones every model in this repository uses,
# plus the ones this model is about: `chi`, the fraction of the baryon density
# carried by deconfined quarks (the author's `fq` column), and the constituent
# masses `M_u`, `M_d`, `M_s`, `M_p`, `M_n`, `M_Lambda` that come out of the gap
# equation.

# %%
tables = {}
for mode in KNOBS.modes:
    header("a grid", mode)
    axes = KNOBS.axes(mode, COLD)
    for set_name in KNOBS.sets:

        def build(set_name=set_name, mode=mode, axes=axes):
            extra = ({"leptons": KNOBS.leptons} if mode.startswith("fixed_")
                     else {})
            return enjl.eos_table(parameters_for(set_name), mode, flags_for(),
                                  axes, direction="up", rows=True, **extra)

        status, rows = run(set_name, build)
        if status != "ok":
            continue
        tables[(set_name, mode, "up")] = rows
        requested = len(KNOBS.n_B_grid())
        first, last = rows[0], rows[-1]
        print(f"  [{set_name}] {len(rows):3d}/{requested} rows   "
              f"P {first['P']:8.3f} -> {last['P']:8.3f}   "
              f"eps {first['eps']:8.3f} -> {last['eps']:8.3f}   "
              f"chi {first['chi']:.3f} -> {last['chi']:.3f}")

# %% [markdown]
# Non-converged points are dropped from their line rather than aborting the
# table, so a row count below the requested count is the table saying which
# points it could not reach — not a silent truncation.
#
# One of these written out, under its automatic name:

# %%
SAVE = (KNOBS.sets[-1], "beta_eq_neutrinoless", "up")

if SAVE in tables:
    set_name, mode, direction = SAVE
    filename = table_name(set_name, mode, KNOBS.axes(mode, COLD), direction)
    path = save_table(tables[SAVE],
                      table_path("enjl", filename, root=TABLE_ROOT),
                      meta={"model": "enjl", "parameter_set": set_name,
                            "mode": mode, "direction": direction,
                            "parameters": parameters_for(set_name),
                            "species": flags_for(),
                            **KNOBS.conditions(mode)})
    print("wrote", path)

# %% [markdown]
# ## 5. The branch pair
#
# **Two branches of one functional.** The same thermodynamic potential admits
# more than one self-consistent state at the same chemical potentials: a
# chirally broken one, where the gap equation holds the light constituent masses
# up near 300 MeV, and a restored one, where they have collapsed to the current
# masses. A first-order transition in this model is a construction *between two
# solutions of one set of equations*, not a boundary between two models.
#
# `eos_table`'s `direction` selects which of them the continuation follows:
# `"up"` starts at the bottom of the density grid on the broken side, `"down"`
# starts at the top from a deconfined guess. Where both exist the two tables
# disagree at the same `n_B`, and that disagreement *is* the transition.
#
# The `enjl_branch_pair` adapter of `eos/mixed/adapters.py` — which wraps the
# same two branches as a `Phase` pair for the composite engine, and is what
# locates the coexistence — belongs to `hybrid_eos`, not to this notebook: the
# physics is ENJL's, but the machinery that constructs across a window is
# `eos/mixed`, and this notebook stays inside the model.

# %%
branches = {}
for set_name in KNOBS.sets:
    header(f"branch pair — {set_name}")
    for direction in KNOBS.directions:

        def build(set_name=set_name, direction=direction):
            return enjl.eos_table(parameters_for(set_name),
                                  "beta_eq_neutrinoless", flags_for(),
                                  {"nB": KNOBS.n_B_grid(), "T": [COLD]},
                                  direction=direction, rows=True)

        status, rows = run(f"{set_name} {direction}", build)
        if status != "ok":
            continue
        branches[(set_name, direction)] = rows
        tables[(set_name, "beta_eq_neutrinoless", direction)] = rows
        print(f"  [{set_name} {direction:4s}] {len(rows):3d} rows   "
              f"n_B {rows[0]['n_B']:.3f} -> {rows[-1]['n_B']:.3f}   "
              f"M_u {rows[0]['M_u']:7.2f} -> {rows[-1]['M_u']:7.2f} MeV")

# %% [markdown]
# Where the two branches overlap, the stable state at fixed `n_B` and T = 0 is
# the one with the **lower energy density** — no branch bookkeeping is needed to
# say which.
#
# The two continuations do not disagree everywhere they overlap. Above the
# transition they converge on the *same* root, and there the difference is at
# the level of the solver tolerance: "which branch" has stopped being a
# question, and naming a winner on the sign of a 1e-12 difference would be
# reporting round-off as physics. So a tolerance separates the two cases, and
# only the densities where the branches are genuinely two states are listed.

# %%
SAME_BRANCH = 1.0e-8            # relative difference in eps below which the
                                # two continuations have found one root

header("where the branches overlap")
splits = {}
for set_name in KNOBS.sets:
    up = branches.get((set_name, "up"), [])
    down = branches.get((set_name, "down"), [])
    if not up or not down:
        continue
    down_by_n = {round(r["n_B"], 9): r for r in down}
    overlap = [(r, down_by_n[round(r["n_B"], 9)]) for r in up
               if round(r["n_B"], 9) in down_by_n]
    distinct = [(a, b) for a, b in overlap
                if abs(a["eps"] - b["eps"]) > SAME_BRANCH * abs(a["eps"])]
    splits[set_name] = [a["n_B"] for a, _ in distinct]
    print(f" {set_name}: {len(overlap)} densities carry both branches, "
          f"{len(distinct)} of them two DISTINCT states")
    for r_up, r_down in distinct:
        d_eps = r_up["eps"] - r_down["eps"]
        winner = "up" if d_eps < 0 else "down"
        print(f"   n_B={r_up['n_B']:.3f}  eps_up={r_up['eps']:9.3f}  "
              f"eps_down={r_down['eps']:9.3f}  "
              f"delta={d_eps:+9.4f}  stable: {winner}")

# %% [markdown]
# A sign change down that `delta` column is a first-order transition, and the
# two densities it happens between bracket the window a Maxwell construction
# would replace by a constant-pressure segment.

# %% [markdown]
# ## 6. The author's tables, reproduced
#
# `test/enjl/reference/` holds five beta-equilibrium tables produced by the
# author's own Maple implementation — the code that made the paper's figures, so
# they pin the model far more tightly than the two or three significant figures
# the paper prints. They are **golden references**: code that disagrees with
# them is wrong until proven otherwise, so the residual is printed rather than
# summarised as "agrees".
#
# Four things about the columns matter and are handled below:
#
# * `E` is the energy density with the vacuum term `E0` already subtracted, so
#   it is directly our `eps`. `epa` is energy *per baryon* and includes the rest
#   mass.
# * **`munr`, not `mun`, is the baryon chemical potential.** They agree while
#   baryons are present and part company by hundreds of MeV once the baryons
#   have dissolved, where `mun` is the vanishing neutron's own potential.
# * A blank `munr` marks a row that is **linear interpolation across a Maxwell
#   plateau** rather than solver output — 203 of the 383 rows of the `fq0.5_B1`
#   file. Quantitative comparison on those rows is meaningless and they are
#   masked out.
# * A handful of densities are **off-grid**: they are the author's own
#   coexistence endpoints, so the comparison matches on the density values in
#   the file rather than on a grid of its own.
#
# `test/` is gitignored, so a fresh clone has neither these files nor a loader
# for them — an open Phase 5 question in `docs/DEFERRED.md`. This notebook
# therefore parses the `.dat` itself and reports their absence as a message
# rather than a traceback, and it imports nothing out of `test/`.

# %%
REFERENCE_FILES = {"fq0.5_B1": "Beta_fq0.5_B1.dat",
                   "fq0.7_B0": "Beta_fq0.7_B0.dat",
                   "fq0.7_B1": "Beta_fq0.7_B1.dat",
                   "fq1.0_B0": "Beta_fq1.0_B0.dat",
                   "fq1.0_B1": "Beta_fq1.0_B1.dat"}

REFERENCE_DIR = ROOT / KNOBS.reference_dir


def load_author_table(path):
    """One Maple table as {column: array}, solved rows only.

    Tab separated with one header line; Maple writes stray tabs, so unnamed
    columns are dropped, and `--` (its marker for a quantity with no value)
    becomes nan. A blank `munr` marks an interpolated plateau row.
    """
    with open(path) as handle:
        names = [c.strip() for c in handle.readline().rstrip("\n").split("\t")]
    raw = np.genfromtxt(path, delimiter="\t", skip_header=1,
                        missing_values="--", filling_values=np.nan)
    col = {name: raw[:, i].astype(float) for i, name in enumerate(names)
           if name and not name.startswith("Derivative")}
    solved = np.isfinite(col["nB"]) & np.isfinite(col["munr"])
    return {key: value[solved] for key, value in col.items()}


header("author tables")
if not REFERENCE_DIR.is_dir():
    print(f"  {REFERENCE_DIR} is absent — `test/` is gitignored, so a fresh "
          f"clone has neither the author's tables nor a loader for them "
          f"(docs/DEFERRED.md, open). Nothing below this cell runs.")
    author = {}
else:
    author = {}
    for set_name, filename in REFERENCE_FILES.items():
        path = REFERENCE_DIR / filename
        if not path.is_file():
            print(f"  [{set_name}] {filename} absent")
            continue
        author[set_name] = load_author_table(path)
        n = author[set_name]["nB"]
        print(f"  [{set_name}] {len(n):3d} solved rows, "
              f"n_B {n[0]:.2f} -> {n[-1]:.2f} fm^-3")

# %% [markdown]
# ### The residual
#
# Our beta-equilibrium continuation, solved at the author's **own** densities so
# nothing is interpolated on either side, compared column by column. The window
# is the knobs cell's `reference_window`: it stops below the first coexistence
# endpoint of every set, because past that endpoint the author's table follows a
# construction and our `direction="up"` table follows a branch, and comparing
# those two would be comparing different physical objects rather than measuring
# an implementation.
#
# `max` is what the reference has to be judged on; the median is printed beside
# it because the maximum is set by the lowest densities, where `P` is a fraction
# of a MeV/fm^3 and a relative residual on it is a hard test of an absolute
# agreement of order 1e-6 MeV/fm^3.

# %%
COMPARE = ("P", "eps", "mu_B", "mu_e")
AUTHOR_COLUMN = {"P": "P", "eps": "E", "mu_B": "munr", "mu_e": "mue"}

residuals = {}
lo, hi = KNOBS.reference_window
APPROACH = 0.40                 # below this every set is clear of its own
                                # first coexistence endpoint
header(f"residual against the author, n_B in [{lo}, {hi}] fm^-3")
for set_name, col in author.items():
    grid = col["nB"]
    window = (grid >= lo) & (grid <= hi)

    def build(set_name=set_name, window=window, grid=grid):
        return enjl.eos_table(parameters_for(set_name),
                              "beta_eq_neutrinoless", flags_for(),
                              {"nB": grid[window], "T": [0.0]},
                              direction="up", rows=True)

    status, rows = run(set_name, build)
    if status != "ok":
        continue
    ours = {round(r["n_B"], 9): r for r in rows}

    errors = {key: [] for key in COMPARE}
    densities = []
    for index in np.where(window)[0]:
        n_B = round(grid[index], 9)
        if n_B not in ours:
            continue
        densities.append(n_B)
        for key in COMPARE:
            theirs = col[AUTHOR_COLUMN[key]][index]
            errors[key].append(abs(ours[n_B][key] / theirs - 1.0))
    residuals[set_name] = (np.array(densities),
                           {k: np.array(v) for k, v in errors.items()})

    # The last densities of the window are the approach to that set's own
    # coexistence endpoint, where the author's table is already following a
    # construction and ours is still following a branch. Both numbers are
    # printed so the difference between the two is visible rather than chosen.
    clear = np.array(densities) <= APPROACH
    print(f" [{set_name}] {len(densities)}/{int(window.sum())} densities "
          f"reproduced")
    for key in COMPARE:
        err = residuals[set_name][1][key]
        print(f"   {key:5s}  max {err.max():.2e} at n_B = "
              f"{densities[int(np.argmax(err))]:.3f}   median "
              f"{np.median(err):.2e}   max below {APPROACH} fm^-3: "
              f"{err[clear].max():.2e}")

# %% [markdown]
# ### Row by row, at one set
#
# The numbers themselves, so the residual above can be read against the values
# it is a fraction of.

# %%
SHOW = KNOBS.sets[-1]

if SHOW in author and SHOW in residuals:
    col = author[SHOW]
    densities, _ = residuals[SHOW]
    rows = enjl.eos_table(parameters_for(SHOW), "beta_eq_neutrinoless",
                          flags_for(), {"nB": densities, "T": [0.0]},
                          direction="up", rows=True)
    ours = {round(r["n_B"], 9): r for r in rows}
    header(f"row by row — {SHOW}")
    print(f"  {'n_B':>6s} {'P (ours)':>12s} {'P (author)':>12s} {'rel':>10s}"
          f" {'eps (ours)':>12s} {'eps (author)':>13s} {'rel':>10s}")
    for index, n_B in enumerate(col["nB"]):
        key = round(n_B, 9)
        if key not in ours or index % 6:
            continue
        r = ours[key]
        print(f"  {n_B:6.3f} {r['P']:12.6f} {col['P'][index]:12.6f} "
              f"{abs(r['P'] / col['P'][index] - 1):10.2e} "
              f"{r['eps']:12.5f} {col['E'][index]:13.5f} "
              f"{abs(r['eps'] / col['E'][index] - 1):10.2e}")

# %% [markdown]
# ## 7. Temperature
#
# The model is closed at any T >= 0 and accepts entropy per baryon wherever it
# accepts a temperature. What is **not** closed is the CONSTRUCTION above T = 0,
# and `docs/DEFERRED.md` says why: locating a coexistence at T > 0 equates the
# Gibbs free energies of the two branches rather than `P` and `mu_B` alone, so
# the entropy enters the coexistence bookkeeping and the plateau's lever rule
# has to average it too.
#
# Three things are asked for below and the answers are three different kinds:
# the warm branch table, which is a result; a two-temperature axis, which is
# refused with an instruction; and the constructed table at T > 0, which is the
# gap. None of them is worked around.

# %%
WARM = float(KNOBS.thermal_grid[-1])
warm_species = dict(KNOBS.species, photons=True, thermal_neutrinos=True)

header(f"a warm branch — T = {WARM} MeV")
for set_name in KNOBS.sets:

    def build(set_name=set_name):
        return enjl.eos_table(parameters_for(set_name),
                              "beta_eq_neutrinoless",
                              enjl.SpeciesFlags(**warm_species),
                              {"nB": KNOBS.n_B_grid(), "T": [WARM]},
                              direction="up", rows=True)

    status, rows = run(set_name, build)
    if status == "ok":
        tables[(set_name, "beta_eq_neutrinoless", f"up T={WARM}")] = rows
        print(f"  [{set_name}] {len(rows):3d} rows   "
              f"P {rows[0]['P']:8.3f} -> {rows[-1]['P']:8.3f}   "
              f"S/B {rows[0]['S_per_B']:.4f} -> {rows[-1]['S_per_B']:.4f}")

# %%
header("what a temperature axis and a construction do")
par = parameters_for(KNOBS.sets[-1])

run("two temperatures in one table", enjl.eos_table, par,
    "beta_eq_neutrinoless", flags_for(),
    {"nB": KNOBS.n_B_grid(), "T": [COLD, WARM]}, rows=True)

run(f"constructed table at T = {WARM} MeV", build_constructed_table,
    TableSpec(nB=KNOBS.n_B_grid(), par=par, T=WARM, species=flags_for()),
    coexistences=[])

run("eos_response", enjl.eos_response, par, "beta_eq_neutrinoless",
    flags_for(), n_B=POINT_N_B, T=COLD)

# %% [markdown]
# The constructed table at T = 0 needs the located windows as an **argument** —
# a model may not import a composite engine, and locating a coexistence needs
# both branches at once, which is `eos.mixed.construction.enjl_coexistences`.
# That call belongs to `hybrid_eos`. With an empty window list the same entry
# point keeps at each density whichever branch has the lower energy density,
# which is the object section 5 assembled by hand.
#
# That is the stable **pure** phase, and it is the stable state only where the
# branches do not cross — `fq1.0_B1`, the set plotted here, is one where they
# do not. Where they do, the minimum of two convex `eps(n_B)` curves is concave
# at the crossing, so `mu_B` jumps down and `P = mu_B n_B - eps` falls with it:
# on this same grid `fq0.5_B1` and `fq0.7_B1` lose 34.6 and 24.5 MeV/fm³ of
# pressure that way. The result therefore carries `deliverable`, and a table
# bound for a structure solver is tested before it goes:

# %%
header("the stable branch, assembled by the library")
status, constructed = run(
    "constructed, no located windows", build_constructed_table,
    TableSpec(nB=KNOBS.n_B_grid(), par=par, T=COLD, species=flags_for()),
    coexistences=[])
if status == "ok":
    rows = constructed.rows
    tables[(KNOBS.sets[-1], "beta_eq_neutrinoless", "stable")] = rows
    print(f"  {len(rows)} rows   P {rows[0]['P']:8.3f} -> "
          f"{rows[-1]['P']:8.3f}")
    print(f"  deliverable: {constructed.deliverable}"
          + (f"  ({constructed.defect})" if constructed.defect else ""))
    # Only over the densities where the two branches are genuinely two states:
    # where they have converged on one root the `branch` label records which
    # continuation happened to supply the row and changes on round-off.
    distinct = set(round(n, 9) for n in splits.get(KNOBS.sets[-1], []))
    picked = [(r["n_B"], r.get("branch")) for r in rows
              if round(r["n_B"], 9) in distinct]
    print(f"  branch kept where the two are distinct: "
          + ", ".join(f"{n:.3f}:{b}" for n, b in picked))

# %% [markdown]
# ## 8. Figures
#
# All styling comes from `eos.general.figure_style` and nothing else, and every
# figure is written to `output/enjl/` — anchored on the same `ROOT` the table
# path is, for the same reason.
#
# The lines are the tables computed above, re-read rather than re-solved, so a
# figure cannot silently disagree with the numbers printed beside it.

# %%
import matplotlib.pyplot as plt

from eos.general import figure_style as fs

fs.set_paper_style(fontsize=10, labelsize=9, legendsize=8)

FIG_DIR = ROOT / "output" / "enjl"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SET_COLOR = dict(zip(KNOBS.sets, fs.OKAB_CAT))


def line(key, *columns):
    """Columns of one stored table, as arrays. Empty where nothing solved."""
    rows = tables.get(key, [])
    if not rows:
        return tuple(np.empty(0) for _ in columns)
    return tuple(np.array([r[c] for r in rows], dtype=float)
                 for c in columns)


print(f"figures into {FIG_DIR}")

# %% [markdown]
# ### 8.1 The branch pair, in the two planes a structure solver reads
#
# Solid is the `"up"` continuation from the chirally broken side, dashed the
# `"down"` continuation from the deconfined guess. Colour is the parameter set.
# Where a dashed line runs beside a solid one at the same density, both branches
# exist there and the transition is between them.

# %%
fig, axes = fs.paper_grid("1x2", "double", aspect=1.2, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_P, ax_eP = axes.ravel()

for set_name in KNOBS.sets:
    for direction, style in (("up", "-"), ("down", "--")):
        n, P = line((set_name, "beta_eq_neutrinoless", direction), "n_B", "P")
        if n.size:
            ax_P.plot(n, P, style, color=SET_COLOR[set_name],
                      label=(set_name if direction == "up" else None))
ax_P.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_P.set_ylabel(r"$P$ [MeV/fm$^3$]")
ax_P.legend(loc="upper left", title="solid: up / dashed: down")
fs.apply_style(ax_P, legend=False)
fs.panel_label(ax_P, "(a)", corner="lower right")

for set_name in KNOBS.sets:
    for direction, style in (("up", "-"), ("down", "--")):
        P, eps = line((set_name, "beta_eq_neutrinoless", direction),
                      "P", "eps")
        if P.size:
            ax_eP.plot(P, eps, style, color=SET_COLOR[set_name])
ax_eP.set_xlabel(r"$P$ [MeV/fm$^3$]")
ax_eP.set_ylabel(r"$\epsilon$ [MeV/fm$^3$]")
fs.apply_style(ax_eP, legend=False)
fs.panel_label(ax_eP, "(b)", corner="upper left")

fs.save_figure(fig, str(FIG_DIR / "enjl_branch_pair"))
plt.show()

# %% [markdown]
# ### 8.2 The gap equation, and the speed of sound
#
# **(a)** The constituent masses the gap equation returns along the `"up"`
# branch. The collapse of `M_u` and `M_d` from ~300 MeV to the current mass is
# chiral restoration, and it happens *inside* the model rather than at a
# hand-placed boundary — which is the whole content of "one functional".
#
# **(b)** The sound speed. `eos_response` refuses for this model, and the cell
# above printed its reason: the second derivatives it would return need the
# branch the derivative is taken along to be settled, and above the first
# transition more than one branch satisfies the equilibrium conditions at the
# same density. So the curve here is a **finite difference along one named
# branch**, `dP/d(eps)` down the `"up"` continuation at T = 0, and it is
# labelled `cs2_adiabatic`: the composition re-equilibrates at every point and
# the entropy per baryon is zero all along the line, which at T = 0 is the same
# derivative the isothermal one would be. The library never returns a bare
# `cs2` and neither does this notebook.

# %%
fig, axes = fs.paper_grid("1x2", "double", aspect=1.2, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_M, ax_cs = axes.ravel()

MASS_KEY = {"u": "M_u", "d": "M_d", "s": "M_s"}
MASS_SET = KNOBS.sets[-1]

for flavour, key in MASS_KEY.items():
    n, M = line((MASS_SET, "beta_eq_neutrinoless", "up"), "n_B", key)
    if n.size:
        color, linestyle = fs.particle_style(flavour)
        ax_M.plot(n, M, color=color, linestyle=linestyle,
                  label=rf"$M_{flavour}$")
ax_M.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_M.set_ylabel(r"$M_q$ [MeV]")
ax_M.set_title(MASS_SET)
ax_M.legend(loc="upper right")
fs.apply_style(ax_M, legend=False)
fs.panel_label(ax_M, "(a)", corner="lower left")

for set_name in KNOBS.sets:
    eps, P = line((set_name, "beta_eq_neutrinoless", "up"), "eps", "P")
    n, = line((set_name, "beta_eq_neutrinoless", "up"), "n_B")
    if eps.size < 3:
        continue
    cs2_adiabatic = np.gradient(P, eps)
    ax_cs.plot(n, cs2_adiabatic, color=SET_COLOR[set_name], label=set_name)
ax_cs.axhline(1.0, color="0.6", lw=0.6, zorder=0)
ax_cs.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_cs.set_ylabel(r"$c_{s,\mathrm{adiabatic}}^2$  [$c^2$]")
ax_cs.legend(loc="upper left")
fs.apply_style(ax_cs, legend=False)
fs.panel_label(ax_cs, "(b)", corner="lower right")

fs.save_figure(fig, str(FIG_DIR / "enjl_masses_and_cs2"))
plt.show()

# %% [markdown]
# The `"up"` branch is a raw continuation and may cross `dP/dn_B < 0` — the
# spike in panel (b) is where the branch jumps rather than a physical sound
# speed. That is real physics being reported honestly: a mechanically unstable
# branch is what a first-order transition looks like before a construction
# replaces the window, and section 7's stable-branch assembly is the object that
# resolves it before a table reaches a structure solver.
#
# ### 8.3 Composition, and the residual against the author
#
# **(a)** The particle fractions `Y_i = n_i / n_B` along the `"up"` branch:
# baryons, quarks and leptons in one panel, coloured through
# `figure_style.particle_style` so a quark curve is the same colour here as
# anywhere else in the repository.
#
# **(b)** The residual of section 6 in `P`, per set, on a log axis — all five
# reference sets, not only the three the notebook swept. This is the figure the
# golden references earn: the agreement falls as `P` grows, because the
# residual is an absolute agreement of order 1e-6 MeV/fm^3 divided by a
# pressure that starts below 1, and it turns back up at the top of the window
# where that set's own coexistence endpoint is being approached.

# %%
fig, axes = fs.paper_grid("1x2", "double", aspect=1.2, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_Y, ax_res = axes.ravel()

COMP_SET = KNOBS.sets[-1]
COMP_SPECIES = (("p", "n_p"), ("n", "n_n"), ("Lambda", "n_Lambda"),
                ("u", "n_u"), ("d", "n_d"), ("s", "n_s"),
                ("e-", "n_e"), ("mu-", "n_mu"))

n, = line((COMP_SET, "beta_eq_neutrinoless", "up"), "n_B")
for label, column in COMP_SPECIES:
    n_i, = line((COMP_SET, "beta_eq_neutrinoless", "up"), column)
    if not n_i.size:
        continue
    color, linestyle = fs.particle_style(label)
    ax_Y.plot(n, n_i / n, color=color, linestyle=linestyle,
                 label=label)
ax_Y.set_yscale("log")
ax_Y.set_ylim(1e-4, 5.0)
# The decade labels go through `figure_style.log_decades`. Matplotlib writes
# its own log tick labels as mathtext and resolves them through the TEXT font,
# which is CMU Serif and has no U+2212 glyph, so the exponent's minus comes out
# a hollow box — the one place the paper style's ASCII-minus rcParam cannot
# reach. This is the repository's guard for it and not a local workaround.
fs.log_decades(ax_Y, "y")
ax_Y.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_Y.set_ylabel(r"$Y_i = n_i / n_B$")
ax_Y.set_title(COMP_SET)
ax_Y.legend(loc="lower right", ncol=2)
fs.apply_style(ax_Y, legend=False)
fs.panel_label(ax_Y, "(a)", corner="upper left")

REFERENCE_COLOR = dict(zip(sorted(REFERENCE_FILES), fs.OKAB_CAT))

if residuals:
    for set_name in sorted(residuals):
        densities, errors = residuals[set_name]
        ax_res.plot(densities, errors["P"], "-",
                    color=REFERENCE_COLOR[set_name], label=set_name)
    ax_res.set_yscale("log")
    fs.log_decades(ax_res, "y")
    ax_res.legend(loc="lower left", ncol=2)
else:
    ax_res.text(0.5, 0.5, "author tables absent", ha="center", va="center",
                transform=ax_res.transAxes)
ax_res.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_res.set_ylabel(r"relative residual in $P$")
fs.apply_style(ax_res, legend=False)
fs.panel_label(ax_res, "(b)", corner="upper right")

fs.save_figure(fig, str(FIG_DIR / "enjl_composition_and_residual"))
plt.show()

# %% [markdown]
# ## 9. The step-by-step treatment, and where the parallel stops
#
# `quark_eos.py` walks `njl` and `ccdm` through five steps: the model with no
# colour superconductivity; the same point with pairing switched on, one pattern
# at a time; unpaired against 2SC against CFL at fixed `(mu_B, T)`; the gap
# `Delta(n_B, T)` mapped per pattern; and the quantities that go with it. The
# same five questions are put to this model below. It answers three of them,
# answers one with a different object, and has nothing to answer the fifth with
# — which is the result, and is written out rather than smoothed into a parallel
# that is not there:
#
# | step, in the quark notebook | ENJL |
# |---|---|
# | 1. no pairing: parameters, gap equations, potential, one point | **has it** — and there is no second configuration to contrast it with, because there is only one |
# | 2. the same point with pairing on, one pattern at a time | **has nothing here.** No diquark channel in the functional: no gap, no 2SC, no CFL |
# | 3. unpaired vs 2SC vs CFL at fixed `(mu_B, T)` | **a different object, the same question.** The states this model has more than one of at fixed `(mu_B, T)` are the branch pair, and the criterion that picks between them is the same one |
# | 4. `Delta(n_B, T)` as a map, per pattern | **no gap to map.** The order parameters are the constituent mass `M_u` (chiral) and `chi` (deconfined fraction), and they are mapped the same way |
# | 5. fractions, `c_s^2`, the phase boundary in the `(mu_B, T)` plane | **has all three** |
#
# **A branch pair is not a pairing pattern**, and step 3 is where that is easiest
# to lose. A pairing pattern is a different *ansatz*: a condensate the Lagrangian
# does not carry until it is put there, so unpaired, 2SC and CFL are three
# different sets of equations, and comparing them compares three candidate ground
# states of one Lagrangian. The two branches here are two *roots of one* set of
# stationarity conditions — nothing is added or removed between them, and which
# one comes back depends only on where the continuation started. The arithmetic
# that picks the winner is the same in both cases, which is why the step maps
# across at all. The objects it picks between are not the same kind of thing.
#
# ### 9.1 Step 1 — one point, every quantity with its symbol and its unit
#
# Parameters, the couplings they pin down *at this state*, the masses the gap
# equations return, the mean fields, the conserved-charge potentials, the
# composition and the thermodynamics — at one `(n_B, T)`, through `eos_point`
# and its result object alone.
#
# Two checks travel with it. The **scaled residual** the solve was accepted on
# says the equations named in the model's `.tex` are satisfied and not merely
# iterated; and the **Euler relation** `eps + P = T s + sum_i mu_i n_i` is
# section 8 of the repository's conventions, the invariant a wrong
# implementation fails first.

# %%
STEP_SET = KNOBS.sets[-1]
STEP_N_B = 0.40
STEP_T = 0.0

par_step = parameters_for(STEP_SET)
header(f"step 1 — one point, {STEP_SET}, n_B = {STEP_N_B} fm^-3, "
       f"T = {STEP_T} MeV")
status, step_result = run(STEP_SET, enjl.eos_point, par_step,
                          "beta_eq_neutrinoless", flags_for(),
                          n_B=STEP_N_B, T=STEP_T)

if status == "ok":
    p = step_result.point           # fm-based view
    st = p.point                    # the same state in natural units

    print("\n parameters — the numbers an inference run varies")
    print(f"   Lambda   = {par_step.Lambda:10.3f} MeV       3-momentum cutoff")
    print(f"   m_u0     = {par_step.m_u0:10.3f} MeV       current u mass")
    print(f"   m_d0     = {par_step.m_d0:10.3f} MeV       current d mass")
    print(f"   m_s0     = {par_step.m_s0:10.3f} MeV       current s mass")
    print(f"   G_S      = {par_step.GS:10.3e} MeV^-2    scalar coupling")
    print(f"   K        = {par_step.K:10.3e} MeV^-5    't Hooft determinant")
    print(f"   f_q      = {par_step.f_q:10.3f}           quark vector-coupling "
          f"rescaling")
    print(f"   f_Lambda = {par_step.f_Lambda:10.4f}           Lambda coupling, "
          f"from U_Lambda(n_0) = -30 MeV")
    print(f"   B        = {par_step.B_GeV_fm3:10.3f} GeV/fm^3  Pauli-blocking "
          f"strength")

    print("\n the couplings AT this state — functions of n_B, never stored")
    print(f"   alpha_S  = {st.alpha_S:10.4f}           structural function")
    print(f"   Gamma_w  = {st.Gw:10.3e} MeV^-2    g_omega^2 / m_omega^2")
    print(f"   Gamma_r  = {st.Gr:10.3e} MeV^-2    g_rho^2 / m_rho^2")

    print("\n the gap equations' answer — constituent masses [MeV]")
    print(f"   M_u = {p.M_q['u']:9.3f}   M_d = {p.M_q['d']:9.3f}   "
          f"M_s = {p.M_q['s']:9.3f}")
    print(f"   M_p = {p.M_b['p']:9.3f}   M_n = {p.M_b['n']:9.3f}   "
          f"M_Lambda = {p.M_b['Lambda']:9.3f}   (three-quark clusters of the "
          f"same masses)")

    print("\n the mean fields [MeV]")
    print(f"   g_omega omega_0 = {st.gomega_omega:10.3f}")
    print(f"   g_rho rho_0     = {st.grho_rho:10.3f}")
    print(f"   Sigma^R_b       = {st.SigmaR_b:10.3f}   rearrangement, baryons")
    print(f"   Sigma^R_q       = {st.SigmaR_q:10.3f}   rearrangement, quarks")

    print("\n the conserved-charge potentials [MeV]")
    print(f"   mu_B = {p.mu_b:9.3f}   mu_C = {p.mu_C:9.3f}   "
          f"mu_S = {p.mu_S:9.3f}   mu_e = {p.mu_e:9.3f}")
    print(f"   beta equilibrium: mu_C + mu_e = {p.mu_C + p.mu_e:.3e} MeV "
          f"(mu_C = mu_p - mu_n, the sign convention of the repository)")

    print("\n the composition [n_i in fm^-3, Y_i = n_i / n_B]")
    for name, n_i in p.densities.items():
        print(f"   n_{name:6s} = {n_i:10.6f}     Y_{name:6s} = "
              f"{n_i / p.n_B:9.6f}")
    chi_step = st.n_bQ / st.n_b if st.n_b > 0 else 0.0
    print(f"   chi        = {chi_step:10.6f}     baryon density carried by "
          f"deconfined quarks")

    print("\n the thermodynamics")
    print(f"   P     = {p.P:12.5f} MeV/fm^3")
    print(f"   eps   = {p.eps:12.5f} MeV/fm^3")
    print(f"   s     = {p.s:12.5f} fm^-3        S/B = "
          f"{p.s / p.n_B:.5f}")
    print(f"   E/B   = {p.EperB:12.5f} MeV          eps/n_B - 938.9 MeV, the "
          f"paper's Fig. 2 ordinate")
    print(f"   Omega/V = -P = {-p.P:9.5f} MeV/fm^3   the grand potential "
          f"density, which is what the branches are compared on")

    print("\n what closed it")
    print(f"   scaled residual = {p.error:.3e}   over the ten unknowns: the "
          f"three M_q, mu_B, mu_C,")
    print(f"                                  n_B^Q, the two mean fields and "
          f"the two rearrangement terms")

    lhs = p.eps + p.P
    rhs = p.T * p.s + sum(st.mu[name] * n_i
                          for name, n_i in p.densities.items())
    print(f"\n the Euler relation [MeV/fm^3]")
    print(f"   eps + P                 = {lhs:14.8f}")
    print(f"   T s + sum_i mu_i n_i    = {rhs:14.8f}")
    print(f"   relative difference     = {abs(lhs / rhs - 1.0):14.3e}")

# %% [markdown]
# ### 9.2 Step 2 — the pairing that is not there
#
# The quark notebook's second step switches a diquark condensate on, one pattern
# at a time. There is nothing to switch here, and the cell below is the evidence
# rather than the assertion: the parameter dataclass and the species flags are
# listed in full, and scanned for anything a pairing channel would need — a gap,
# a diquark coupling, a pattern selector.
#
# The scan finds one name and it is a false positive worth printing, because it
# is exactly the confusion this step exists to prevent: `deltas` is the
# Delta(1232) baryon resonance, a species flag every model in the repository
# carries, and has nothing to do with a gap `Delta`. This model fixes it at
# `False`, and section 2 above printed the refusal.
#
# The other half of the evidence is already on the page: `cfl` is not one of
# this model's modes, and section 4 asked for it and printed the refusal by
# name. A colour-flavour-locked phase is a statement about which phase a model
# describes; this one describes matter in which the locking is not imposed, so
# there is no pattern to select and no `Delta0` to set.

# %%
import dataclasses

header("step 2 — is there a pairing channel?")

PAIRING_WORDS = ("delta", "gap", "diquark", "pair", "csc", "2sc", "cfl")

parameter_names = [f.name for f in dataclasses.fields(enjl.Parameters)]
species_names = [f.name for f in dataclasses.fields(enjl.SpeciesFlags)]

print(f"  parameters   : {', '.join(parameter_names)}")
print(f"  species flags: {', '.join(species_names)}")

hits = [name for name in parameter_names + species_names
        if any(word in name.lower() for word in PAIRING_WORDS)]
print(f"  scanned for {PAIRING_WORDS}")
print(f"  hits: {', '.join(hits) if hits else 'none'}")
print(f"  modes this model closes: {', '.join(KNOBS.modes)}   — no 'cfl'")

# %% [markdown]
# ### 9.3 Step 3 — the two states at one `(mu_B, T)`, and which one is favoured
#
# The quark notebook compares its pairing patterns at fixed `(mu_B, T)`, where
# the state that is realised is the one of lowest grand potential. Here the two
# states at one `(mu_B, T)` are the two branches of section 5, and the criterion
# is the same: `Omega/V = -P`, so **at fixed `mu_B` and `T` the favoured branch
# is the one with the higher pressure**.
#
# That is a different criterion from the one section 5 used, and both are right
# in their own ensemble. At fixed `n_B` the stable state is the one of lower
# energy density; at fixed `mu_B` it is the one of higher pressure. Where the two
# curves `P(mu_B)` cross, the branches have equal pressure at equal `mu_B` and
# `T` — which is the **Maxwell condition**, so the crossing is the transition
# point that the density bracket of section 5 straddles.
#
# The comparison is made **only at the densities where section 5 found two
# distinct states**, under the same tolerance it used. Above the transition the
# two continuations have converged on one root — identical `mu_B` and identical
# `P` to the last digit — and a difference computed there is round-off, which
# reported as a Maxwell crossing would be twenty transitions where the model has
# one.
#
# The lines come from the tables section 5 already built; nothing is re-solved.

# %%
header("step 3 — the two branches at one (mu_B, T)")

favoured = {}
for set_name in KNOBS.sets:
    up = branches.get((set_name, "up"), [])
    down = branches.get((set_name, "down"), [])
    distinct = set(round(n, 9) for n in splits.get(set_name, []))
    if not up or not down or not distinct:
        print(f" [{set_name}] the two continuations found one root at every "
              f"shared density; there is nothing to compare")
        continue

    mu_down = np.array([r["mu_B"] for r in down])
    P_down = np.array([r["P"] for r in down])
    order = np.argsort(mu_down)

    # Whether mu_B is monotone along a branch decides how the comparison may be
    # made at all: where it is not, the branch doubles back on itself in this
    # plane and sorting by mu_B would reorder physical points.
    backwards = [d for d in KNOBS.directions
                 if np.any(np.diff([r["mu_B"]
                                    for r in branches.get((set_name, d), [])])
                           < 0.0)]
    print(f"\n [{set_name}] mu_B walks backwards along: "
          f"{', '.join(backwards) if backwards else 'neither branch'}")
    print(f" [{set_name}] the densities where the two are two states")
    print(f"   {'n_B':>6s} {'mu_B (up)':>10s} {'P up':>10s} "
          f"{'P down':>10s} {'P_up - P_down':>14s}  favoured")
    previous = None
    for r in up:
        if round(r["n_B"], 9) not in distinct:
            continue
        if not mu_down.min() <= r["mu_B"] <= mu_down.max():
            print(f"   {r['n_B']:6.3f} {r['mu_B']:10.2f} {r['P']:10.3f} "
                  f"{'—':>10s} {'—':>14s}  the down branch does not reach "
                  f"this mu_B")
            continue
        P_other = float(np.interp(r["mu_B"], mu_down[order], P_down[order]))
        delta = r["P"] - P_other
        winner = "up" if delta > 0 else "down"
        favoured[(set_name, round(r["n_B"], 9))] = winner
        print(f"   {r['n_B']:6.3f} {r['mu_B']:10.2f} {r['P']:10.3f} "
              f"{P_other:10.3f} {delta:14.4f}  {winner}")
        if previous is not None and np.sign(delta) != np.sign(previous[1]):
            print(f"   -> equal pressure between n_B = {previous[0]:.3f} and "
                  f"{r['n_B']:.3f} fm^-3 — the Maxwell condition")
        previous = (r["n_B"], delta)
    if previous is not None and all(
            v == "up" for (s, _), v in favoured.items() if s == set_name):
        print(f"   no crossing inside this window: the up branch is favoured "
              f"throughout it, and the two curves merge above it")

# %% [markdown]
# **The two criteria do not put the transition in the same grid interval, and
# that is not a disagreement.** Section 5 compares energy densities at fixed
# `n_B` and switches one grid step above where the comparison here switches, in
# every set. Both are right about their own question, and the gap between them
# is the coexistence window: inside it neither pure branch is the stable state —
# the stable state is a mixture of the two, which is what the construction
# builds — so asking which pure branch has the lower `eps` at a density inside
# the window is asking about two states that are both metastable there. The
# fixed-`mu_B` comparison is the one that returns the Maxwell point, because
# equal `P` at equal `mu_B` and `T` *is* the coexistence condition. On a grid
# this coarse the answer is an interval either way; the interval is a property
# of the density grid, not of the model.
#
# **(a)** the two branches in the plane the comparison is made in, held to the
# window where they are two states — outside it they lie on one another to the
# last digit and the panel would be one curve drawn twice over four decades of
# `P`. The turn-over on the `"up"` curve is not a plotting artefact: it is the
# swallowtail of a first-order transition, and it is why the cell above
# interpolates the `"down"` branch at the `"up"` branch's `mu_B` instead of
# sorting either curve. That cell prints which branches walk backwards in
# `mu_B`, and where one does, a sort reorders physical points and cuts the
# corner off the curve. The step is a few MeV wide — far too little to see in
# the panel, and quite enough to break a sort.
#
# **(b)** the difference itself over the densities where the two are two states,
# against `n_B` — the variable that stays single-valued through the swallowtail.
# A zero crossing there is the transition.

# %%
fig, axes = fs.paper_grid("1x2", "double", aspect=1.2, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_Pmu, ax_dP = axes.ravel()

for set_name in KNOBS.sets:
    for direction, style in (("up", "-"), ("down", "--")):
        rows = branches.get((set_name, direction), [])
        if not rows:
            continue
        # Parametric in n_B, deliberately unsorted: sorting by mu_B would cut
        # the corner off the swallowtail and invent a monotone branch.
        ax_Pmu.plot([r["mu_B"] for r in rows], [r["P"] for r in rows], style,
                    color=SET_COLOR[set_name],
                    label=(set_name if direction == "up" else None))
# The window the two branches are two states in is a small corner of the full
# curve — at the top of the grid P is thousands of MeV/fm^3 and the two lie on
# one another — so the panel is held to that corner. Everything outside it is
# one curve drawn twice.
zoom_mu, zoom_P = [], []
for set_name in KNOBS.sets:
    distinct = set(round(n, 9) for n in splits.get(set_name, []))
    for direction in KNOBS.directions:
        for r in branches.get((set_name, direction), []):
            if round(r["n_B"], 9) in distinct:
                zoom_mu.append(r["mu_B"])
                zoom_P.append(r["P"])
if zoom_mu:
    pad_mu = 0.15 * (max(zoom_mu) - min(zoom_mu))
    pad_P = 0.15 * (max(zoom_P) - min(zoom_P))
    ax_Pmu.set_xlim(min(zoom_mu) - pad_mu, max(zoom_mu) + pad_mu)
    ax_Pmu.set_ylim(min(zoom_P) - pad_P, max(zoom_P) + pad_P)

ax_Pmu.set_xlabel(r"$\mu_B$ [MeV]")
ax_Pmu.set_ylabel(r"$P$ [MeV/fm$^3$]")
ax_Pmu.legend(loc="upper left", title="solid: up / dashed: down")
fs.apply_style(ax_Pmu, legend=False)
fs.panel_label(ax_Pmu, "(a)", corner="lower right")

for set_name in KNOBS.sets:
    up = branches.get((set_name, "up"), [])
    down = branches.get((set_name, "down"), [])
    distinct = set(round(n, 9) for n in splits.get(set_name, []))
    if not up or not down or not distinct:
        continue
    mu_down = np.array([r["mu_B"] for r in down])
    P_down = np.array([r["P"] for r in down])
    order = np.argsort(mu_down)
    n_axis, delta_axis = [], []
    for r in up:
        if round(r["n_B"], 9) not in distinct:
            continue
        if not mu_down.min() <= r["mu_B"] <= mu_down.max():
            continue
        n_axis.append(r["n_B"])
        delta_axis.append(r["P"] - float(np.interp(r["mu_B"],
                                                   mu_down[order],
                                                   P_down[order])))
    if n_axis:
        ax_dP.plot(n_axis, delta_axis, "o-", ms=3,
                   color=SET_COLOR[set_name], label=set_name)
ax_dP.axhline(0.0, color="0.6", lw=0.6, zorder=0)
ax_dP.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_dP.set_ylabel(r"$P_{\rm up} - P_{\rm down}$ at equal $\mu_B$ "
                 r"[MeV/fm$^3$]")
ax_dP.legend(loc="upper right")
fs.apply_style(ax_dP, legend=False)
fs.panel_label(ax_dP, "(b)", corner="lower left")

fs.save_figure(fig, str(FIG_DIR / "enjl_branches_at_fixed_mu"))
plt.show()

# %% [markdown]
# ### 9.4 Step 4 — the order parameters, mapped
#
# There is no `Delta(n_B, T)` to map. What this model has in its place are two
# order parameters that come out of the same stationarity conditions the gap
# equations belong to:
#
# * **`M_u`**, the light constituent mass. It sits near 330 MeV where chiral
#   symmetry is broken and falls to the current mass where it is restored, so
#   `M_u / M_u(low density)` is a chiral order parameter on `[0, 1]`.
# * **`chi = n_B^Q / n_B`**, the fraction of the baryon density carried by
#   deconfined quarks — zero in purely baryonic matter, one in fully deconfined
#   matter, and already on `[0, 1]` without normalizing.
#
# Both are mapped over `(n_B, T)` below, on the `"up"` continuation, and both
# slices ticket-16's step 4 asks for are taken: against `n_B` at fixed `T`, and
# against `T` at fixed `n_B`.
#
# The set mapped is the **first** of the knobs cell rather than the last one the
# figures above use, and for a measured reason: it is the set whose `"up"` branch
# shows *both* order parameters moving over this density window. The `chi`
# column of section 4's grids is that statement already in printed form — at the
# top of the density window it falls set by set as `f_q` rises, until at the
# largest `f_q` the deconfined fraction is at the percent level along this
# branch and only the chiral parameter moves.
#
# The thermal sectors that carry no conserved charge are on for the whole map,
# `photons` and `thermal_neutrinos` together, so one flag set spans it; both are
# identically zero at `T = 0` and so change no row of the cold line.

# %%
MAP_SET = KNOBS.sets[0]
MAP_N_B = np.linspace(0.10, 1.60, 31)
MAP_T = np.linspace(0.0, 80.0, 9)          # MeV
MAP_FLAGS = enjl.SpeciesFlags(**dict(KNOBS.species, photons=True,
                                     thermal_neutrinos=True))

header(f"step 4 — the order parameters over (n_B, T), {MAP_SET}")

map_rows = {}
M_u_map = np.full((len(MAP_T), len(MAP_N_B)), np.nan)
chi_map = np.full((len(MAP_T), len(MAP_N_B)), np.nan)
mu_B_map = np.full((len(MAP_T), len(MAP_N_B)), np.nan)

for row_index, T in enumerate(MAP_T):

    def build(T=T):
        return enjl.eos_table(parameters_for(MAP_SET),
                              "beta_eq_neutrinoless", MAP_FLAGS,
                              {"nB": MAP_N_B, "T": [T]},
                              direction="up", rows=True)

    status, rows = run(f"{MAP_SET} T={T:.0f}", build)
    if status != "ok":
        continue
    map_rows[T] = rows
    column = {round(float(n), 9): index for index, n in enumerate(MAP_N_B)}
    for r in rows:
        index = column.get(round(float(r["n_B"]), 9))
        if index is None:
            continue
        M_u_map[row_index, index] = r["M_u"]
        chi_map[row_index, index] = r["chi"]
        mu_B_map[row_index, index] = r["mu_B"]
    print(f"  [T={T:5.1f} MeV] {len(rows):2d}/{len(MAP_N_B)} rows   "
          f"M_u {rows[0]['M_u']:7.2f} -> {rows[-1]['M_u']:6.2f} MeV   "
          f"chi {min(r['chi'] for r in rows):.3f} -> "
          f"{max(r['chi'] for r in rows):.3f}")

# %% [markdown]
# **Where the chiral parameter falls.** `M_u` does not fall smoothly: it drops
# by two hundred MeV between one grid density and the next, which is what a
# first-order transition looks like on a grid. So the cell below reports a
# **bracket** — the last density at which `M_u` is above half its low-density
# value and the first at which it is below — and the `mu_B` interval that
# bracket corresponds to. Interpolating across a discontinuity would invent a
# crossing density the model does not have, and the bracket's width is set by
# the grid, not by the physics.

# %%
header("step 4 — the chiral bracket, per temperature")

HALF = 0.5
boundary = []          # (T, mu_B_lo, mu_B_hi, n_B_lo, n_B_hi)
for row_index, T in enumerate(MAP_T):
    M_u = M_u_map[row_index]
    if not np.isfinite(M_u).any():
        continue
    reference = M_u[np.isfinite(M_u)][0]
    below = np.where(np.isfinite(M_u) & (M_u < HALF * reference))[0]
    if below.size == 0:
        print(f"  [T={T:5.1f} MeV] M_u stays above {HALF:.1f} M_u(low) over "
              f"the whole grid")
        continue
    first = int(below[0])
    last = first - 1
    boundary.append((T, mu_B_map[row_index, last], mu_B_map[row_index, first],
                     MAP_N_B[last], MAP_N_B[first]))
    print(f"  [T={T:5.1f} MeV] M_u {M_u[last]:6.1f} -> {M_u[first]:5.1f} MeV "
          f"between n_B = {MAP_N_B[last]:.3f} and {MAP_N_B[first]:.3f} fm^-3, "
          f"mu_B = {mu_B_map[row_index, last]:7.1f} -> "
          f"{mu_B_map[row_index, first]:7.1f} MeV")

# %% [markdown]
# **(a)** `M_u` over the `(n_B, T)` plane, the map that stands where the quark
# notebook maps `Delta`. **(b)** both order parameters against `n_B` at three
# temperatures, `M_u` normalized to its low-density value so the two share an
# axis. **(c)** the same two against `T` at three densities — the slice that
# says how much of the transition temperature moves it, which for this model
# over this window is: not much, and the numbers above say how much.

# %%
fig, axes = fs.paper_grid("1x3", "double", aspect=1.0, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_map, ax_nB, ax_T = axes.ravel()

image = ax_map.pcolormesh(MAP_N_B, MAP_T, M_u_map, shading="nearest",
                          cmap="viridis")
bar = fig.colorbar(image, ax=ax_map)
bar.set_label(r"$M_u$ [MeV]")
if boundary:
    ax_map.plot([0.5 * (lo + hi) for _, _, _, lo, hi in boundary],
                [T for T, _, _, _, _ in boundary], "w.-", lw=1.0, ms=3)
ax_map.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_map.set_ylabel(r"$T$ [MeV]")
ax_map.set_title(MAP_SET)
fs.panel_label(ax_map, "(a)", corner="upper left")

SLICE_T = [float(MAP_T[0]), float(MAP_T[len(MAP_T) // 2]), float(MAP_T[-1])]
for T in SLICE_T:
    row_index = int(np.argmin(np.abs(MAP_T - T)))
    M_u = M_u_map[row_index]
    if not np.isfinite(M_u).any():
        continue
    reference = M_u[np.isfinite(M_u)][0]
    color = fs.get_T_color(T)
    ax_nB.plot(MAP_N_B, M_u / reference, "-", color=color,
               label=rf"$T = {T:.0f}$ MeV")
    ax_nB.plot(MAP_N_B, chi_map[row_index], "--", color=color)
ax_nB.set_ylim(0.0, 1.45)
ax_nB.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_nB.set_ylabel(r"$M_u/M_u^{\rm low}$ (solid), $\chi$ (dashed)")
ax_nB.legend(loc="upper right")
fs.apply_style(ax_nB, legend=False)
fs.panel_label(ax_nB, "(b)", corner="upper left")

SLICE_N_B = [float(MAP_N_B[len(MAP_N_B) // 4]),
             float(MAP_N_B[len(MAP_N_B) // 2]),
             float(MAP_N_B[-1])]
plotted = []
for n_B in SLICE_N_B:
    index = int(np.argmin(np.abs(MAP_N_B - n_B)))
    M_u = M_u_map[:, index]
    if not np.isfinite(M_u).any():
        continue
    reference = M_u_map[np.isfinite(M_u_map[:, 0]), 0]
    scale = reference[0] if reference.size else 1.0
    ax_T.plot(MAP_T, M_u / scale, "-",
              label=rf"$n_B = {MAP_N_B[index]:.2f}$ fm$^{{-3}}$")
    ax_T.plot(MAP_T, chi_map[:, index], "--",
              color=ax_T.get_lines()[-1].get_color())
    plotted.extend([M_u / scale, chi_map[:, index]])
ax_T.set_ylim(0.0, 1.45 * max(np.nanmax(np.concatenate(plotted)), 0.05)
              if plotted else 1.45)
ax_T.set_xlabel(r"$T$ [MeV]")
ax_T.set_ylabel(r"$M_u/M_u^{\rm low}$ (solid), $\chi$ (dashed)")
ax_T.legend(loc="upper right")
fs.apply_style(ax_T, legend=False)
fs.panel_label(ax_T, "(c)", corner="upper left")

fs.save_figure(fig, str(FIG_DIR / "enjl_order_parameters"))
plt.show()

# %% [markdown]
# ### 9.5 Step 5 — the quantities that go with it
#
# The three the quark notebook's fifth step asks for, all of which this model
# has: the quark and lepton fractions, the sound speed, and the phase boundary
# in the `(mu_B, T)` plane.
#
# **The fractions** first, at three densities of the map, cold and warm. `Y_i`
# is `n_i / n_B` for every species, and `Y_C` is the non-leptonic charge
# fraction — the leptons are not in it, which is why it is not zero while the
# matter as a whole is neutral.

# %%
header(f"step 5 — composition along the map, {MAP_SET}")

FRACTION_SPECIES = (("p", "n_p"), ("n", "n_n"), ("Lambda", "n_Lambda"),
                    ("u", "n_u"), ("d", "n_d"), ("s", "n_s"),
                    ("e", "n_e"), ("mu", "n_mu"))

for T in (MAP_T[0], MAP_T[-1]):
    rows = map_rows.get(float(T), [])
    if not rows:
        continue
    print(f"\n  T = {T:.0f} MeV")
    print("   " + f"{'n_B':>6s}" + "".join(f"{'Y_' + name:>10s}"
                                           for name, _ in FRACTION_SPECIES)
          + f"{'Y_C':>10s}{'chi':>8s}")
    for r in rows[::len(rows) // 4 or 1]:
        cells = "".join(f"{r[column] / r['n_B']:10.5f}"
                        for _, column in FRACTION_SPECIES)
        print(f"   {r['n_B']:6.3f}{cells}{r['Y_C']:10.5f}{r['chi']:8.4f}")

# %% [markdown]
# **The sound speed.** `eos_response` refuses for this model — its message was
# printed in section 7 — so the curve is a finite difference `dP/d(eps)` down
# one named branch, as in figure 8.2(b).
#
# What it is *called* depends on the line it is taken along, and the two lines
# here are not the same quantity. Along the `T = 0` line the composition
# re-equilibrates at every point and the entropy per baryon is zero all along
# it, so the adiabatic and isothermal derivatives coincide and 8.2(b) labels the
# curve `cs2_adiabatic`. Along a `T > 0` line the temperature is what is held,
# so the derivative is **`cs2_isothermal`** and nothing else; at finite `T` the
# two differ by `C_P/C_V`. The library never returns a bare `cs2` and neither
# does this notebook.
#
# **The phase boundary** is the bracket of 9.4 read in the `(mu_B, T)` plane:
# the vertical bar at each temperature is the `mu_B` interval the chiral
# parameter falls across, and its height is the density grid rather than an
# uncertainty in the model.

# %%
fig, axes = fs.paper_grid("1x2", "double", aspect=1.2, placeholder=False,
                          fontsize=10, labelsize=9, legendsize=8)
ax_boundary, ax_cs = axes.ravel()

if boundary:
    for T, mu_lo, mu_hi, n_lo, n_hi in boundary:
        ax_boundary.plot([mu_lo, mu_hi], [T, T], "-",
                         color=fs.get_T_color(T), lw=1.4)
    ax_boundary.plot([0.5 * (mu_lo + mu_hi)
                      for _, mu_lo, mu_hi, _, _ in boundary],
                     [T for T, _, _, _, _ in boundary], "k.--", ms=4, lw=0.8)
    ax_boundary.set_title(MAP_SET)
else:
    ax_boundary.text(0.5, 0.5, "no chiral bracket on this grid", ha="center",
                     va="center", transform=ax_boundary.transAxes)
ax_boundary.set_xlabel(r"$\mu_B$ [MeV]")
ax_boundary.set_ylabel(r"$T$ [MeV]")
fs.apply_style(ax_boundary, legend=False)
fs.panel_label(ax_boundary, "(a)", corner="upper right")

for T in SLICE_T:
    rows = map_rows.get(float(T), [])
    if len(rows) < 3:
        continue
    n = np.array([r["n_B"] for r in rows])
    P = np.array([r["P"] for r in rows])
    eps = np.array([r["eps"] for r in rows])
    label = (rf"$T = {T:.0f}$ MeV"
             + (" (adiabatic = isothermal)" if T == 0.0 else ""))
    ax_cs.plot(n, np.gradient(P, eps), "-", color=fs.get_T_color(T),
               label=label)
ax_cs.axhline(1.0, color="0.6", lw=0.6, zorder=0)
ax_cs.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_cs.set_ylabel(r"$c_{s,\mathrm{isothermal}}^2$  [$c^2$]")
ax_cs.legend(loc="upper left")
fs.apply_style(ax_cs, legend=False)
fs.panel_label(ax_cs, "(b)", corner="lower right")

fs.save_figure(fig, str(FIG_DIR / "enjl_boundary_and_cs2"))
plt.show()

# %% [markdown]
# The spike in (b) is the same one figure 8.2(b) carries and has the same cause:
# the `"up"` continuation crosses its first-order transition rather than being
# constructed across it, and a raw branch there is mechanically unstable. It is
# reported, not smoothed; section 7's stable-branch assembly is what resolves it
# before a table reaches a structure solver.

# %% [markdown]
# ## 10. Benchmarks
#
# What this model costs, per parameter set and per configuration. Every timing
# comes from `time`/`timeit` around a **public** call, or out of the `progress`
# callback the table builder already carries — **no timing hook is added to
# library code**, and nothing below reads a solver internal.
#
# Four numbers, and they are four because they answer four different questions:
#
# * **cold point** — one `eos_point` with no warm start, which is what an
#   inference sampler pays per proposal. Best of `BENCH_REPEAT` runs, so it is
#   the cost of the call rather than the cost of the first-ever call in a
#   process. For this model it carries a second piece of information as well:
#   whether the cold point converges *at all*, which above about 0.5 fm^-3 is
#   not a given. A cold point that fails is still timed and is flagged in the
#   table; failing is not free.
# * **warm point** — the per-point cost *inside* a sweep, where each solved
#   point seeds the next: the line's `elapsed_s` divided by its `n_solved`. It
#   is the number that matters for building a table, and it is not the cold
#   number. Where a line has non-converged points their cost is in `elapsed_s`
#   but not in `n_solved`, which inflates this figure — the honest reading,
#   since a table pays for the attempts too.
# * **line wall time** — one full `n_B` line at one temperature, one fraction
#   combination and one branch, straight from the callback's `elapsed_s`.
# * **non-converged** — the count, and the `n_B` where they fall.
#   Non-convergence is a *return value*, so the benchmark counts them and keeps
#   going; it never crashes on them and never reports them as time saved.
#
# The configuration axis carries one entry this model's siblings do not have:
# the **branch**. `direction` changes every number in a table and it changes the
# cost by more than an order of magnitude, so `"up"` and `"down"` are separate
# rows rather than an average, exactly as two modes would be.
#
# The line spans 0.05 to 2.0 fm^-3, wider at the bottom than a production table
# — deliberately, since that is where a continuation started from the deconfined
# side has nothing to find and the non-convergence counter reports the real
# thing rather than a column of zeros.
#
# **This section is the expensive part of the notebook**, about a minute of the
# two the whole of it takes, and nearly all of that is the `"down"` lines and
# the profile below them. That is not an accident of the grid: it is the
# measurement.

# %%
import cProfile
import io
import pstats
import timeit

BENCH_N_B = np.linspace(0.05, 2.0, 24)
BENCH_REPEAT = 3

# (mode, T, the mode's fractions, branch). `leptons` is not in here: it is the
# knobs cell's flag, applied to the fixed-fraction modes exactly as everywhere
# above.
BENCH_CONFIGS = (("beta_eq_neutrinoless", 0.0, {}, "up"),
                 ("beta_eq_neutrinoless", 0.0, {}, "down"),
                 ("fixed_YC", 10.0, {"Y_C": KNOBS.Y_C}, "up"),
                 ("fixed_YC", 10.0, {"Y_C": KNOBS.Y_C}, "down"))


def bench_line(set_name, mode, T, conditions, direction):
    """One benchmark row: one parameter set, one configuration, one branch.

    Returns None when the model REFUSES the configuration — a refusal is not a
    slow result and does not belong in a timing table. A cold point that does
    not CONVERGE is a different thing and is kept: it is a return value, it is
    flagged, and what it cost is measured, because a sampler pays for it.
    """
    par, species = parameters_for(set_name), flags_for()
    extra = {"leptons": KNOBS.leptons} if mode.startswith("fixed_") else {}
    n_B_probe = float(np.median(BENCH_N_B))

    def one_point():
        return enjl.eos_point(par, mode, species, n_B=n_B_probe, T=T,
                              **conditions, **extra)

    status, _ = run(f"{set_name} cold point", one_point)
    if status == "unsupported":
        return None
    cold_s = min(timeit.repeat(one_point, repeat=BENCH_REPEAT, number=1))

    axes = {"nB": BENCH_N_B, "T": np.array([T])}
    for key, value in conditions.items():
        axes[key] = np.array([value])

    lines = []
    built, rows = run(f"{set_name} {direction}", enjl.eos_table, par, mode,
                      species, axes, direction=direction, rows=True,
                      progress=lines.append, **extra)
    if built != "ok" or not lines:
        return None
    info = lines[-1]          # one temperature, one fraction combination

    solved = {round(float(row["n_B"]), 9) for row in rows}
    missed = [float(x) for x in BENCH_N_B if round(float(x), 9) not in solved]
    return dict(set=set_name, mode=mode, T=T, direction=direction,
                cold_ms=1e3 * cold_s,
                cold_ok=(status == "ok"),
                warm_ms=1e3 * info["elapsed_s"] / max(info["n_solved"], 1),
                line_s=info["elapsed_s"],
                n_solved=info["n_solved"],
                n_requested=info["n_requested"],
                missed=missed)


benchmarks = []
for mode, T, conditions, direction in BENCH_CONFIGS:
    header(f"benchmark — {direction}", mode)
    for set_name in KNOBS.sets:
        row = bench_line(set_name, mode, T, conditions, direction)
        if row is not None:
            benchmarks.append(row)
            print(f"  [{set_name:9s} {direction:4s}] cold "
                  f"{row['cold_ms']:8.3f} ms{'' if row['cold_ok'] else '*'}   "
                  f"warm {row['warm_ms']:8.3f} ms/pt   "
                  f"line {row['line_s']:7.3f} s   "
                  f"{row['n_solved']}/{row['n_requested']} points")

# %% [markdown]
# A `*` on the cold column marks a cold point that did **not** converge; its
# time is what the failure cost, not what a result cost.
#
# **The two branches are not two spellings of one cost.** The `"up"` line and
# the `"down"` line solve the same equations over the same densities and differ
# by where the continuation starts, and their wall times differ by more than an
# order of magnitude. Nothing about the solved points changed: the `"down"` line
# spends its clock on the points it never solves, each retried through halved
# steps back towards the last solved point. Those attempts are in `elapsed_s`
# and not in `n_solved`, which is why its `warm` column sits far above its own
# cold point. That is the honest arithmetic for anyone budgeting a table, which
# pays for the attempts too.
#
# ### 10.1 Where the cold start runs out
#
# The cold column above is one density. This model's `eos_point` docstring says
# the parameter-free starting points stop converging around 0.5 fm^-3, and that
# is worth measuring rather than quoting, because it is the reason a table here
# is a continuation and not a loop over `eos_point`: **above that density the
# warm start is not an optimization, it is what makes the point reachable.**

# %%
header("cold start, density by density")
COLD_SCAN = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.50, 2.00)

for set_name in KNOBS.sets:
    marks, timings = [], []
    for n_B in COLD_SCAN:
        started = timeit.default_timer()
        result = enjl.eos_point(parameters_for(set_name),
                               "beta_eq_neutrinoless", flags_for(),
                               n_B=n_B, T=0.0)
        timings.append(1e3 * (timeit.default_timer() - started))
        marks.append("ok" if result.ok else "--")
    print(f"  [{set_name:9s}] " + "  ".join(f"{n:.2f}:{m}"
                                            for n, m in zip(COLD_SCAN, marks)))
    print(f"  {'':11s} " + "  ".join(f"{t:7.1f}ms" for t in timings))

# %% [markdown]
# A `--` is a cold point the solver could not reach from any of its
# parameter-free starts. It is a *return value* — `.ok` is False, nothing
# raised — and the row beneath shows what it cost to find that out.
#
# What the scan measures is **a band, not a ceiling**, and the band sits where
# the branches do: the cold starts fail somewhere between 0.6 and 1.0 fm^-3,
# set by set, and converge again above it, where the restored branch is the only
# root left and a cold start lands on it. That is a sharper statement than the
# docstring's, and it is the same physics as section 5's: the densities a cold
# start cannot reach are the ones at which the model has more than one state and
# no continuation to say which.
#
# The cost column is not a clean split between the two outcomes either. A `--`
# costs what it takes to try every start and give up; a converged point that
# only the last start reaches costs the same, which is why the slowest entry in
# some rows is an `ok`. Neither number is the cost of the *warm* point beside it
# in the table above, which is the whole reason the two are reported apart.
#
# ### 10.2 Where the line did not converge
#
# The count and the densities, per set, configuration and branch. A line that
# solved every requested point says so; nothing is inferred from a row count
# alone.

# %%
header("non-converged points")
for row in benchmarks:
    missed = row["missed"]
    label = (f"  [{row['set']:9s} {row['mode']:20s} T={row['T']:4.1f} "
             f"{row['direction']:4s}]")
    if not missed:
        print(f"{label} 0 of {row['n_requested']}")
        continue
    shown = ", ".join(f"{x:.3f}" for x in missed[:8])
    more = "" if len(missed) <= 8 else f", ... (+{len(missed) - 8})"
    print(f"{label} {len(missed)} of {row['n_requested']}  "
          f"at n_B = {shown}{more} fm^-3")

# %% [markdown]
# **Every miss is on a `"down"` line, and the `"up"` lines miss nothing** over
# the same grid. Most of them are the bottom of the line — the densities at
# which a continuation started from the deconfined side has no state to continue
# to. That is the same physics the branch pair of section 5 is about, seen from
# the solver's end: the branch does not merely become unfavourable there, it
# stops existing.
#
# One line is different and the densities are printed so that it shows: the
# lowest-`f_q` set at fixed `Y_C` misses an **interior band** as well as the
# bottom two points. Reported, not diagnosed — an interior hole is worth a look
# before anyone leans on that line, and a mechanism from here would be a guess.
#
# ### 10.3 Bottlenecks
#
# `cProfile` over one representative line, profiled *after* the benchmark cells
# so first-touch caches are warm. This model ships no jitted kernel, so a cold
# profile here would not report a compiler the way a `T = 0` hadronic line does
# — the reason for the ordering is the caches alone.
#
# **Twenty entries rather than fifteen**, and deliberately: the first fifteen by
# cumulative time are all SciPy's, and the model's own frames start below them.
# A top-15 here would print the least informative half of the answer.

# %%
PROFILE = (KNOBS.sets[-1], "beta_eq_neutrinoless", 0.0, {}, "down")

profile_set, profile_mode, profile_T, profile_conditions, profile_dir = PROFILE
profile_axes = {"nB": BENCH_N_B, "T": np.array([profile_T])}
for key, value in profile_conditions.items():
    profile_axes[key] = np.array([value])
profile_extra = ({"leptons": KNOBS.leptons}
                 if profile_mode.startswith("fixed_") else {})

profiler = cProfile.Profile()
profiler.enable()
enjl.eos_table(parameters_for(profile_set), profile_mode, flags_for(),
               profile_axes, direction=profile_dir, **profile_extra)
profiler.disable()

report = io.StringIO()
pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(20)
print(f"=== cProfile — {profile_set} {profile_mode} T={profile_T} "
      f"{profile_dir}, by cumulative time ===")
print(report.getvalue())

# The chain the cumulative list ends on continues below its cut; sorting the
# same profile by INTERNAL time names the leaf the time is actually spent in,
# which is the other half of the reading and is not visible in the list above.
internal = io.StringIO()
pstats.Stats(profiler, stream=internal).sort_stats("tottime").print_stats(8)
print("=== the same profile, by internal time ===")
print(internal.getvalue())

# %% [markdown]
# **Reading it** (for the default selection, the `"down"` branch in beta
# equilibrium at `T = 0`): the line is a bounded least-squares solve whose
# Jacobian is built by finite differences, and that Jacobian is where the time
# goes. Most of the cumulative time sits under `scipy.optimize.least_squares` →
# `trf_bounds` → `approx_derivative`, and what `approx_derivative` spends it on
# is the model's own `solver.residual` → `state_at` → `thermodynamics`, the
# constituent masses and the kinetic integrals rebuilt from scratch on every
# residual call. In the internal-time list `state_at` is the largest entry that
# belongs to this repository at all; the one entry above it is SciPy's own
# trust-region bookkeeping, which is the other side of the same finding.
#
# The call counts are the other half of the reading, and they are the reason
# this is the expensive branch: of order **eighty thousand residual evaluations
# for twenty-four requested densities**, several thousand per attempted point,
# because each Jacobian costs one residual per unknown and there are ten
# unknowns — before the retries that the unsolved points at the bottom of the
# line each trigger. Section 10.4 is the direct consequence: no analytic
# Jacobian ships for this model, so every one of those evaluations is a
# difference quotient.
#
# ### 10.4 Reference against fast backend
#
# Section 9 of the repository's conventions asks for the two flavours side by
# side **where a model ships one**. This one does not, and that is checked below
# rather than asserted: no `eos/enjl/backends/`, and no backend switch on
# `eos_point` or `eos_table`. The reference NumPy/SciPy path is the only path,
# so every number above is a reference number and needs no second column.

# %%
import inspect

header("backends — is there a fast flavour to compare against?")
has_dir = (ROOT / "eos" / "enjl" / "backends").is_dir()
for entry in (enjl.eos_point, enjl.eos_table):
    switches = [p for p in inspect.signature(entry).parameters
                if p in ("analytic_jac", "backend", "fast", "jit")]
    print(f"  [enjl] backends/ {'present' if has_dir else 'absent':7s}  "
          f"backend switch on {entry.__name__}: "
          f"{', '.join(switches) or 'none'}")

# %% [markdown]
# The nearest thing this model ships to a second flavour is the **branch pair**,
# and it is not one: `"up"` and `"down"` are not two implementations of one
# calculation but two different states, and their timings above differ because
# they solve different physics, not because one path is faster. Reporting them
# as a backend comparison would be the misreading this cell exists to prevent.
#
# ### 10.5 The summary table
#
# One row per set, configuration and branch, and the same rows written out under
# the naming convention of section 3. The model slot carries the study rather
# than a parameter set, because the table spans all three — the set of each row
# is a column inside it. `missed` is a list and does not survive as a table
# column, so the file keeps the count and the first density; the densities
# themselves are printed in 10.2.
#
# `table_path` is given the same absolute `root` section 3 built, for the same
# reason: its default is relative and a kernel started in `notebooks/` would
# write to `notebooks/output/`.

# %%
header("summary")
print(f"  {'set':10s} {'mode':22s} {'T':>5s} {'branch':>7s} {'cold ms':>9s} "
      f"{'warm ms/pt':>11s} {'line s':>8s} {'solved':>10s}")
for row in benchmarks:
    print(f"  {row['set']:10s} {row['mode']:22s} {row['T']:5.1f} "
          f"{row['direction']:>7s} "
          f"{row['cold_ms']:9.3f}{'' if row['cold_ok'] else '*'} "
          f"{row['warm_ms']:11.3f} {row['line_s']:8.3f} "
          f"{row['n_solved']:4d}/{row['n_requested']:<5d}")

# %%
bench_rows = [dict(parameter_set=row["set"], mode=row["mode"], T=row["T"],
                   direction=row["direction"], cold_ms=row["cold_ms"],
                   cold_ok=float(row["cold_ok"]), warm_ms=row["warm_ms"],
                   line_s=row["line_s"], n_solved=row["n_solved"],
                   n_requested=row["n_requested"],
                   n_missed=len(row["missed"]),
                   n_B_first_missed=(row["missed"][0] if row["missed"]
                                     else float("nan")))
              for row in benchmarks]

if bench_rows:
    bench_name = standard_name(
        "enjl", "benchmark", {},
        {"nB": BENCH_N_B,
         "T": np.array([T for _, T, _, _ in BENCH_CONFIGS])},
        KNOBS.species, leptons=KNOBS.leptons)
    bench_path = save_table(bench_rows,
                            table_path("enjl", bench_name, root=TABLE_ROOT),
                            meta={"study": "enjl benchmark",
                                  "parameter_sets": ",".join(KNOBS.sets),
                                  "modes": ",".join(m for m, _, _, _
                                                    in BENCH_CONFIGS),
                                  "directions": ",".join(d for _, _, _, d
                                                         in BENCH_CONFIGS),
                                  "species": KNOBS.species,
                                  "leptons": KNOBS.leptons,
                                  "repeat": BENCH_REPEAT})
    print("wrote", bench_path)

# %% [markdown]
# ## 11. What was written
#
# Every file this notebook produced, with the root each was anchored on.

# %%
header("output")
print(f" tables : {TABLE_ROOT}")
for path in sorted((ROOT / "output" / "tables" / "enjl").glob("*")):
    print(f"   {path.name}")
print(f" figures: {FIG_DIR}")
for path in sorted(FIG_DIR.glob("*")):
    print(f"   {path.name}")
