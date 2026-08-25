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
# point still returns the stable EoS, keeping at each density whichever branch
# has the lower energy density, which is the object section 5 assembled by hand:

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
# **(b)** The residual of section 6, per set, on a log axis. This is the figure
# the golden references earn: agreement over the reproduced window, and the
# density where the worst row sits.

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
ax_Y.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_Y.set_ylabel(r"$Y_i = n_i / n_B$")
ax_Y.set_title(COMP_SET)
ax_Y.legend(loc="lower right", ncol=2)
fs.apply_style(ax_Y, legend=False)
fs.panel_label(ax_Y, "(a)", corner="upper left")

if residuals:
    for set_name, (densities, errors) in residuals.items():
        ax_res.plot(densities, errors["P"], "-",
                    color=SET_COLOR.get(set_name, "0.4"),
                    label=f"{set_name}, P")
        ax_res.plot(densities, errors["eps"], "--",
                    color=SET_COLOR.get(set_name, "0.4"),
                    label=f"{set_name}, eps")
    ax_res.set_yscale("log")
    ax_res.legend(loc="best", ncol=1)
else:
    ax_res.text(0.5, 0.5, "author tables absent", ha="center", va="center",
                transform=ax_res.transAxes)
ax_res.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax_res.set_ylabel("relative residual against the author")
fs.apply_style(ax_res, legend=False)
fs.panel_label(ax_res, "(b)", corner="upper right")

fs.save_figure(fig, str(FIG_DIR / "enjl_composition_and_residual"))
plt.show()

# %% [markdown]
# ### What was written
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
