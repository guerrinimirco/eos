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
# # Hybrid equations of state — the hadron-quark mixed phase
#
# The composite engine of `eos`, driven through the public API and nothing else:
# `eos_point`, `eos_table`, `hybrid_table`, and the phase adapters that declare a
# pairing. No solver internal is touched and no helper module sits beside this
# notebook — everything it needs is either in the library or in the cells below.
#
# A composite engine takes the same three entry points a model takes and adds the
# transition itself. Three things follow, and they are what this notebook is
# about:
#
# * **the parameter argument is a pair.** A hybrid equation of state has two
#   models in it, so both parameter sets are arguments. Either they are named
#   separately through the DD2 + vMIT front door, or the pairing is declared as
#   two `Phase` objects, each closing over its own model's parameters.
# * **`eta` is a scalar per call, never an axis.** It chooses how much of
#   electric-charge neutrality is imposed phase by phase — `eta = 0` is Gibbs,
#   `eta = 1` Maxwell — and it changes the shape of the unknown vector, so a grid
#   over it is a grid of separate solves, not a table axis.
# * **`eos_table` returns `(rows, windows)`.** The phase boundaries found on each
#   line are part of the answer, not something recovered afterwards by scanning
#   the rows for where `chi` left `[0, 1]`.
#
# What is here:
#
# 1. **The knobs** — every choice this notebook makes, in one cell.
# 2. **Reporting a gap** — the three distinct things that can happen, and why
#    they must stay three.
# 3. **The pairing** — the two calling forms, shown to give the same floats.
# 4. **The shipped pairings** — one point through each, on one engine.
# 5. **eta** — the construction, one scalar call at a time.
# 6. **A table is rows plus windows** — the headline DD2 + vMIT run, saved.
# 7. **The stitched hybrid table** — hadronic wing, mixed window, quark core,
#    and the handoff into the structure solver.
#
# Units are the ones every public boundary uses: densities in fm^-3,
# temperatures and chemical potentials in MeV, pressure and energy density in
# MeV/fm^3.

# %%
import importlib
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

from eos.general.table_io import save_table, standard_name, table_path
from eos.mixed import eos_point, eos_table, hybrid_table
import eos.mixed.adapters as adapters

# %% [markdown]
# ## 1. The knobs
#
# Everything selectable is selectable here and nowhere else; no cell below
# reaches past this one for a number.
#
# Two entries have no counterpart in a single-phase notebook. **The pairing** is
# a hadronic adapter and a quark adapter, named together with the parameter set
# each is to close over — a `Phase` pair *is* the parameter argument of the
# composite engine. And **`eta`** is a plain float rather than a grid: it selects
# the construction, and a construction is not a coordinate. `eta_examples` below
# is a list of separate calls, not an axis, which is why it lives apart from the
# grids.
#
# `mode` and the fractions are the library's equilibrium conditions, unchanged
# by there being two phases: a fixed-`Y_C` hybrid is fixed-`Y_C` in the hadronic
# wing, in the mixed window and in the quark core. `leptons` is orthogonal to
# them and is never an entry in `conditions()`.

# %%
from dataclasses import dataclass, field


@dataclass
class Knobs:
    """Every choice this notebook makes, in one place."""

    # --- the pairing: two adapters and two parameter sets ----------------
    # Hadronic: dd2, sfho, did, zl.   Quark: vmit, alphabag, njl, ccdm.
    # The headline pairing is DD2 + vMIT, the one with a retired notebook's
    # tables to be checked against (that comparison is its own study).
    hadronic: str = "dd2"
    quark: str = "vmit"

    # Per side: "default" -> Parameters.default(), ("named", X) -> .named(X).
    hadronic_parameters: object = "default"
    quark_parameters: object = "default"

    # The DD2 + vMIT front door takes (par, flags, vmit_params) directly; any
    # other pairing goes through `phases=`. Section 3 runs both and compares.
    front_door: bool = True

    # --- the construction ------------------------------------------------
    # A SCALAR per call. eta = 0 is Gibbs (charge neutrality imposed on the
    # mixture as a whole), eta = 1 is Maxwell (imposed within each phase),
    # and values between stand in for the surface and Coulomb cost of the
    # mixed-phase structures. It changes the shape of the unknown vector, so
    # it is not a table axis and `eos_table` takes one value per call.
    eta: float = 0.5
    eta_examples: tuple = (0.0, 0.5, 1.0)   # separate calls, not a grid

    # --- the equilibria to exercise, and the fractions they take ---------
    modes: tuple = ("beta_eq_neutrinoless", "fixed_YC", "fixed_YC_YS")
    Y_C: float = 0.1                   # fixed_YC, fixed_YC_YS
    Y_S: float = 0.0                   # fixed_YC_YS
    Y_Le: float = 0.3                  # beta_eq_neutrino_trapped
    leptons: bool = True               # orthogonal to the mode

    # --- the grid -------------------------------------------------------
    n_B: tuple = (0.10, 1.40, 27)      # (lo, hi, count), fm^-3
    thermal: str = "T"                 # "T" or "SnB"
    thermal_grid: tuple = (0.0, 20.0, 2)   # MeV, or k_B per baryon

    # --- the sectors ----------------------------------------------------
    # The six named degrees of freedom, spelled the same way in every model
    # of the repository. Every one is explicit: a sector that is off is off
    # because its flag is False, never because a coupling happens to vanish.
    species: dict = field(default_factory=lambda: dict(
        hyperons=False, deltas=False, muons=False,
        thermal_mesons=False, thermal_neutrinos=False, photons=True))

    # DD2's matter-composition electron neutrino, which the trapped mode needs
    # and which is NOT `thermal_neutrinos` above. It is a dd2 flag rather than
    # one of the six, so it is named apart from them.
    neutrinos: bool = True

    # `window_only=True` solves the mixed system only between the located
    # boundaries, where it is the only thing that can answer the question;
    # outside them the far cheaper pure-phase solvers give the same state.
    # False solves at every grid point, which is what studying the
    # continuation root of chi outside [0, 1] needs.
    window_only: bool = True

    def n_B_grid(self):
        lo, hi, count = self.n_B
        return np.linspace(lo, hi, count)

    def thermal_values(self):
        lo, hi, count = self.thermal_grid
        return np.linspace(lo, hi, count)

    def conditions(self, mode):
        """Only the fractions THIS mode takes, under the library's names."""
        taken = {"beta_eq_neutrinoless": (),
                 "beta_eq_neutrino_trapped": ("Y_Le",),
                 "fixed_YC": ("Y_C",),
                 "fixed_YC_YS": ("Y_C", "Y_S")}[mode]
        return {name: getattr(self, name) for name in taken
                if getattr(self, name) is not None}

    def axes(self, mode):
        """The grid as `eos_table`'s `axes` argument, fraction axes included.

        `eta` is deliberately absent: it is an argument of the call, not an
        entry here.
        """
        axes = {"nB": self.n_B_grid(), self.thermal: self.thermal_values()}
        for name, value in self.conditions(mode).items():
            axes[name] = np.array([value])
        return axes


KNOBS = Knobs()
KNOBS

# %% [markdown]
# ## 2. Reporting a gap without presenting it as a result
#
# Asking the engine for something can end three different ways, and collapsing
# them would be the single most misleading thing this notebook could do.
#
# * **not supported** — the pairing refuses the mode, the flag or the
#   temperature, and says which phase refused. A refusal is the adapter contract
#   working (a strangeness-free hadronic model under a mode that conserves S; a
#   T = 0 surface asked for finite T; a pairing without a frozen-composition
#   block), not a defect, and it is never dressed up as a result.
# * **did not converge** — the solve ran and failed. That is a *return value*,
#   not an exception, so no `except` clause ever sees it; it is found by testing
#   `.ok`. A sampler walking into an unphysical corner has to be able to score
#   the point and move on.
# * **ok** — there is a number.
#
# A fourth outcome belongs to a composite engine alone and is **not** a failure:
# a converged point whose `chi` lies outside `[0, 1]`. That is the engine saying
# which side of the transition the density is on — the mixed system still has a
# root there, but it is an analytic continuation and the state is the pure phase.
# `.point.phase` is `'H'`, `'mix'` or `'Q'` and is printed alongside every number
# below, because in a composite engine which regime a point is in decides which
# of its numbers mean anything.
#
# `TypeError` is deliberately not caught. An unexpected keyword argument is this
# notebook's own bug, and a broad `except` would file it under "the engine does
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
    """One printed header, so a skipped pairing is visible in the output."""
    if mode is None:
        print(f"\n=== {title} ===")
    else:
        print(f"\n=== {title} — mode={mode} {KNOBS.conditions(mode)} "
              f"eta={KNOBS.eta} leptons={KNOBS.leptons} ===")


def parameters_for(name, choice):
    """The parameter object the knobs ask for. Parameters are arguments."""
    module = importlib.import_module(f"eos.{name}")
    if choice == "default":
        return module.Parameters.default()
    kind, published = choice
    assert kind == "named", f"unknown parameter choice {choice!r}"
    return module.Parameters.named(published)


def hadronic_flags(name):
    """The species flags of one hadronic model, from the six named sectors.

    `neutrinos` is added for `dd2` alone: it is that model's own matter-
    composition electron neutrino and not one of the six that every model
    spells alike, so it is passed where it exists and never silently dropped
    where it does not.
    """
    module = importlib.import_module(f"eos.{name}")
    extra = {"neutrinos": KNOBS.neutrinos} if name == "dd2" else {}
    return module.SpeciesFlags(**KNOBS.species, **extra)


# %% [markdown]
# ## 3. The pairing — two calling forms for the same physics
#
# The parameter argument of a composite engine is two parameter sets, and there
# are two ways to hand them over.
#
# The **front door** is the DD2 + vMIT signature: the hadronic parameters and
# species flags in the positions every model uses, and `vmit_params` beside
# them. It exists because that pairing is the one with published results behind
# it, and it keeps those calls looking like every other model call.
#
# The **general form** is `phases=(hadronic, quark)`: two declared `Phase`
# objects, each built by a factory that closes over its own model's parameters.
# `par`, `species` and `vmit_params` must then all be `None` — the pair carries
# everything — and `muons=` says whether muons join the neutralizing lepton
# domains, a question the flags answered in the front door.
#
# A `Phase` bundles the adapter callable with what the engine must *know* about
# a phase and must not *assume*: whether its baryon slot carries the kinetic
# potential (DD2, DID, whose rearrangement term depends on the density the solve
# is finding) or the physical one (SFHo, ZL, vMIT, alphaBag, NJL, CCDM), whether
# it supports strangeness at all, the highest temperature it is written for, and
# which optional capabilities — wings, a frozen-composition block, an analytic
# Jacobian — it provides. Everything else about the model stays behind the
# adapter.
#
# The two forms are the same solve, so they agree to the last bit.

# %%
def phase_pair():
    """The `(hadronic, quark)` Phase pair the knobs name.

    Each factory closes over its model's own parameters, which is how "model
    parameters are arguments" reads for a composite engine.
    """
    had_par = parameters_for(KNOBS.hadronic, KNOBS.hadronic_parameters)
    quark_par = parameters_for(KNOBS.quark, KNOBS.quark_parameters)
    factory = getattr(adapters, f"{KNOBS.hadronic}_phase")
    # zl is written in (n_p, n_n) and takes no species flags; the other three
    # hadronic adapters take the model's own flags.
    if KNOBS.hadronic == "zl":
        hadronic = factory(had_par)
    else:
        hadronic = factory(had_par, hadronic_flags(KNOBS.hadronic))
    quark = getattr(adapters, f"{KNOBS.quark}_phase")(quark_par)
    return hadronic, quark


PAIR = phase_pair()
print("pairing:", PAIR[0].name, "+", PAIR[1].name)
for phase in PAIR:
    print(f"  {phase.name:10s} slot={phase.slot('H' if phase is PAIR[0] else 'Q'):16s}"
          f" supports_S={phase.supports_S}  max_T={phase.max_T}"
          f"  wings={'yes' if phase.wing_sweep else 'no'}"
          f"  frozen={'yes' if phase.frozen_thermo else 'no'}")

# %%
PROBE_N_B = 0.75      # inside the DD2 + vMIT coexistence window at every eta

header("the two calling forms")

par = parameters_for(KNOBS.hadronic, KNOBS.hadronic_parameters)
flags = hadronic_flags(KNOBS.hadronic)
quark_par = parameters_for(KNOBS.quark, KNOBS.quark_parameters)

status, general = run(
    "phases=", eos_point, None, "beta_eq_neutrinoless", None,
    n_B=PROBE_N_B, T=0.0, eta=KNOBS.eta, phases=PAIR,
    muons=KNOBS.species["muons"])

if KNOBS.front_door and (KNOBS.hadronic, KNOBS.quark) == ("dd2", "vmit"):
    status_fd, front = run(
        "front door", eos_point, par, "beta_eq_neutrinoless", flags,
        n_B=PROBE_N_B, T=0.0, eta=KNOBS.eta, vmit_params=quark_par)
    if status == "ok" and status_fd == "ok":
        a, b = general.point, front.point
        print(f"  front door : chi={b.chi:+.10f}  P={b.P:.10f}  eps={b.eps:.10f}")
        print(f"  phases=    : chi={a.chi:+.10f}  P={a.P:.10f}  eps={a.eps:.10f}")
        print(f"  identical  : {a.chi == b.chi and a.P == b.P and a.eps == b.eps}")
else:
    print("  front door is DD2 + vMIT only; this pairing goes through phases=")

# %% [markdown]
# ## 4. The shipped pairings
#
# Eight adapters ship — `dd2`, `sfho`, `did`, `zl` on the hadronic side,
# `vmit`, `alphabag`, `njl`, `ccdm` on the quark side — and any hadronic one
# pairs with any quark one through the same engine. A new pairing is a new
# adapter, never a new engine, and that is the whole claim being exercised here:
# one call, sixteen combinations, no per-pairing code.
#
# Read the `chi` column, not just the `ok` column. At one fixed density most of
# these pairings are *not* in coexistence — their windows sit elsewhere — and a
# `chi` outside `[0, 1]` says exactly that. A pairing that neither converges nor
# refuses is reported as it is, without being folded into either of the other
# two.

# %%
HADRONIC = ("dd2", "sfho", "did", "zl")
QUARK = ("vmit", "alphabag", "njl", "ccdm")

header("one point through every shipped pairing")
print(f"  n_B={PROBE_N_B} fm^-3, T=0, eta={KNOBS.eta}, beta_eq_neutrinoless")

for had_name in HADRONIC:
    for quark_name in QUARK:

        def solve(had_name=had_name, quark_name=quark_name):
            had_par = parameters_for(had_name, "default")
            quark_par = parameters_for(quark_name, "default")
            had_factory = getattr(adapters, f"{had_name}_phase")
            hadronic = (had_factory(had_par) if had_name == "zl"
                        else had_factory(had_par, hadronic_flags(had_name)))
            quark = getattr(adapters, f"{quark_name}_phase")(quark_par)
            return eos_point(None, "beta_eq_neutrinoless", None,
                             n_B=PROBE_N_B, T=0.0, eta=KNOBS.eta,
                             phases=(hadronic, quark),
                             muons=KNOBS.species["muons"])

        label = f"{had_name}+{quark_name}"
        status, result = run(label, solve)
        if status == "ok":
            point = result.point
            print(f"  [{label:14s}] phase={point.phase:3s} chi={point.chi:+8.4f}"
                  f"  P={point.P:8.3f}  eps={point.eps:9.3f}")

# %% [markdown]
# ## 5. eta — the construction, one call at a time
#
# `eta` is the fraction of electric-charge neutrality imposed *locally*, phase by
# phase, rather than on the mixture as a whole:
#
# | `eta` | construction | neutrality |
# |---|---|---|
# | 0 | Gibbs | global — each phase may be charged, the mixture is not |
# | 1 | Maxwell | local — each phase is neutral on its own |
# | between | interpolation | stands in for surface tension and the Coulomb cost |
#
# At `eta = 0` the two phases share one electron sea and one `mu_C`, so the
# unknown vector has one global lepton potential in it. At `eta = 1` each phase
# neutralizes itself and there are two local ones. The shape of the system is
# therefore a function of `eta`, which is why the library takes it as a scalar
# per call and refuses to treat it as a table axis: a grid over `eta` is a grid
# of separate solves, and each cell below is one of them.
#
# The Maxwell limit is visible in the numbers rather than asserted. At `eta = 1`
# the two phases are separately neutral and coexist at one pressure, so `P` is
# the same at every density inside the window while `chi` runs from 0 to 1 — the
# plateau a Maxwell construction is. At `eta = 0` there is no plateau: the shared
# electron sea lets `mu_C` vary through the window and `P` rises across it.

# %%
header("eta, one scalar call at a time")

for eta in KNOBS.eta_examples:

    def solve(eta=eta):
        return eos_point(None, "beta_eq_neutrinoless", None, n_B=PROBE_N_B,
                         T=0.0, eta=eta, phases=PAIR,
                         muons=KNOBS.species["muons"])

    status, result = run(f"eta={eta}", solve)
    if status == "ok":
        point = result.point
        print(f"  [eta={eta:.2f}] phase={point.phase:3s} chi={point.chi:+8.4f}"
              f"  P={point.P:8.3f}  eps={point.eps:9.3f}"
              f"  mu_B={point.mu_B:9.3f}")

# %% [markdown]
# The plateau, at two densities inside the window at both constructions. `P` at
# `eta = 1` is one number; at `eta = 0` it is two.
#
# These are cold solves, one density at a time, which is the hard way to ask: the
# table builders below warm-start along the density axis, each solved point
# seeding the next, and reach parts of a window a cold start does not.

# %%
PLATEAU_N_B = (0.70, 0.75)

header("Maxwell against Gibbs across the window")
for eta in (0.0, 1.0):
    pressures = []
    for n_B in PLATEAU_N_B:

        def solve(n_B=n_B, eta=eta):
            return eos_point(None, "beta_eq_neutrinoless", None, n_B=n_B,
                             T=0.0, eta=eta, phases=PAIR,
                             muons=KNOBS.species["muons"])

        status, result = run(f"eta={eta} n_B={n_B}", solve)
        if status != "ok":
            break
        point = result.point
        pressures.append(point.P)
        print(f"  [eta={eta:.2f}] n_B={n_B:.2f}  chi={point.chi:+8.4f}"
              f"  P={point.P:12.6f}")
    if len(pressures) == 2:
        print(f"           dP across the two = {pressures[1] - pressures[0]:.3e}"
              f" MeV/fm^3")

# %% [markdown]
# ## 6. A table is rows plus windows
#
# `eos_table` returns `(rows, windows)`. The rows are the long format the table
# writer and the structure solver both read; the windows are the `n_onset` and
# `n_offset` located on each (temperature, fraction) line — the two densities the
# transition curves are made of. They are returned rather than recovered
# afterwards from the rows, because a boundary is found by a solve of its own (a
# fixed-`chi` solve at `chi = 0` and `chi = 1`) and reading it back off a grid
# would cost half a grid spacing of resolution.
#
# With `window_only=True` the rows are the in-window points: outside the
# boundaries the mixed system is not what answers the question, the pure phases
# are, and section 7 is where the three pieces are stitched into one table.
# A row count below the requested count is the table saying which points it could
# not solve — never a silent truncation.
#
# Each row carries the conserved charges *resolved by phase* as well as globally:
# `Y_C_H + Y_C_Q == Y_C` and `Y_B_H + Y_B_Q == 1`, on a volume-weighted
# convention. That partition is what `eta` controls and what a global sum cannot
# show — at one fixed total `Y_C` the hadronic phase can be far more positively
# charged than the average while the quark phase carries the compensating
# negative charge.

# %%
tables = {}
for mode in KNOBS.modes:
    header("a grid", mode)
    axes = KNOBS.axes(mode)

    def build(mode=mode, axes=axes):
        extra = ({"leptons": KNOBS.leptons} if mode.startswith("fixed_")
                 else {})
        return eos_table(None, mode, None, axes, eta=KNOBS.eta, phases=PAIR,
                         muons=KNOBS.species["muons"],
                         window_only=KNOBS.window_only, verbose=True, **extra)

    status, result = run(f"{KNOBS.hadronic}+{KNOBS.quark}", build)
    if status != "ok":
        continue
    rows, windows = result
    tables[mode] = (rows, windows)
    for key, window in sorted(windows.items()):
        if window is None or not np.isfinite(window.n_onset):
            print(f"  line {key}: no transition on this grid")
            continue
        print(f"  line {key}: n_onset={window.n_onset:.4f}  "
              f"n_offset={window.n_offset:.4f} fm^-3")
    if rows:
        first, last = rows[0], rows[-1]
        print(f"  {len(rows)} rows in the windows   "
              f"P {first['P']:8.3f} -> {last['P']:8.3f}   "
              f"chi {first['chi']:+.4f} -> {last['chi']:+.4f}")

# %% [markdown]
# The per-phase charge partition, on the first table's rows. These three sums are
# invariants of a solved point, so they are the cheapest check that a row means
# what it says:

# %%
CHECK = KNOBS.modes[0]

if CHECK in tables:
    rows, _ = tables[CHECK]
    print(f"  {CHECK}: {len(rows)} rows")
    worst_B = max(abs(r["Y_B_H"] + r["Y_B_Q"] - 1.0) for r in rows)
    worst_C = max(abs(r["Y_C_H"] + r["Y_C_Q"] - r["Y_C"]) for r in rows)
    worst_S = max(abs(r["Y_S_H"] + r["Y_S_Q"] - r["Y_S"]) for r in rows)
    print(f"  max |Y_B_H + Y_B_Q - 1|      = {worst_B:.3e}")
    print(f"  max |Y_C_H + Y_C_Q - Y_C|    = {worst_C:.3e}")
    print(f"  max |Y_S_H + Y_S_Q - Y_S|    = {worst_S:.3e}")

# %% [markdown]
# ### Saving one
#
# The automatic name carries every choice that changes a number — the pairing,
# the mode, the mode's fractions, **`eta`**, the thermal axis, the density axis,
# the sectors that are on, and `nolep` when the neutralizing leptons are off — so
# two runs cannot collide silently and a folder listing says months later how
# each file was made. `eta` is in the name for the same reason it is an argument:
# two tables that differ only in the construction are two different tables.
#
# The windows go into the file alongside the rows. A mixed table that dropped
# them would have thrown away half of its own result.

# %%
SAVE = "beta_eq_neutrinoless"

if SAVE in tables:
    rows, windows = tables[SAVE]
    label = f"{KNOBS.hadronic}{KNOBS.quark}"
    filename = standard_name(label, SAVE, KNOBS.conditions(SAVE),
                             KNOBS.axes(SAVE), KNOBS.species,
                             leptons=KNOBS.leptons, eta=KNOBS.eta)
    print(filename)
    # `table_path` builds its folder relative to the working directory, and a
    # notebook's is `notebooks/`. ROOT was already found for the import above,
    # so the tables land in the repository's own `output/tables/<pairing>/`
    # wherever the notebook is run from.
    path = save_table(rows, table_path(label, filename,
                                       root=str(ROOT / "output" / "tables")),
                      windows=windows,
                      meta={"pairing": label, "mode": SAVE, "eta": KNOBS.eta,
                            "hadronic_parameters": par,
                            "quark_parameters": quark_par,
                            "species": flags,
                            **KNOBS.conditions(SAVE)})
    print("wrote", path)

# %% [markdown]
# ## 7. The stitched hybrid table
#
# `eos_table` above solves the mixed system where the mixed system is the
# answer. `hybrid_table` is the other shape of the same physics: one equilibrium
# across the whole density range, stitched out of the pure hadronic wing below
# the onset, the eta-mixed phase inside the window, and the pure quark core above
# the offset. The mode holds in all three segments — a fixed-`Y_C` hybrid is
# fixed-`Y_C` everywhere, not a fixed-`Y_C` window between beta-equilibrium
# wings — and only `eta` is specific to the mixed region.
#
# It returns a `HybridResult`: test `.ok` before `.table`, because a boundary
# that cannot be located and a pressure inversion too large to be round-off are
# both return values here, not exceptions. The wings need a `wing_sweep`
# capability from each phase; a pairing whose adapter has none raises naming the
# phase rather than quietly returning a shorter table.
#
# `.table.to_tov()` is the handoff into `eos.astro.tov`. It is an
# `EOSTable_for_TOV`, which lives in `general/` — the layer the engine and the
# structure solver may both import — and that object is the whole of the
# contract between them. Section 8's gate applies before it is integrated: the
# pressure delivered to a structure solver is non-decreasing in `n_B`, and inside
# a Maxwell window it is constant, which is real physics and is why the
# monotonicity is enforced against round-off rather than assumed.

# %%
for mode in KNOBS.modes:
    header("the stitched table", mode)

    def build(mode=mode):
        extra = ({"leptons": KNOBS.leptons} if mode.startswith("fixed_")
                 else {})
        return hybrid_table(None, mode, None, n_B_grid=KNOBS.n_B_grid(),
                            eta=KNOBS.eta, T=0.0, phases=PAIR,
                            muons=KNOBS.species["muons"],
                            **KNOBS.conditions(mode), **extra)

    status, result = run(f"{KNOBS.hadronic}+{KNOBS.quark}", build)
    if status != "ok":
        continue
    table = result.table
    counts = {tag: int(np.sum(table.phase == tag)) for tag in ("H", "mix", "Q")}
    print(f"  {len(table.n_B)} rows   {counts}")
    if table.has_transition:
        # P_trans is the Maxwell plateau pressure and is nan for eta < 1,
        # where there is no plateau to report.
        print(f"  n_onset={table.n_onset:.4f}  n_offset={table.n_offset:.4f}"
              f"  P_trans={table.P_trans}")
    else:
        print("  no transition on this grid: the table is pure hadronic")
    increasing = bool(np.all(np.diff(table.P) >= 0.0))
    print(f"  P non-decreasing in n_B: {increasing}")
    for_tov = table.to_tov()
    print(f"  to_tov(): {type(for_tov).__name__} with {len(for_tov.nB)} points, "
          f"P {for_tov.P[0]:.3f} -> {for_tov.P[-1]:.3f} MeV/fm^3")

# %% [markdown]
# ## Two boundaries this notebook does not cross
#
# **ENJL is not here.** `enjl_branch_pair` is an `eos/mixed` adapter, but it
# pairs two branches of one functional rather than two models, and the physics
# that makes it interesting is ENJL's own — so it belongs to the ENJL notebook,
# not to this one.
#
# **Figures are not here.** This notebook produces converged tables and the
# windows that go with them; overlaying them, and checking them against the
# tables of the retired `eos_tables_DD2vMIT` notebook, is a study of its own.
