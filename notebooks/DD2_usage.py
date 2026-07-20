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
# # DD2 EoS engine — usage notebook
#
# A guided tour of the `eos.dd2` density-dependent RMF engine. The heavy code
# lives in the installed `eos` package; every non-trivial plotting/analysis
# routine lives in **`eos/dd2/notebook_api.py`** (imported here as `api`), so
# this notebook stays thin and diff-able.
#
# **This notebook is paired to `DD2_usage.py` via jupytext** (`formats:
# ipynb,py:percent`): edit *either* file and the other updates on save. The `.py`
# is the review-friendly source of truth.
#
# Layout:
# - **Part I — Setup & knobs.** Imports and *every* tunable (parametrization,
#   particle content, grids, table mode). Edit here, then Run-All. *Always run.*
# - **Part II — Physics plots.** The 11-figure set for the selected parametrization.
# - **Part III — Second parametrization.** The same set for an NMP-built par.
# - **Part IV — Generate & export a table.** Pick species/mode/axes, write a file.
# - **Part V — Speed test.** DD2 vs SFHo at T=0 and T>0.

# %% [markdown]
# # Part I — Setup & knobs
# ## I.1 — Imports & installs
#
# Run once from a fresh kernel. Keep the GitHub-install lines active for a clean
# environment (Colab); switch to the commented local-editable block when
# developing the `eos` package (repo edits then take effect after a kernel
# restart only).

# %%
import sys

# ─── Install the packages from GitHub (latest commit) ────────────────────────
# !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/guerrinimirco/eos.git --quiet
print("eos package loaded successfully!")
# !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/guerrinimirco/metastability-nucleation.git --quiet
print("nucleation package loaded successfully!")

# Local-dev alternative (comment the two installs above, uncomment these):
# # !{sys.executable} -m pip install -e .. --quiet
# # !{sys.executable} -m pip install -e ../../metastability-nucleation --quiet

# ─── Scientific Python ───────────────────────────────────────────────────────
import numpy as np

# ─── DD2 engine (plotting helpers live in eos/dd2/notebook_api.py) ───────────
from eos.dd2 import Parametrization, SpeciesFlags
from eos.dd2 import notebook_api as api

# %% [markdown]
# ## I.2 — Knobs — **EDIT THIS CELL**
#
# Everything you'd normally change lives here. Downstream cells only *read* these.
#
# - `PAR` — the parametrization. `from_dd2_defaults()` (nucleonic DD2),
#   `from_dd2y_defaults()` (DD2Y, needed for hyperons), or an NMP-built one
#   (Part III).
# - `FLAGS` — particle content. Hyperons/Δ need a par that carries their
#   couplings (DD2Y / a Δ-calibrated par); the plain DD2 par is nucleonic only.
# - grids / temperatures — the axes of the plots.
# - `TABLE_*` — what Part IV exports.

# %%
# ── parametrization ──────────────────────────────────────────────────────────
PAR = Parametrization.from_dd2_defaults()      # ← nucleonic DD2 (default)
# PAR = Parametrization.from_dd2y_defaults()   #   DD2Y (enables hyperons below)

# ── particle content ─────────────────────────────────────────────────────────
FLAGS = SpeciesFlags(hyperons=False,           # ← Λ,Σ,Ξ octet (needs DD2Y)
                     deltas=False,             # ← Δ quartet (needs a Δ par)
                     muons=True,               #   e always on; μ optional
                     phi_field=False)          #   hidden-strange φ (DD2Y default)

# ── axes ─────────────────────────────────────────────────────────────────────
NB_GRID   = np.geomspace(0.06, 1.2, 60)        # ← β-eq density grid [fm^-3]
T_FIXED   = 10.0                               # ← T for the heat-capacity plot [MeV]
S_VALUES  = (1.0, 2.0, 4.0)                    # ← isentropic S = s/n_B values

# ── observational / reference data (repo-relative from notebooks/) ───────────
# These live in the repo under plot/data, NOT in the pip-installed package, so
# reference them by path relative to this notebook (as the 2fam notebook does).
DATA_DIR    = "../plot/data"
CONTOUR_DIR = DATA_DIR + "/contours"                 # fig II.8 M-R overlays
CHIRAL_EFT  = DATA_DIR + "/samples/chiral_eft.txt"   # fig II.10 chiral band

# ── Part IV: table to export ─────────────────────────────────────────────────
TABLE_MODE = "beta"                # 'beta','YC','YS','YC+YS','YL'
TABLE_FIXED = {}                   # e.g. {"Y_C": 0.1} for YC, {"Y_L": 0.4} for YL
TABLE_TEMP = {"T": [0.0]}          # temperature axis: {"T":[...]} or {"SnB":[1,2]}
TABLE_NB = np.geomspace(0.05, 1.2, 80)
TABLE_PATH = "DD2_table_beta_T0.dat"

print("PAR    :", PAR.__class__.__name__, "| n_sat =", round(PAR.n_sat, 4))
print("FLAGS  :", FLAGS)
print("NB_GRID:", NB_GRID[0], "→", NB_GRID[-1], "fm^-3,", len(NB_GRID), "points")

# %% [markdown]
# # Part II — Physics plots
#
# The 11-figure set for the selected `PAR` / `FLAGS`. Cold / T=0 unless the plot
# is about temperature. To regenerate for a different parametrization, change
# `PAR` in I.2 and Run-All-Below — or just read Part III.

# %% [markdown]
# ## II.1 — Pressure vs $n_B$ (β-equilibrium)
# Neutrino-transparent npeμ matter along β-equilibrium (`sweep_beta_eq_octet`),
# log-y.

# %%
api.plot_p_vs_nb(PAR, flags=FLAGS, grid=NB_GRID)

# %% [markdown]
# ## II.2 — Composition $Y_i$ vs $n_B$
# Particle fractions along β-eq. With a nucleonic `PAR` this is n, p, e, μ; with a
# DD2Y `PAR` + `hyperons=True` the hyperon onsets appear (see II.2b).

# %%
api.plot_composition(PAR, flags=FLAGS, grid=NB_GRID)

# %% [markdown]
# ## II.2b — Hyperon onsets (DD2Y)
# The plain DD2 par carries no hyperon couplings, so hyperons need the DD2Y par.
# This cell is self-contained (it builds DD2Y locally) regardless of the I.2 knob.

# %%
api.plot_composition(Parametrization.from_dd2y_defaults(), flags=api.OCTET,
                     grid=NB_GRID)

# %% [markdown]
# ## II.3 — Isentropic temperature
# Temperature along constant entropy-per-baryon paths S = s/n_B.

# %%
api.plot_isentropic_T(PAR, flags=FLAGS, S_values=S_VALUES)

# %% [markdown]
# ## II.4 — Speed of sound $c_s^2$
# Frozen (fixed-composition) and equilibrium $c_s^2$, with the causal limit.

# %%
api.plot_sound_speed(PAR, flags=FLAGS)

# %% [markdown]
# ## II.5 — Heat capacities $C_V$, $C_P$ (needs T>0)
# Per-baryon $C_V$ and $C_P$ at fixed `T_FIXED`.

# %%
api.plot_heat_capacity(PAR, flags=FLAGS, T=T_FIXED)

# %% [markdown]
# ## II.6–8 — TOV structure (M-R, Λ-M, constraints)
# Cold β-eq core + BPS crust. The TOV sequence is solved **once** here and reused
# by the three figures. NS structure uses nucleonic flags (a hyperonic core needs
# a DD2Y `PAR`).

# %%
FLAGS_TOV = api.NUCLEONIC if not FLAGS.hyperons else FLAGS
tov = api.compute_tov(PAR, FLAGS_TOV)
print(f"M_max={tov['M_max']:.3f} M_sun | R_1.4={tov['R_1p4']:.2f} km | "
      f"Lambda_1.4={tov['Lambda_1p4']:.0f}")

# %% [markdown]
# ## II.6 — Mass-radius

# %%
api.plot_mass_radius(PAR, flags=FLAGS_TOV, tov=tov)

# %% [markdown]
# ## II.7 — Tidal deformability Λ-M

# %%
api.plot_lambda_mass(PAR, flags=FLAGS_TOV, tov=tov)

# %% [markdown]
# ## II.8 — M-R vs observational constraints
# The curve over the shipped J0030 / J0740 / HESS / GW170817 / GW190425 posteriors
# (`add_observational_constraints` from the `nucleation` figure utilities).

# %%
api.plot_mr_with_constraints(PAR, flags=FLAGS_TOV, tov=tov, contour_dir=CONTOUR_DIR)

# %% [markdown]
# ## II.9 — Pressure vs $n_B$ (symmetric matter)
# Symmetric nuclear matter (Y_p = 0.5), log-y (positive branch above saturation).

# %%
api.plot_p_vs_nb_snm(PAR, grid=NB_GRID)

# %% [markdown]
# ## II.10 — Pure neutron matter vs chiral EFT
# PNM energy per particle against the shipped chiral-EFT band.

# %%
api.plot_pnm_chiral(PAR, chiral_path=CHIRAL_EFT)

# %% [markdown]
# ## II.11 — Nuclear-matter parameters
# `compute_nmp(PAR)` next to the DD2 reference values.

# %%
print(api.format_nmp_comparison(PAR))

# %% [markdown]
# # Part III — Second parametrization (NMP-built)
#
# `Parametrization.from_nmp` inverts a target NMP set to DD2 couplings. Here we
# nudge `L_sym` 55 → 70 MeV (a stiffer symmetry energy) so the two passes visibly
# differ. We check the inverter converged before plotting.

# %%
PAR_NMP, nmp_status = api.build_nmp_par()
print("NMP inversion:", nmp_status.message, "| ok =", nmp_status.ok)
assert nmp_status.ok, "inverter did not converge — inspect nmp_status, do not plot"
print(api.format_nmp_comparison(PAR_NMP))

# %% [markdown]
# ## III.1 — Pressure, composition, sound speed (NMP par)

# %%
api.plot_p_vs_nb(PAR_NMP, grid=NB_GRID)

# %%
api.plot_composition(PAR_NMP, grid=NB_GRID)

# %%
api.plot_sound_speed(PAR_NMP)

# %% [markdown]
# ## III.2 — PNM vs chiral EFT (NMP par)
# The clearest visual difference from the `L_sym` nudge.

# %%
api.plot_pnm_chiral(PAR_NMP, chiral_path=CHIRAL_EFT)

# %% [markdown]
# ## III.3 — TOV structure (NMP par)
# Stiffer `L_sym` ⇒ larger $R_{1.4}$ and $\Lambda_{1.4}$.

# %%
tov_nmp = api.compute_tov(PAR_NMP, api.NUCLEONIC)
print(f"NMP: M_max={tov_nmp['M_max']:.3f} | R_1.4={tov_nmp['R_1p4']:.2f} km | "
      f"Lambda_1.4={tov_nmp['Lambda_1p4']:.0f}")
api.plot_mass_radius(PAR_NMP, tov=tov_nmp)

# %%
api.plot_lambda_mass(PAR_NMP, tov=tov_nmp)

# %% [markdown]
# # Part IV — Generate & export a table
#
# Build a table for the species / mode / axes you need (via `TableSpec` +
# `build_table`) and write it to disk. The knobs come from I.2 (`TABLE_MODE`,
# `TABLE_FIXED`, `TABLE_TEMP`, `TABLE_NB`, `TABLE_PATH`) — edit them there.
#
# Modes: `beta` (charge-neutral β-eq), `YC` (fixed charge fraction),
# `YS` (fixed strangeness), `YC+YS`, `YL` (fixed lepton fraction, trapped ν).
# Temperature axis is either `{"T":[...]}` or `{"SnB":[...]}` (entropy per baryon).

# %%
result, path = api.export_eos_table(
    PAR, FLAGS,
    mode=TABLE_MODE,
    nB=TABLE_NB,
    T=TABLE_TEMP.get("T"),
    SnB=TABLE_TEMP.get("SnB"),
    fixed=TABLE_FIXED,
    path=TABLE_PATH,
)
n_rows = sum(len(line) for line in result.points)
print(f"wrote {n_rows} rows to {path}")
print("first data line:")
print(open(path).read().splitlines()[2])

# %% [markdown]
# # Part V — Speed test: DD2 vs SFHo
#
# Matched β-equilibrium density sweep. DD2 fast path
# (`sweep_beta_eq_octet(..., analytic_jac=True)`, first call discarded for the
# Numba compile) vs the SFHo table generator, at **T=0 and T>0**.
#
# Anchor (dev machine): a 100-pt nucleonic DD2 sweep is ~35 ms (~0.35 ms/pt) at
# T=0. At T>0 the DD2 analytic path is NumPy (the JEL integrals don't jit), so its
# ms/pt is naturally larger — treat these as sanity values, not targets.

# %%
for T in (0.0, 10.0):
    b = api.benchmark_dd2_vs_sfho(PAR, T=T)
    print(f"T = {T:>4} MeV | {b['n_points']} pts | "
          f"DD2 {b['dd2_ms_per_pt']:.3f} ms/pt | "
          f"SFHo {b['sfho_ms_per_pt']:.3f} ms/pt | "
          f"ratio SFHo/DD2 = {b['ratio']:.2f}")
