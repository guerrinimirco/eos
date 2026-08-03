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
# # DD2 + vMIT mixed-phase EoS — usage notebook
#
# Production tour of the `eos.mixed` engine: a first-order hadron→quark transition
# coupling the validated **DD2** hadronic RMF (`eos.dd2`) to the validated **vMIT**
# bag model (`eos.vmit`) through a continuous local/global charge-neutrality
# parameter **η** — η=0 is a **Gibbs** mixed phase (P rises through the window),
# η=1 is a **Maxwell** plateau (constant P, a density jump). The four named
# equilibrium modes (A/β-eq, B/trapped-ν, C/fixed-Y_C, D/fixed-Y_C+Y_S) are
# *configurations of one solver*, not separate codes.
#
# The heavy code lives in the installed `eos` package; every plot/analysis routine
# is in **`eos/mixed/notebook_api.py`** (imported as `api`), so this notebook stays
# thin. **Paired to `mixed_usage.py` via jupytext** (`formats: ipynb,py:percent`):
# edit either file, the other updates on save (`jupytext --sync mixed_usage.py`).
#
# Layout:
# - **Part I — Setup & knobs.** Imports and *every* tunable (parametrization,
#   particles, vMIT bag, n_B/T/η grids, the cases to compare). *Always run.*
# - **Part II — EoS tables.** The all-modes driver `build_mixed_table`
#   (n_B × Y_C × T, n_B × Y_L × S, n_B × Y_C × Y_S × T, …) as a DataFrame.
# - **Part III — Physics plots.** P–n_B, χ–n_B, composition Y_i, phase boundaries
#   vs η, sound speed, C_V/C_P, adiabatic c_ad², baryon susceptibility.
# - **Part IV — TOV structure.** M(R) and tidal Λ(M) across cases (the Maxwell
#   density-jump tidal correction is applied automatically inside `eos.tov`).

# %% [markdown]
# # Part I — Setup & knobs

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eos.dd2 import Parametrization, SpeciesFlags
from eos.vmit.parameters import get_vmit_default, get_vmit_custom
from eos.mixed import mode_A, mode_B, mode_C, mode_D
from eos.mixed.table import MixedTableSpec, build_mixed_table
import eos.mixed.notebook_api as api

# %% [markdown]
# ## I.1 — Knobs (edit here, then Run-All)

# %%
# ---- DD2 parametrization -------------------------------------------------
# from_dd2_defaults() nucleonic; from_dd2y_defaults() hyperons; from_nmp(NMP)
# to pin the nuclear-matter parameters yourself (e.g. L_sym).
PAR = Parametrization.from_dd2_defaults()

# ---- Particles -----------------------------------------------------------
FLAGS = SpeciesFlags(hyperons=False, muons=False)          # nucleons + electrons
# hyperonic:  FLAGS = SpeciesFlags(hyperons=True, phi_field=True, muons=False)

# ---- vMIT quark bag ------------------------------------------------------
# get_vmit_default() is B^{1/4}=180 MeV; lower B pushes the transition down in
# density. With hyperons, use B4≈160 to place it in the hyperonic regime.
VMIT = get_vmit_default()
# VMIT = get_vmit_custom(B4=160.0)

# ---- Grids ---------------------------------------------------------------
NB = api.default_grid(n_lo=0.10, n_hi=1.20, n=56)          # n_B [fm^-3]
ETAS = [0.0, 0.5, 1.0]                                     # η axis for boundaries
T = 0.0                                                    # MeV (cold NS)
T_WARM = 15.0                                              # MeV, for C_V/C_P/c_ad²

# ---- Cases to compare (label, η, spec) -----------------------------------
# Any mode × any η; all overlaid on the P–n_B / χ / sound-speed / TOV figures.
CASES = [
    ("Gibbs (η=0)",   0.0, mode_A()),
    ("η=0.5",         0.5, mode_A()),
    ("Maxwell (η=1)", 1.0, mode_A()),
]

# %% [markdown]
# # Part II — EoS tables (all modes, one driver)
#
# `build_mixed_table` drives **every** mode through the one regime solver. The
# axes dict always has `nB` and exactly one of `T` / `SnB`; add `Y_C` / `Y_S` /
# `Y_L` as axes to sweep a fraction. Returns long-format rows → a DataFrame.

# %%
# β-equilibrium (Mode A) table at two temperatures:
spec_beta = MixedTableSpec(PAR, FLAGS, "beta",
                           axes={"nB": NB[::4], "T": [0.0, T_WARM]},
                           eta=0.0, vmit_params=VMIT)
df_beta = pd.DataFrame(build_mixed_table(spec_beta))
df_beta[["n_B", "T", "chi", "P", "eps", "Y_C"]].head()

# %%
# Fixed-Y_C (Mode C) as n_B × Y_C × T; swap to 'YC+YS' + a 'Y_S' axis for Mode D,
# or 'YL' + 'SnB' for trapped-ν isentropic tables.
spec_yc = MixedTableSpec(PAR, FLAGS, "YC",
                         axes={"nB": NB[::4], "T": [0.0], "Y_C": [0.05, 0.10]},
                         eta=0.0, vmit_params=VMIT)
df_yc = pd.DataFrame(build_mixed_table(spec_yc))
df_yc[["n_B", "Y_C", "chi", "P", "eps"]].head()

# %% [markdown]
# # Part III — Physics plots

# %%
api.plot_p_vs_nb(PAR, FLAGS, NB, CASES, vp=VMIT, T=T)          # P vs n_B
plt.show()

# %%
api.plot_chi_vs_nb(PAR, FLAGS, NB, CASES, vp=VMIT, T=T)        # quark fraction χ
plt.show()

# %%
# Composition of one case (both phases; hadrons solid, quarks dashed):
api.plot_composition(PAR, FLAGS, NB, 0.0, mode_A(), vp=VMIT, T=T)
plt.show()

# %%
api.plot_phase_boundaries(PAR, FLAGS, NB, mode_A(), ETAS, vp=VMIT, T=T)
plt.show()

# %%
api.plot_sound_speed(PAR, FLAGS, NB, CASES, vp=VMIT, T=T)      # equilibrium c_s²
plt.show()

# %%
api.plot_susceptibility(PAR, FLAGS, NB, CASES, vp=VMIT, T=T)   # χ_B = dn_B/dμ_B
plt.show()

# %% [markdown]
# ## III.1 — Thermal response at T>0 (finite-difference on the equilibrium solve)
# C_V/C_P and the adiabatic sound speed need T>0; these re-solve on a small
# stencil, so they are slower than the cold panels.

# %%
api.plot_heat_capacity(PAR, FLAGS, NB[::2], 0.0, mode_A(), vp=VMIT, T=T_WARM)
plt.show()

# %%
api.plot_adiabatic_cs2(PAR, FLAGS, NB[::2], 0.0, mode_A(), vp=VMIT, T=T_WARM)
plt.show()

# %% [markdown]
# # Part IV — TOV structure: M(R) and tidal Λ(M)
#
# One TOV sequence per case (BPS crust + mixed core). The Takátsy–Kovács tidal
# ΔY correction across the Maxwell density discontinuity is applied automatically
# whenever the η=1 table carries a plateau — no flag needed. Uses the trusted
# scipy backend; pass `backend="fast"` for the numba path on big η scans.

# %%
CASES_TOV = {label: api.compute_tov(PAR, FLAGS, NB, eta, spec, vp=VMIT, T=T)
             for label, eta, spec in CASES}

# %%
api.plot_mass_radius(CASES_TOV)
plt.show()

# %%
api.plot_lambda_mass(CASES_TOV)
plt.show()

# %%
# Summary table of the stellar observables per case:
pd.DataFrame([
    dict(case=label, M_max=t["M_max"], R_Mmax=t["R_Mmax"],
         R_1p4=t["R_1p4"], Lambda_1p4=t["Lambda_1p4"])
    for label, t in CASES_TOV.items()
])
