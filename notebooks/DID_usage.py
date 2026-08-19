# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DID / DIDY — usage and validation
#
# `eos.did` implements the relativistic mean field of
#
# > G. Frohaug, K. Maslov, V. Dexheimer, J. Grefa, J. Jahan, C. Ratti and
# > T. E. Restrepo, *Relativistic mean-field model with density- and
# > isospin-density-dependent couplings*, arXiv:2511.15646
#
# a DD-RMF whose baryon–meson couplings depend on the isospin asymmetry
# $\beta = \sum_i \tau_{3i} n_i / n_B$ as well as on the density. That second
# dependence is the point of the model: it lets the hyperon single-particle
# potentials be reproduced in **neutron** matter as well as in symmetric
# matter, and the later hyperon onsets that follow are what keep $M_{\max}$
# above $2\,M_\odot$ with a full octet.
#
# **DID and DIDY are the same parameter set.** What distinguishes them is
# `SpeciesFlags(hyperons=True)`.
#
# This notebook is a tour and a check at once: every section recomputes
# something the paper publishes and prints the difference. If a number here
# drifts, something has regressed.

# %%
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

from eos.did import (
    MULTIPLET_OF, Parameters, SpeciesFlags, compute_nmp, eos_point,
    eos_response, eos_table, nuclear_matter, single_particle_potential,
    solve_beta_eq_neutrinoless, tau3, warm_start,
)
from eos.general.figure_style import (
    STANDARD_COLORS, apply_style, particle_style, set_global_style,
)
from eos.general.modes import beta_eq_neutrinoless
from eos.general.particles import get_particle

set_global_style()

par = Parameters.default()
DID = SpeciesFlags(muons=False)                        # nucleons, electrons
DIDY = SpeciesFlags(hyperons=True, muons=False)        # + the hyperon octet
print(f"n_0 = {par.n_0} fm^-3   g_sigmaN^S = {par.g_sigma_N_S}")

# %% [markdown]
# ## 1. The couplings
#
# $g_{Mi}(n_B,\beta) = [1-w]\,g^S_{Mi} + w\,g^N_{Mi}$ with
# $w = \beta^2\tanh(3x)$ and $x = n_B/n_0$. The shaded band between $\beta=0$
# (ISM) and $\beta=-1$ (NM) is the whole isospin dependence — moderate for the
# isoscalars, much stronger for the $\rho$. This is the paper's Fig. 3.

# %%
x = np.linspace(0.01, 6.0, 300)
n_grid = x * par.n_0

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, meson in zip(axes, ("sigma", "omega", "rho")):
    ism = np.array([par.couplings_at(n, 0.0)[(meson, "N")][0] for n in n_grid])
    nm = np.array([par.couplings_at(n, -1.0)[(meson, "N")][0] for n in n_grid])
    ax.fill_between(x, ism, nm, alpha=0.25, color=STANDARD_COLORS.get("DD2", "C0"))
    ax.plot(x, ism, "-", label=r"ISM  $\beta = 0$")
    ax.plot(x, nm, "--", label=r"NM  $\beta = -1$")
    ax.axvline(1.0, ls=":", lw=0.8, color="0.4")
    ax.set_xlabel(r"$n_B / n_0$")
    ax.set_ylabel(rf"$g_{{\{meson} N}}$")
    apply_style(ax, legend=(meson == "sigma"))
fig.tight_layout()

# %% [markdown]
# ## 2. Saturation, and the pressures the model was fitted to
#
# $n_0$ is calibrated so that $P(n_0)=0$ in symmetric matter — the first line
# below is therefore a test of the whole parameter transcription, not a
# tautology. The four pressures are the $\chi$EFT and heavy-ion keypoints of
# the paper's Table III.

# %%
ism = nuclear_matter(par, par.n_0, 0.0)
print(f"P(n_0) in ISM = {ism.P:+.3e} MeV/fm^3   (calibration: 0)")

for n_B, beta, published, label in ((0.08, -1.0, 0.4569, "P_NM(0.08)"),
                                    (0.16, -1.0, 3.233, "P_NM(0.16)"),
                                    (0.32, 0.0, 12.11, "P_ISM(0.32)"),
                                    (0.56, 0.0, 109.0, "P_ISM(0.56)")):
    P = nuclear_matter(par, n_B, beta).P
    print(f"{label:12s} {P:9.4f}   paper {published:8.4f}")

# %% [markdown]
# ## 3. Nuclear-matter parameters (Table VI)
#
# All predictions of the couplings; only $n_0$ is imposed. Note the two
# symmetry energies: $S - S_2 < 0$ in DID and $> 0$ in DD2 and DDB, which is
# what makes the $\beta$-equilibrium proton fraction 0.034 rather than 0.05.

# %%
nmp = compute_nmp(par)
paper = dict(n_0=0.158800, B=-15.40, K=227.06, Q=-608.09, M=1122.72,
             S_2=32.44, S=29.72, L=59.95, K_sym=-97.32, X_p_eq=0.0336)
for key, published in paper.items():
    print(f"{key:8s} {nmp[key]:12.4f}   paper {published:10.4f}")
print(f"\nS - S_2 = {nmp['S'] - nmp['S_2']:+.2f} MeV   (paper -2.72)")

# %% [markdown]
# ## 4. Hyperon potentials in ISM and in NM (Table IV)
#
# This is what the model exists for. $U_Y$ is evaluated for a **test**
# particle at the medium's fields, so it is meaningful before any hyperon has
# appeared. In neutron matter the $\Sigma$ splitting is carried almost
# entirely by the isospin rearrangement term $\Sigma^t$ — the $\rho$ coupling
# to the $\Sigma$ is 0.0055.

# %%
targets = {"ISM": {"Lambda": -27.87, "Sigma+": 14.99, "Sigma0": 14.99,
                   "Sigma-": 14.99, "Xi0": -3.97, "Xi-": -3.97},
           "NM": {"Lambda": -25.54, "Sigma+": 6.85, "Sigma0": 15.79,
                  "Sigma-": 24.74, "Xi0": -12.13, "Xi-": 5.85}}

for medium, beta in (("ISM", 0.0), ("NM", -1.0)):
    point = nuclear_matter(par, par.n_0, beta)
    couplings = par.couplings_at(point.n_B, point.beta)
    print(f"\n{medium}:  sigma = {point.sigma:.2f}, omega = {point.omega:.2f}, "
          f"rho = {point.rho:.3f}, Sigma^r = {point.Sigma_r:.3f}, "
          f"Sigma^t = {point.Sigma_t:.3f} MeV")
    for name, published in targets[medium].items():
        U = single_particle_potential(couplings, point.fields(),
                                      MULTIPLET_OF[name],
                                      tau3(get_particle(name)), point.Sigma_r)
        print(f"   U_{name:8s} {U:+8.3f}   paper {published:+8.2f}")

# %% [markdown]
# ## 5. Composition in $\beta$ equilibrium, and the inverted hyperon hierarchy
#
# $\Sigma^-$ appears **before** $\Lambda$ (0.470 against 0.578 fm$^{-3}$), and
# $\Xi^0$ never appears. The usual DD-RMF ordering is the other way round; this
# is the paper's Table VII and its headline result.

# %%
n_grid = np.arange(0.1, 1.3, 0.005)
spec = beta_eq_neutrinoless()
rows, x0 = [], None
for n_B in n_grid:
    point = solve_beta_eq_neutrinoless(par, float(n_B), DIDY, T=0.0, x0=x0)
    if not point.converged:
        x0 = None
        continue
    rows.append(point)
    x0 = warm_start(point, spec)

fig, ax = plt.subplots(figsize=(6, 4))
for name in ("n", "p", "Lambda", "Sigma-", "Xi-"):
    colour, linestyle = particle_style(name)
    ax.plot([p.n_B for p in rows], [p.Y(name) for p in rows],
            color=colour, ls=linestyle, label=name)
ax.plot([p.n_B for p in rows], [p.Y_e for p in rows], ls=":", color="0.4",
        label=r"$e^-$")
ax.set_yscale("log")
ax.set_ylim(1e-4, 1.2)
ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax.set_ylabel(r"$Y_i$")
apply_style(ax)

onsets = {}
for point in rows:
    for name in ("Sigma-", "Lambda", "Xi-", "Xi0"):
        if name not in onsets and point.Y(name) > 1e-6:
            onsets[name] = point.n_B
print("onsets [fm^-3]:", {k: round(v, 3) for k, v in onsets.items()},
      " paper: Sigma- 0.470, Lambda 0.578, Xi- 0.978, no Xi0")

# %% [markdown]
# ## 6. The speed of sound (Fig. 8)
#
# Non-monotonic, peaking near 0.71 at about $4\,n_0$ and coming back down — a
# purely hadronic model showing a feature usually read as the signature of an
# exotic phase.

# %%
n_cs = np.arange(0.2, 1.3, 0.02)
cs2 = [eos_response(par, "beta_eq_neutrinoless", DIDY, n_B=float(n), T=0.0)
       ["cs2_isothermal"] for n in n_cs]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(n_cs, cs2, label="DIDY")
ax.axhline(1 / 3, ls=":", color="0.4", label=r"$c_s^2 = 1/3$")
ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax.set_ylabel(r"$c_s^2$")
apply_style(ax)
print(f"peak c_s^2 = {max(cs2):.3f} at n_B = {n_cs[int(np.argmax(cs2))]:.2f} "
      f"fm^-3   (paper ~0.71 at ~0.66)")

# %% [markdown]
# ## 7. A finite-temperature table
#
# The simulation-table mode: fixed non-leptonic charge fraction with
# neutralizing leptons, on a $(n_B, T, Y_C)$ grid. Any temperature axis may be
# replaced by an entropy per baryon (`SnB`), which makes $T$ an unknown of the
# same solve.

# %%
table = eos_table(par, "fixed_YC", DIDY,
                  axes={"nB": np.geomspace(0.05, 1.2, 60),
                        "T": [0.1, 10.0, 30.0], "Y_C": [0.1, 0.3, 0.5]},
                  leptons=True, verbose=True)

fig, ax = plt.subplots(figsize=(6, 4))
for conditions, line in zip(table.lines, table.points):
    if conditions["Y_C"] != 0.3:
        continue
    ax.plot([p.n_B for p in line], [p.P for p in line],
            label=rf"$T = {conditions['T']:g}$ MeV")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$n_B$ [fm$^{-3}$]")
ax.set_ylabel(r"$P$ [MeV fm$^{-3}$]")
apply_style(ax)

# %%
isentrope = eos_point(par, "fixed_YC", DIDY, n_B=0.3, SnB=2.0, Y_C=0.3,
                      leptons=True)
print(f"S/A = 2 at n_B = 0.3, Y_C = 0.3  ->  T = {isentrope.point.T:.3f} MeV, "
      f"s/n_B = {isentrope.point.entropy_per_baryon:.6f}")

# %% [markdown]
# ## 8. Mass–radius
#
# `eos.did.verify.tov` builds the cold $\beta$-equilibrium table, attaches the
# BPS crust and runs the repository's TOV solver. The paper's Table VIII gives
# $M_{\max} = 2.245$ (DID) and $2.196\,M_\odot$ (DIDY) with $R_{1.4} = 11.99$
# km; the radius differs here by the crust model, which is theirs (NSE) rather
# than BPS.

# %%
from eos.did.verify.tov import mass_radius

for label, flags in (("DID ", DID), ("DIDY", DIDY)):
    out = mass_radius(par, flags)
    print(f"{label}: M_max = {out['M_max']:.3f} M_sun, "
          f"R(M_max) = {out['R_Mmax']:.2f} km, R_1.4 = {out['R_1p4']:.2f} km")

# %% [markdown]
# ## 9. Pairing DID with a quark phase
#
# `eos.mixed` couples any two declared phases through the phase-adapter
# contract, so DID pairs with the vector-bag quark engine (or any other quark
# adapter) without new engine code. `chi` outside $[0,1]$ means the density is
# on a pure-phase side of the transition — that is how the engine reports it.

# %%
from eos.mixed import did_phase, vmit_phase
from eos.mixed.api import eos_point as mixed_point

pair = (did_phase(par, DID), vmit_phase())
for n_B in (0.4, 0.6, 0.8, 1.0):
    result = mixed_point(None, "beta_eq_neutrinoless", None, n_B=n_B, T=0.0,
                         eta=0.0, phases=pair, muons=False)
    if result.ok:
        print(f"n_B = {n_B:.2f}: chi = {result.point.chi:+.4f}, "
              f"P = {result.point.P:8.3f} MeV/fm^3")
