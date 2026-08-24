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
# # CCDM — chiral colour-dielectric quark matter
#
# `eos.ccdm` implements a mean-field model of deconfined $u,d,s$ matter in
# which **confinement and chiral symmetry breaking are two faces of one
# mechanism**:
#
# > A. Drago, M. Fiolhais and U. Tambini, *Quark matter in the chiral
# > colour-dielectric model*, Nucl. Phys. A **588**, 801 (1995);
# > S. K. Ghosh and S. C. Phatak, Phys. Rev. C **52**, 2195 (1995).
#
# A dilaton field carries the gluon condensate; the dielectric function built
# from it measures how transparent the medium is to colour, and it sits in the
# **denominator** of the quark masses,
#
# $$\chi = (1-\bar\varphi^{\,4})^p,\qquad
#   M^*_{u,d} = \frac{g_q\sigma + m_{u,d}}{\chi},\qquad
#   M^*_s = \frac{g_s\zeta + m_s}{\chi}.$$
#
# So as the condensate reaches its vacuum value the medium goes opaque, the
# effective masses diverge, and the quarks leave the medium entirely. That is
# not a suppression to be smoothed: **at $T=0$ a mode with $M^*\ge\mu^*$
# contributes identically zero**, and it is what makes deconfinement first
# order here rather than a crossover.
#
# The pairing machinery lives in `eos.general.pairing`, shared with the NJL
# model, because the pairing sector of the two is the same sector.
#
# This notebook is a tour and a check at once: every section recomputes
# something the specification states and prints the difference. If a number
# here drifts, something has regressed.

# %%
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

from eos.general.figure_style import STANDARD_COLORS, set_global_style
from eos.general.pairing import pair_block
from eos.general.physics_constants import hc3
from eos.ccdm import (
    Parameters, SpeciesFlags, bag_constant, chiral_potential, dielectric,
    effective_masses, eos_point, eos_response, eos_table, glue_potential,
    mode_thermo, solve, thermo_from_mu,
)

set_global_style()

par = Parameters.default()
PLAIN = SpeciesFlags(csc=False)          # unpaired
CSC = SpeciesFlags(csc=True)             # + the diquark sector
d = par.derived
print(f"B_g^(1/4) = {par.B_g_quarter} MeV   g_q = {par.g_q}   g_s = {par.g_s}   "
      f"m_sigma = {par.m_sigma} MeV")
print(f"gbar_omega = {par.gbar_omega}   n_c = {par.n_c} fm^-3   "
      f"G_D = {par.G_D:.1e} MeV^-2   Lambda = {par.Lambda} MeV")

# %% [markdown]
# ## 1. The vacuum-fixed block
#
# Nothing in the scalar sector is fitted. Given $f_\pi$, $m_\pi$, $f_K$, $m_K$
# and $m_\zeta$, every constant of the Mexican hat follows in closed form, and
# $C_0$ is whatever puts $V(\sigma_0,\zeta_0)=0$ — which is why $\Omega$ needs
# **no vacuum subtraction anywhere**, unlike the NJL companion.
#
# Note $v_\zeta^2 < 0$ at the baseline $m_\zeta$: the strange quartic is
# convex, explicit breaking dominating. Never write $v_\zeta$ as a square root.

# %%
published = {"zeta_0": 94.05, "lam": 16.39, "lam_zeta": 31.41,
             "v_zeta2": -4039.0, "C_0": 2.435e9}
for name, value in published.items():
    got = getattr(d, name)
    print(f"  {name:10s} {got:14.4f}   specification {value:12.4g}")
print(f"  {'v':10s} {np.sqrt(d.v2):14.4f}   specification {86.53:12.4g}")
print(f"  {'phi_0':10s} {d.phi_0:14.4f} MeV")
print(f"\n  v_zeta^2 is NEGATIVE: {d.v_zeta2 < 0}")

# %% [markdown]
# ## 2. The bag constant is derived, and the chiral sector supplies most of it
#
# $B_{\rm eff} = [U(0)-U(\varphi_0)] + [V(0,0)-V(f_\pi,\zeta_0)]
#              = B_g + B_\chi$.
#
# Quoting $B_g$ alone as "the bag constant" of this model is wrong by a factor
# of six in energy density.

# %%
B_eff = bag_constant(par)
B_chi = B_eff - par.B_g
print(f"  B_g^(1/4)    = {par.B_g ** 0.25:7.2f} MeV")
print(f"  B_chi^(1/4)  = {B_chi ** 0.25:7.2f} MeV   <-- the larger part")
print(f"  B_eff^(1/4)  = {B_eff ** 0.25:7.2f} MeV  = {B_eff / hc3:7.1f} MeV/fm^3"
      f"   (specification: (240 MeV)^4 = 429 MeV/fm^3)")
print(f"\n  V(sigma_0, zeta_0) = {chiral_potential(par, d.sigma_0, d.zeta_0):.3e}"
      f"   (must be 0)")

# %% [markdown]
# ## 3. The dielectric, and confinement as a pinning
#
# The left panel is $\chi(\bar\varphi)$ and the effective masses it produces
# in the confined branch. The right one is the mechanism itself: a mode whose
# mass has run above its own potential contributes **exactly** zero, not a
# small number. Smoothing that threshold destroys the first-order transition.

# %%
phi_bar = np.linspace(0.0, 0.98, 300)
chi = np.array([dielectric(par, p ** 4) for p in phi_bar])
M_ud = np.array([effective_masses(par, p ** 4, d.sigma_0, d.zeta_0)[0]
                 for p in phi_bar])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(phi_bar, chi, color=STANDARD_COLORS["Blue"], label=r"$\chi$")
axes[0].set_xlabel(r"$\bar\varphi$")
axes[0].set_ylabel(r"$\chi = (1-\bar\varphi^{4})^{p}$")
twin = axes[0].twinx()
twin.semilogy(phi_bar, M_ud, color=STANDARD_COLORS["Red"], ls="--")
twin.set_ylabel(r"$M^*_{u,d}$ [MeV] (confined branch)",
                color=STANDARD_COLORS["Red"])
axes[0].set_title("the medium closes, and the quarks get heavy")

# the specification's section 10 table: g_q = 3.0 inverted from these
for p, expected in ((0.90, 826.0), (0.95, 1531.0)):
    got = effective_masses(par, p ** 4, d.sigma_0, d.zeta_0)[0]
    print(f"  phi_bar = {p}:  M*_(u,d) = {got:8.1f} MeV   "
          f"specification {expected:7.1f}")

mu_star = np.linspace(300.0, 600.0, 400)
n = np.array([mode_thermo(m, 450.0, 0.0).n for m in mu_star])
axes[1].plot(mu_star, n / hc3, color=STANDARD_COLORS["Green"])
axes[1].axvline(450.0, color="0.5", lw=0.8, ls=":")
axes[1].set_xlabel(r"$\mu^*$ [MeV]")
axes[1].set_ylabel(r"$n$ [fm$^{-3}$], one mode at $M^*=450$ MeV, $T=0$")
axes[1].set_title("the pinning: identically zero below threshold")
fig.tight_layout()

print(f"\n  a mode at M* = 450 > mu* = 400 MeV returns "
      f"{mode_thermo(400.0, 450.0, 0.0).n} exactly")

# %% [markdown]
# ## 4. Why the solve variable is $\Phi = \bar\varphi^{\,4}$
#
# Written in $\bar\varphi$ the dilaton residual has a **spurious root at
# $\bar\varphi=0$**: both of its terms vanish as $\bar\varphi^{\,3}$ there, so
# it is satisfied for *any* scalar density. It is an artefact of the
# parametrisation — the Jacobian $\mathrm d\Phi/\mathrm d\bar\varphi =
# 4\bar\varphi^{\,3}$ vanishes, not the physics. In $\Phi$ the same equation
# reads $\mathrm dU/\mathrm d\Phi = B_g\ln\Phi$, which runs to $-\infty$ there.

# %%
Phi = np.logspace(-6, -0.01, 300)
dU_dPhi = par.B_g * np.log(Phi)
dU_dphibar = 16.0 * par.B_g * Phi ** 0.75 * np.log(Phi ** 0.25)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Phi ** 0.25, dU_dphibar / par.B_g, color=STANDARD_COLORS["Blue"],
        label=r"$\partial U/\partial\bar\varphi\;/\;B_g$  (spurious root at 0)")
ax.plot(Phi ** 0.25, dU_dPhi / par.B_g, color=STANDARD_COLORS["Red"],
        label=r"$\partial U/\partial\Phi\;/\;B_g = \ln\Phi$  (no root)")
ax.axhline(0.0, color="0.5", lw=0.8)
ax.set_xlabel(r"$\bar\varphi$")
ax.set_ylabel("glue-potential derivative")
ax.set_ylim(-8, 2)
ax.legend()
ax.set_title("the same equation, two parametrisations")
fig.tight_layout()

# %% [markdown]
# ## 5. The equation of state, and where the model is defined
#
# Below the deconfinement onset there is **no deconfined root at fixed
# density at all** — the quarks are not in the medium. That comes back as a
# status, not an exception. The pressure crosses zero near
# $n_B \simeq 1.35$ fm$^{-3}$, and just below that $\mathrm dP/\mathrm dn_B<0$:
# the mechanically unstable side of a first-order transition, which a
# construction (`eos.mixed`) removes before any table reaches a structure
# solver.

# %%
grid = np.linspace(1.0, 2.4, 29)
table = eos_table(par, "beta_eq_neutrinoless", PLAIN,
                  axes={"nB": grid, "T": [0.0, 30.0]}, verbose=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for line, conditions, colour in zip(table.points, table.lines,
                                    (STANDARD_COLORS["Blue"],
                                     STANDARD_COLORS["Red"])):
    n_B = np.array([p.n_B for p in line])
    label = rf"$T = {conditions['T']:g}$ MeV"
    axes[0].plot(n_B, [p.P_total for p in line], color=colour, label=label)
    axes[1].plot(n_B, [p.phi_bar for p in line], color=colour, label=label)
    axes[2].plot(n_B, [p.Y_s for p in line], color=colour, label=label)
    axes[2].plot(n_B, [p.Y_u for p in line], color=colour, ls="--")

axes[0].axhline(0.0, color="0.5", lw=0.8)
axes[0].set_xlabel(r"$n_B$ [fm$^{-3}$]")
axes[0].set_ylabel(r"$P$ [MeV fm$^{-3}$]")
axes[0].set_title("the onset is where $P$ crosses zero")
axes[1].set_xlabel(r"$n_B$ [fm$^{-3}$]")
axes[1].set_ylabel(r"$\bar\varphi$")
axes[1].set_title("the dilaton melts with density")
axes[2].set_xlabel(r"$n_B$ [fm$^{-3}$]")
axes[2].set_ylabel(r"$Y_s$ (solid), $Y_u$ (dashed)")
axes[2].set_title("strangeness switches on")
for ax in axes:
    ax.legend()
fig.tight_layout()

print("\nbelow the onset the model reports honestly rather than inventing:")
print("  ", eos_point(par, "beta_eq_neutrinoless", PLAIN, n_B=0.4, T=0.0).message)

# %% [markdown]
# ## 6. The audit that caught two errors in the specification
#
# $\varepsilon + P = Ts + \sum_j\mu_jn_j$ must hold at every solved point. It
# is what says that
#
# * the vector field energy enters $\varepsilon$ with a **plus** sign (the
#   specification's §4.3 writes minus — a repulsive interaction *adds* energy);
# * $\Omega$ carries $-\Sigma_R n_q$, the rearrangement term the specification
#   puts in $\mu^*$ without its counterpart.
#
# With either error the residual is of order percent while the equation of
# state still looks entirely reasonable.

# %%
print(" mode                       n_B     T        Euler residual")
for mode, kwargs in (("beta_eq_neutrinoless", dict(T=0.0)),
                     ("beta_eq_neutrinoless", dict(T=30.0)),
                     ("beta_eq_neutrino_trapped", dict(T=20.0, Y_Le=0.3)),
                     ("fixed_YC", dict(T=20.0, Y_C=0.1)),
                     ("fixed_YC", dict(T=20.0, Y_C=0.1, leptons=False))):
    r = eos_point(par, mode, PLAIN, n_B=1.5, **kwargs)
    tag = mode + ("" if kwargs.get("leptons", True) else " (charged)")
    print(f"  {tag:32s} {r.point.n_B:5.2f} {r.point.T:5.1f}   "
          f"{r.point.state.euler_residual():+.2e}")

st, ok, _ = thermo_from_mu(par, 1450.0, -30.0, 0.0, 0.0, branch="restored")
print(f"\n  Sigma_R = {st.Sigma_R:8.2f} MeV puts "
      f"{st.Sigma_R * st.n_q / hc3:8.1f} MeV/fm^3 into P and nothing into eps")
print(f"  (it is {abs(st.Sigma_R * st.n_q / st.eps):.1%} of eps, so neither "
      f"identity is passing by being small)")

# %% [markdown]
# ## 7. Colour superconductivity
#
# Which pairing pattern the matter is in is an **outcome**, decided by free
# energy among the enumerated candidates — and so is the chiral/dielectric
# **branch**. The candidate set is the product of the two, because which
# pattern survives depends on the strange quark's effective mass, which is a
# property of the branch.
#
# The colour potentials $\mu_3,\mu_8$ are solved *inside* the phase: they are
# not conserved charges of a mixed system, so `eos.mixed` never learns they
# exist.

# %%
print(" n_B    T   branch    pattern   |Delta| [MeV]           mu_8     n_3, n_8")
for n_B, T in ((1.4, 0.0), (1.5, 0.0), (1.6, 30.0)):
    p = solve("beta_eq_neutrinoless", n_B, T, par, CSC)
    print(f" {n_B:4.1f} {T:5.1f}  {p.branch:9s} {p.pattern:8s} "
          f"{np.round(p.Delta, 1)}  {p.mu_8:7.2f}  "
          f"{p.state.n_3:+.1e} {p.state.n_8:+.1e}")

# %% [markdown]
# ### The gap sign is a gauge
#
# $\Omega$ is invariant under flipping any subset of the three gaps, and each
# kernel flips with its own gap — so $-\Delta$ is a root whenever $\Delta$ is,
# and the solve lands on whichever the seed was nearest. What is *reported* is
# the magnitude.

# %%
M = np.array([8.0, 8.0, 300.0])
mu = np.full(9, 470.0)
base = pair_block(M, mu, np.full(3, 60.0), 20.0, par.Lambda)
print(f"  (+,+,+):  delta_Omega = {base.delta_omega:.10e}")
for signs in ((1, -1, -1), (1, 1, -1), (-1, -1, -1)):
    b = pair_block(M, mu, 60.0 * np.array(signs, float), 20.0, par.Lambda)
    print(f"  {str(signs):10s} delta_Omega = {b.delta_omega:.10e}   "
          f"relative difference {abs(b.delta_omega / base.delta_omega - 1):.1e}")

# %% [markdown]
# ### A gapped phase freezes out — but only where the Fermi surfaces match
#
# The entropy suppression is a property of the **spectrum**, not of the gap
# parameter. A large strange mass mismatches the Fermi surfaces, pushes the
# lowest branch down, and most of the suppression goes away. That is the
# difference between a strange quark star that cools like a superconductor and
# one that does not.

# %%
print("  M*                 lowest branch [MeV]   s_paired/s_unpaired at T = 5 MeV")
for M_star in ([8.0, 8.0, 8.0], [8.0, 8.0, 150.0], [8.0, 8.0, 300.0]):
    Ms = np.array(M_star)
    block = pair_block(Ms, mu, np.full(3, 60.0), 5.0, par.Lambda)
    unpaired = sum(mode_thermo(mu[j], Ms[j // 3], 5.0).s for j in range(9))
    ratio = abs(unpaired + block.delta_s) / abs(unpaired)
    print(f"  {str(M_star):20s} {block.min_energy:8.2f}          {ratio:.3e}")

# %% [markdown]
# ## 8. Sound speed — one-sided, on each branch
#
# The transition is first order, so a central difference straddling it returns
# the chord across the jump rather than a tangent to either branch.
# `eos_response` returns `branch_changed` for exactly that reason: there is no
# way to see it from the number alone.

# %%
print("  n_B    cs2_isothermal   branch_changed")
for n_B in (1.4, 1.6, 1.8, 2.0, 2.2):
    r = eos_response(par, "beta_eq_neutrinoless", PLAIN, n_B=n_B, T=0.0)
    print(f"  {n_B:4.1f}   {r['cs2_isothermal']:+.4f}          "
          f"{r['branch_changed']}")

# %% [markdown]
# ## 9. Pairing it with a hadronic phase
#
# The adapter is the only surface `eos.mixed` touches. It closes colour
# neutrality inside, chooses branch and pattern by pressure, and reports both
# as fields — a mixed table that does not say which quark phase it found is
# not reporting its own result.
#
# The confined branch is excluded by default: its pressure is exactly zero, so
# it is the vacuum, and in a hybrid construction the hadronic phase occupies
# that side of the transition.

# %%
from eos.mixed import beta_eq_neutrinoless, locate_window
from eos.mixed.adapters import ccdm_phase, did_phase
from eos.did import Parameters as DIDParameters, SpeciesFlags as DIDFlags

phases = (did_phase(DIDParameters.default(), DIDFlags(muons=True)),
          ccdm_phase(par, PLAIN))
th = phases[1].thermo(1450.0, -30.0, 0.0, 0.0)
print(f"  branch  = {th.fields['branch']}   pattern = {th.fields['pattern']}")
print(f"  phi_bar = {th.fields['phi_bar']:.4f}   chi = {th.fields['chi_diel']:.4f}")
print(f"  n_B = {th.n_B:.4f} fm^-3   P = {th.P:.2f} MeV/fm^3")
print(f"  Euler through the block: {th.eps + th.P - th.mu_dot_n:+.2e}")

# NOTE: locating a DID+CCDM window is minutes of work -- every adapter call is
# a full internal Newton solve and the seed cannot be cached, because it
# chooses the branch. Uncomment to run it.
#
# window = locate_window(None, None, np.linspace(0.6, 2.6, 21), 0.0,
#                        beta_eq_neutrinoless(), phases=phases, muons=True,
#                        n_probe=8, refine="bisect")
# print(f"  window: {window.n_onset:.3f} -> {window.n_offset:.3f} fm^-3")

# %% [markdown]
# ## 10. The invariant suite
#
# Everything above, plus the reduction chain and the calibration gates, is in
# `eos.ccdm.verify`. Run it after any change to the solver.
#
# ```
# python -m eos.ccdm.verify.run_full_check
# ```

# %%
from eos.ccdm.verify import run_all

print(run_all(par, include_csc=False))
