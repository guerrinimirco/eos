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
# # NJL — three-flavour quark matter with colour superconductivity
#
# `eos.njl` implements the standard three-flavour Nambu–Jona-Lasinio model with
# a diquark condensate:
#
# > P. Rehberg, S. P. Klevansky and J. Hüfner, *Hadronization in the SU(3) NJL
# > model*, Phys. Rev. C **53**, 410 (1996) — the parameter set;
# > S. B. Rüster *et al.*, Phys. Rev. D **72**, 034004 (2005) — the neutral
# > three-flavour pairing sector.
#
# Almost nothing in it is an input. The constituent masses come out of a gap
# equation, the effective bag constant is a *derived* vacuum pressure
# difference, and **which colour-superconducting pattern the matter is in is an
# outcome chosen by free energy**, not a declaration.
#
# The pairing machinery itself lives in `eos.general.pairing`, shared with the
# colour-dielectric model, because the pairing sector of the two is the same
# sector.
#
# This notebook is a tour and a check at once: every section recomputes
# something published and prints the difference. If a number here drifts,
# something has regressed.

# %%
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

from eos.general.figure_style import (
    STANDARD_COLORS, apply_style, set_global_style,
)
from eos.general.pairing import (
    delta_omega_pair, gap_matrix, gap_residuals, gap_roots, pair_block,
)
from eos.njl import (
    Parameters, SpeciesFlags, bag_constant, effective_exponent, eos_point,
    eos_response, eos_table, kinetic_thermo, solve, surface_term,
    vacuum_solution, vector_coupling,
)

set_global_style()

par = Parameters.default()
PLAIN = SpeciesFlags(csc=False)          # ordinary unpaired NJL
CSC = SpeciesFlags(csc=True)             # + the diquark sector
print(f"Lambda = {par.Lambda} MeV   G_S Lambda^2 = {par.GS_Lambda2}   "
      f"K Lambda^5 = {par.K_Lambda5}   eta_D = {par.eta_D}")

# %% [markdown]
# ## 1. The vacuum
#
# The gap equation
# $M_u = m_u - 4G_S\phi_u + 2K\phi_d\phi_s$ (and cyclic) is solved by a
# **damped fixed point on the masses**, not by a root finder on the
# condensates — the latter diverges and returns masses that increase with
# density. Everything below is reproduced with no fitting at all.

# %%
vac = vacuum_solution(par)
published = {"M_u": 367.7, "M_s": 549.5, "(-phi_u)^(1/3)": 241.9,
             "(-phi_s)^(1/3)": 257.7, "f_pi": 92.4}
computed = {"M_u": vac.M[0], "M_s": vac.M[2],
            "(-phi_u)^(1/3)": (-vac.phi[0]) ** (1 / 3),
            "(-phi_s)^(1/3)": (-vac.phi[2]) ** (1 / 3), "f_pi": vac.f_pi}
for name, value in published.items():
    print(f"  {name:16s} {computed[name]:9.3f}   published {value:7.1f}")

B = bag_constant(par)
print(f"\n  B_eff^(1/4)      {B ** 0.25:9.2f} MeV  = "
      f"{B / 197.3269804 ** 3:.2f} MeV/fm^3   (a DERIVED quantity)")
print(f"  Omega_vac - eps_vac = {vac.Omega - vac.eps}  (must be exactly 0)")

# %% [markdown]
# ## 2. The surface term — why $P$ comes from the logarithm form
#
# The two standard pressure integrals differ by a boundary term that does
# **not** vanish when the integral is cut. At $T=0$ below the cutoff they
# agree, which is exactly how the error hides until a table is built at finite
# temperature.

# %%
print("   M     mu     T  |  P_log      P_log - P_k4   share of P")
for M, mu, T in ((100., 500., 20.), (40., 590., 30.),
                 (140., 700., 5.), (140., 700., 50.)):
    r = kinetic_thermo(mu, M, T, par.Lambda)
    difference = r.P - r.P_k4
    closed = surface_term(mu, M, T, par.Lambda)
    print(f"  {M:5.0f} {mu:6.0f} {T:5.1f}  |  {r.P:.4e}   {difference:.4e}   "
          f"{difference / r.P:6.1%}   (closed form agrees to "
          f"{abs(closed / difference - 1):.1e})")

# %% [markdown]
# ## 3. The gap matrix, and why multiplicities are derived
#
# $G_{(fa),(gb)} = \sum_\eta \Delta_\eta\,\epsilon^{ab\eta}\epsilon_{fg\eta}$.
# Its eigenvalue multiplicities are a **property of the pattern**, never
# assigned by hand — and with independent gaps the $\pm\sqrt2\Delta$ of uSC
# generalises to $\pm\sqrt{\Delta_2^2+\Delta_3^2}$.

# %%
for name, Delta in (("unpaired", (0, 0, 0)), ("2SC", (0, 0, 60)),
                    ("CFL", (60, 60, 60)), ("uSC", (0, 60, 60)),
                    ("dSC", (60, 0, 60)), ("free", (0, 40, 70))):
    spectrum = np.round(np.linalg.eigvalsh(gap_matrix(Delta)), 4)
    print(f"  {name:9s} Delta={str(Delta):14s} {spectrum}")

# %% [markdown]
# ## 4. The gap equation has three roots
#
# With a Fermi-surface mismatch $R(\Delta)$ vanishes at $\Delta=0$, at a
# *barrier* maximum, and at the physical BCS root. A fixed bracket handed to
# `brentq` returns whichever it happens to contain, silently — so scan, then
# bracket each sign change.
#
# The free-energy crossover is the Clogston–Chandrasekhar limit, recovered at
# $0.970$ of the weak-coupling $\Delta_0/\sqrt2$; the 3 % deficit is the finite
# cutoff.

# %%
G_D = par.G_D
M_light = np.full(3, 5.5)


def residual_at(Delta, dmu):
    mu = np.concatenate([np.full(3, 450. - dmu), np.full(3, 450. + dmu),
                         np.full(3, 450.)])
    block = pair_block(M_light, mu, (0, 0, Delta), 0.0, par.Lambda,
                       nodes_per_panel=48)
    return float(gap_residuals((0, 0, Delta), G_D, block)[2]), block


grid = np.linspace(1.0, 140.0, 70)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for dmu, colour in zip((0.0, 50.0, 65.0), STANDARD_COLORS):
    values = np.array([residual_at(d, dmu)[0] for d in grid])
    ax.plot(grid, values / 1e7, color=colour,
            label=rf"$\delta\mu = {dmu:.0f}$ MeV")
    for root in gap_roots(lambda d: residual_at(d, dmu)[0], 150., n_scan=60):
        ax.plot(root, 0.0, "o", color=colour, ms=5)
        print(f"  dmu = {dmu:5.1f} MeV   root at Delta = {root:6.2f} MeV")
ax.axhline(0.0, color="0.5", lw=0.8)
ax.set_xlabel(r"$\Delta_3$  [MeV]")
ax.set_ylabel(r"$R(\Delta)\ [10^{7}\,\mathrm{MeV}^3]$")
ax.legend()
apply_style(ax)
fig.tight_layout()

# %% [markdown]
# ## 5. The solved anchor points
#
# Neutral quark matter at $\mu_B = 1500$ MeV, $T = 0$, $\eta_D = 0.75$,
# unpaired and 2SC. Pairing lowers the free energy and the pressure rises from
# 302 to 325 MeV/fm³; $\mu_3 = 0$ as the $r\leftrightarrow g$ symmetry of the
# pattern requires, and $\mu_8$ is the small colour potential the phase needs
# in order to be colour neutral.

# %%
for label, n_B, flags, patterns in (("unpaired", 1.4319, PLAIN, None),
                                    ("2SC", 1.4887, CSC, ("2SC",))):
    p = solve("beta_eq_neutrinoless", n_B, 0.0, par, flags, patterns=patterns)
    print(f"  {label:9s} mu_B = {p.mu_B:7.2f}  mu_C = {p.mu_C:7.2f}  "
          f"mu_8 = {p.mu_8:6.2f}  Delta_3 = {p.Delta[2]:6.2f}")
    print(f"            M = ({p.M[0]:.2f}, {p.M[1]:.2f}, {p.M[2]:.2f}) MeV   "
          f"n_B = {p.n_B:.4f} fm^-3   P = {p.P_total:.2f} MeV/fm^3")
    print(f"            Euler residual {p.state.euler_residual():.1e}   "
          f"colour n_3/n_q = {abs(p.state.n_3 / p.state.n_q):.1e}")

# %% [markdown]
# ## 6. The pattern is an outcome, not an input
#
# `solve` enumerates candidates — unpaired, 2SC, CFL and one asymmetric free
# seed — solves each to self-consistency, and returns the one with the lowest
# $f = \varepsilon - Ts$. Every point reports the winner, the three gaps, the
# two colour potentials and whether the state is gapless.
#
# This cell is the slow one in the notebook: a paired solve diagonalises a
# batch of 18×18 matrices at every quadrature node.

# %%
densities = np.array([1.0, 1.2, 1.4, 1.6])
rows = []
for n_B in densities:
    chosen = solve("beta_eq_neutrinoless", n_B, 0.0, par, CSC)
    per_pattern = {}
    for name in ("unpaired", "2SC", "CFL"):
        candidate = solve("beta_eq_neutrinoless", n_B, 0.0, par, CSC,
                          patterns=(name,))
        per_pattern[name] = candidate.f_total if candidate.converged else np.nan
    rows.append((n_B, chosen.pattern, per_pattern))
    print(f"  n_B = {n_B:.1f}  winner = {chosen.pattern:9s}  "
          f"Delta = ({chosen.Delta[0]:5.1f}, {chosen.Delta[1]:5.1f}, "
          f"{chosen.Delta[2]:5.1f})  mu_8 = {chosen.mu_8:7.2f}  "
          f"gapless = {chosen.gapless}")

fig, ax = plt.subplots(figsize=(5.2, 3.6))
for name, colour in zip(("unpaired", "2SC", "CFL"), STANDARD_COLORS):
    reference = np.array([r[2]["unpaired"] for r in rows])
    values = np.array([r[2][name] for r in rows])
    ax.plot(densities, values - reference, "o-", color=colour, label=name)
ax.axhline(0.0, color="0.5", lw=0.8)
ax.set_xlabel(r"$n_B$  [fm$^{-3}$]")
ax.set_ylabel(r"$f - f_{\rm unpaired}$  [MeV fm$^{-3}$]")
ax.legend(title="pattern")
apply_style(ax)
fig.tight_layout()

# %% [markdown]
# ## 7. A table, and the density ceiling of the sharp cutoff
#
# With every integral cut at $\Lambda$ the nine modes can hold no more than
# $n_B = \Lambda^3/\pi^2 = 2.881$ fm$^{-3}$, and the density freezes there.
# That ceiling is the regularization's, not the solver's, and it is why the
# conformal asymptotics of the vector sector cannot be exhibited at
# $\lambda = \Lambda_{\rm UV}/\Lambda = 1$ at all — reaching them needs
# RG-consistent regularization (see `docs/DEFERRED.md`).

# %%
ceiling = par.Lambda ** 3 / np.pi ** 2 / 197.3269804 ** 3
table = eos_table(par, "beta_eq_neutrinoless", PLAIN,
                  axes={"nB": np.linspace(0.6, 2.95, 26), "T": [0.0, 30.0]},
                  verbose=True)
print(f"\n  ceiling Lambda^3/pi^2 = {ceiling:.3f} fm^-3   "
      f"highest solved = {max(p.n_B for p in table.points[0]):.3f} fm^-3")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
for conditions, line, colour in zip(table.lines, table.points,
                                    STANDARD_COLORS):
    n_B = np.array([p.n_B for p in line])
    axes[0].plot(n_B, [p.P_total for p in line], color=colour,
                 label=rf"$T = {conditions['T']:.0f}$ MeV")
    axes[1].plot(n_B, [p.Y_s for p in line], color=colour)
for ax in axes:
    ax.axvline(ceiling, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel(r"$n_B$  [fm$^{-3}$]")
    apply_style(ax)
axes[0].set_ylabel(r"$P$  [MeV fm$^{-3}$]")
axes[1].set_ylabel(r"$Y_s$")
axes[0].legend()
fig.tight_layout()

# %% [markdown]
# ## 8. The vector coupling and the conformal exponent
#
# With chiral symmetry restored the high-density behaviour is set entirely by
# the vector term, and $c_s^2 \to \max(1-\alpha,\ 1/3)$ for
# $G_V \sim n^{-\alpha}$. A **constant** $G_V$ has $\alpha = 0$ and sends
# $c_s^2$ to 1 (Zel'dovich); the gluon-exchange form reaches the marginal
# $\alpha = 2/3$ with no tuning at all.
#
# Once $G_V$ depends on the density the rearrangement term is mandatory:
# $\Sigma_V = dW/dn = (2-\alpha)G_V n$, not $2G_V n$.

# %%
gluon = Parameters.named("gluon_exchange")
print("   n_q [MeV^3]   G_V/G_S    alpha_eff")
for n_q in (1e6, 1e8, 1e9, 1e10, 1e14):
    print(f"   {n_q:.0e}      {vector_coupling(gluon, n_q) / gluon.G_S:7.4f}"
          f"    {effective_exponent(gluon, n_q):7.4f}")

response = {n: eos_response(par, "beta_eq_neutrinoless", PLAIN, n_B=n, T=0.0)
            for n in (1.0, 1.5, 2.0)}
print("\n   n_B      c_s^2 (isothermal)")
for n, out in response.items():
    print(f"   {n:.1f}      {out['cs2_isothermal']:.4f}")

# %% [markdown]
# ## 9. The phase-adapter surface
#
# `eos.mixed` consumes this model through one function: given
# $(\mu_B,\mu_C,\mu_S,T)$ it returns the phase block, having closed the
# model's own internal system — masses, gaps, $\Sigma_V$ **and the two colour
# potentials**. Colour neutrality is internal because $\mu_3$ and $\mu_8$ are
# not conserved charges of the mixed system: no hadronic phase carries them
# and there is nothing across the interface for them to equilibrate with.

# %%
from eos.mixed import njl_phase

phase = njl_phase(par, PLAIN)
block = phase.thermo(1500.0, -34.20, 0.0, 0.0)
print(f"  {phase.name}: potential_kind = {phase.potential_kind}, "
      f"seed_cacheable = {phase.seed_cacheable}")
print(f"  n_B = {block.n_B:.4f} fm^-3   P = {block.P:.2f} MeV/fm^3   "
      f"eps = {block.eps:.2f} MeV/fm^3")
print(f"  Euler residual {block.euler_residual():.1e}")
print(f"  fields: {dict(block.fields)}")
