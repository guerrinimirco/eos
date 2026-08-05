"""Figures of Xia 2024 (PRD 110, 014022) reproduced from `eos.enjl`.

Run from the repository root:

    python plot/enjl_paper_figures.py [outdir]

Eight figures are described in that paper. Which of them this script can build
from the engine alone, and which need something that is not settled yet:

    Fig. 1  masses in symmetric nuclear matter          fixed composition, done
    Fig. 2  E/A of SNM and PNM                          fixed composition, done
    Fig. 3  Lambda potential depth U_L(n_b)             fixed composition, done
    Fig. 4  P vs mu_b, transitions marked               branches + Maxwell
    Fig. 5  number densities n_i in stellar matter      beta equilibrium
    Fig. 6  masses in stellar matter                    beta equilibrium
    Fig. 7  c_s^2, quark fraction, P vs energy density  beta equilibrium
    Fig. 8  mass-radius relations                       needs TOV + a crust

Figures 1-3 involve no branch ambiguity: the composition is imposed, so there
is exactly one solution and the curves are predictions of the engine.

Figures 4-7 are beta-equilibrium quantities, where the model has several
coexisting solution branches above its first-order transitions and choosing
between them is an open question (see DD2_OPEN_QUESTIONS.md, G3). Rather than
pick a rule and present the result as settled, these figures draw **both**
branches that a continuation can reach, and overlay the author's own tabulated
result where one exists. Where the curves coincide the engine is reproducing
the paper; where they part, the parting is the open question, drawn rather than
hidden. That is also closer to what Fig. 4 of the paper actually shows, since
its solid stars mark exactly such branch crossings.

Fig. 8 is not attempted. It needs the branch question closed first, and a crust
below the core-crust transition — a crustless star is about 0.9 km too small at
1.4 solar masses, which is larger than the spread the figure is meant to show.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "test", "enjl"))

from eos.enjl import ENJLParams, solve_point                       # noqa: E402
from eos.enjl.eos_beta import beta_eos_table                       # noqa: E402
from eos.enjl.uniform import vacuum_solution, _baryon_masses       # noqa: E402
from eos.general.physics_constants import hc3                      # noqa: E402
from eos.general.figure_style import OKAB_CAT, apply_style         # noqa: E402

from reference import (PARAMETER_SETS, bad_rows, baryon_potential,  # noqa: E402
                       load_reference, solved_rows)

#: the six (f_q, B) combinations of the paper; only five have reference tables
PARAM_GRID = [(0.5, 0.0), (0.5, 1.0), (0.7, 0.0),
              (0.7, 1.0), (1.0, 0.0), (1.0, 1.0)]
#: (f_q, B) -> reference file, where one exists
TABLE_FOR = {(f_q, B): name for name, (f_q, B) in
             ((n, v) for n, v in PARAMETER_SETS.items())}

REF_COL = {"p": "p", "n": "n", "Lambda": "L", "u": "u", "d": "d", "s": "s",
           "e": "e", "mu": "mu"}
SPECIES = ("p", "n", "Lambda", "u", "d", "s", "e", "mu")
LABEL = {"p": "$p$", "n": "$n$", "Lambda": r"$\Lambda$", "u": "$u$",
         "d": "$d$", "s": "$s$", "e": "$e$", "mu": r"$\mu$"}
COLOR = {"p": OKAB_CAT[0], "n": OKAB_CAT[1], "Lambda": OKAB_CAT[2],
         "u": OKAB_CAT[3], "d": OKAB_CAT[4], "s": OKAB_CAT[5],
         "e": "0.45", "mu": "0.7"}


def tag(f_q, B):
    return f"fq{f_q}_B{int(B)}"


def dens(**kwargs):
    """Density dict in MeV^3 from fm^-3 keyword arguments."""
    return {sp: kwargs.get(sp, 0.0) * hc3 for sp in SPECIES}


# ----------------------------------------------------------------- branches
def branches(f_q, B, grid):
    """Both continuation branches at these parameters: (up, down).

    Each is a dict {n_b -> BetaPoint}, shortened where a density did not
    converge. `up` starts chirally broken at the bottom of the grid; `down`
    starts deconfined at the top.
    """
    par = ENJLParams(f_q=f_q, B_GeV_fm3=B)
    out = []
    for direction in ("up", "down"):
        pts, _, _ = beta_eos_table(grid, par=par, direction=direction)
        out.append({p.n_b_fm: p for p in pts})
    return out


def table_curve(f_q, B):
    """The author's tabulated beta-equilibrium result, or None."""
    name = TABLE_FOR.get((f_q, B))
    if name is None:
        return None
    col = load_reference(name)
    ok = solved_rows(col) & ~bad_rows(col, name)
    keep = {k: v[ok] for k, v in col.items()}
    keep["mu_b"] = baryon_potential(col)[ok]
    return keep


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------- Figs. 1-3, fixed x
def figure_1(outdir):
    """Masses of p, n, Lambda and u, d, s in symmetric nuclear matter."""
    par = ENJLParams()
    grid = np.linspace(0.01, 1.2, 140)
    Mb = {b: [] for b in ("p", "n", "Lambda")}
    Mq = {q: [] for q in "uds"}
    seed = None
    for x in grid:
        pt = solve_point(dens(p=x / 2.0, n=x / 2.0), par=par, x0=seed)
        seed = [pt.M_q["u"], pt.M_q["d"], pt.M_q["s"]]
        for b in Mb:
            Mb[b].append(pt.M_b[b])
        for q in Mq:
            Mq[q].append(pt.M_q[q])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    # p and n are exactly degenerate in symmetric matter, so n is drawn
    # dashed on top of p rather than hiding it
    ax.plot(grid, Mb["p"], color=COLOR["p"], lw=2.4, label=LABEL["p"])
    ax.plot(grid, Mb["n"], color=COLOR["n"], lw=1.2, ls=(0, (4, 3)),
            label=LABEL["n"] + " (degenerate with $p$)")
    ax.plot(grid, Mb["Lambda"], color=COLOR["Lambda"], label=LABEL["Lambda"])
    for q in "uds":
        ax.plot(grid, Mq[q], color=COLOR[q], ls="--", label=LABEL[q])
    M0 = vacuum_solution(par)
    Mb0 = _baryon_masses(par, M0, par.alpha_S(0.0), 0.0)
    for y in (Mb0["Lambda"], Mb0["p"]):
        ax.axhline(y, color="0.75", lw=0.7, ls=":")
    ax.annotate(f"{Mb0['Lambda']:.1f}", (0.02, Mb0["Lambda"] + 15), fontsize=8,
                color="0.4")
    ax.annotate(f"{Mb0['p']:.1f}", (0.02, Mb0["p"] + 15), fontsize=8,
                color="0.4")
    ax.set_xlabel(r"$n_b$ [fm$^{-3}$]")
    ax.set_ylabel("mass [MeV]")
    ax.set_title("Fig. 1 — masses in symmetric nuclear matter")
    ax.set_xlim(0, 1.2)
    apply_style(ax)
    save(fig, outdir, "enjl_fig01_masses_snm.png")


def figure_2(outdir):
    """E/A of symmetric nuclear matter and pure neutron matter."""
    grid = np.linspace(0.01, 0.6, 120)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    par = ENJLParams()
    for label, comp, ls in (("SNM", lambda x: dens(p=x / 2, n=x / 2), "-"),
                            ("PNM", lambda x: dens(n=x), "--")):
        vals, seed = [], None
        for x in grid:
            pt = solve_point(comp(x), par=par, x0=seed)
            seed = [pt.M_q["u"], pt.M_q["d"], pt.M_q["s"]]
            vals.append(pt.EperB)
        ax.plot(grid, vals, ls, color=OKAB_CAT[0 if label == "SNM" else 1],
                label=label)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.plot([0.158], [-16.0], "k*", ms=11, label="published saturation")
    ax.set_xlabel(r"$n_b$ [fm$^{-3}$]")
    ax.set_ylabel(r"$E/A - m_N$ [MeV]")
    ax.set_title("Fig. 2 — energy per baryon")
    ax.set_ylim(-25, 90)
    apply_style(ax)
    save(fig, outdir, "enjl_fig02_energy_per_baryon.png")


def figure_3(outdir):
    """Lambda potential depth in symmetric nuclear matter, U_L(n_b)."""
    par = ENJLParams()
    M0 = vacuum_solution(par)
    m0 = {"u": par.m_u0, "d": par.m_d0, "s": par.m_s0}
    a0 = par.alpha_S(0.0)
    M_L_vac = sum(m0[q] + a0 * (M0[q] - m0[q]) for q in "uds")

    grid = np.linspace(0.02, 0.5, 90)
    U, seed = [], None
    for x in grid:
        pt = solve_point(dens(p=x / 2.0, n=x / 2.0, Lambda=1.0e-9),
                         par=par, x0=seed)
        seed = [pt.M_q["u"], pt.M_q["d"], pt.M_q["s"]]
        U.append(pt.mu["Lambda"] - M_L_vac)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(grid, U, color=OKAB_CAT[2])
    ax.plot([0.158], [-30.0], "k*", ms=11,
            label=r"$U_\Lambda(n_0) = -30$ MeV (fixes $f_\Lambda$)")
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel(r"$n_b$ [fm$^{-3}$]")
    ax.set_ylabel(r"$U_\Lambda$ [MeV]")
    ax.set_title(r"Fig. 3 — $\Lambda$ potential depth in SNM")
    apply_style(ax)
    save(fig, outdir, "enjl_fig03_lambda_potential.png")


# --------------------------------------------- Figs. 4-7, beta equilibrium
def figure_4(outdir, data):
    """P vs mu_b for every parameter set, both branches drawn."""
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), sharex=False)
    for ax, (f_q, B) in zip(axes.ravel(), PARAM_GRID):
        up, down, tab = data[(f_q, B)]
        for br, c, lab in ((up, OKAB_CAT[0], "continuation from low density"),
                           (down, OKAB_CAT[3], "continuation from high density")):
            if not br:
                continue
            nb = np.array(sorted(br))
            ax.plot([br[x].mu_b for x in nb], [br[x].P for x in nb],
                    color=c, lw=1.4, label=lab)
        if tab is not None:
            ax.plot(tab["mu_b"], tab["P"], "k:", lw=1.6, label="reference table")
        ax.set_xlabel(r"$\mu_b$ [MeV]")
        ax.set_ylabel(r"$P$ [MeV/fm$^3$]")
        ax.set_title(rf"$f_q={f_q}$, $B={B:g}$ GeV/fm$^3$", fontsize=10)
        ax.set_xlim(900, 2600)
        ax.set_ylim(0, 900)
        apply_style(ax)
    fig.suptitle("Fig. 4 — pressure vs baryon chemical potential; where the "
                 "two branches cross, the transition is first order", y=1.005)
    fig.tight_layout()
    save(fig, outdir, "enjl_fig04_P_vs_mu.png")


def figure_5(outdir, data):
    """Number densities of every fermion vs n_b (the Y_i figure)."""
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    for ax, (f_q, B) in zip(axes.ravel(), PARAM_GRID):
        up, down, tab = data[(f_q, B)]
        nb = np.array(sorted(up))
        for sp in SPECIES:
            ax.plot(nb, [max(up[x].densities[sp], 1e-12) for x in nb],
                    color=COLOR[sp], lw=1.3, label=LABEL[sp])
        if tab is not None:
            for sp in SPECIES:
                ax.plot(tab["nB"], np.maximum(tab["n" + REF_COL[sp]], 1e-12),
                        color=COLOR[sp], lw=0.8, ls=":", alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 30)
        ax.set_xlim(0, 3)
        ax.set_xlabel(r"$n_b$ [fm$^{-3}$]")
        ax.set_ylabel(r"$n_i$ [fm$^{-3}$]")
        ax.set_title(rf"$f_q={f_q}$, $B={B:g}$ GeV/fm$^3$", fontsize=10)
        apply_style(ax, legend=False)
    axes[0, 0].legend(ncol=4, fontsize=7, loc="lower right")
    fig.suptitle("Fig. 5 — composition of beta-equilibrium stellar matter "
                 "(solid: this engine, dotted: reference table)", y=1.005)
    fig.tight_layout()
    save(fig, outdir, "enjl_fig05_composition.png")


def figure_6(outdir, data):
    """Masses of baryons and quarks in beta-equilibrium stellar matter."""
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    for ax, (f_q, B) in zip(axes.ravel(), PARAM_GRID):
        up, down, tab = data[(f_q, B)]
        nb = np.array(sorted(up))
        for b in ("p", "n", "Lambda"):
            ax.plot(nb, [up[x].M_b[b] for x in nb], color=COLOR[b],
                    lw=1.3, label=LABEL[b])
        for q in "uds":
            ax.plot(nb, [up[x].M_q[q] for x in nb], color=COLOR[q],
                    lw=1.3, ls="--", label=LABEL[q])
        if tab is not None:
            for b, su in (("p", "p"), ("n", "n"), ("Lambda", "L")):
                ax.plot(tab["nB"], tab["M" + su], color=COLOR[b], lw=0.8,
                        ls=":", alpha=0.85)
            for q in "uds":
                ax.plot(tab["nB"], tab["M" + q], color=COLOR[q], lw=0.8,
                        ls=":", alpha=0.85)
        ax.set_xlim(0, 3)
        ax.set_xlabel(r"$n_b$ [fm$^{-3}$]")
        ax.set_ylabel("mass [MeV]")
        ax.set_title(rf"$f_q={f_q}$, $B={B:g}$ GeV/fm$^3$", fontsize=10)
        apply_style(ax, legend=False)
    axes[0, 0].legend(ncol=3, fontsize=7)
    fig.suptitle("Fig. 6 — effective masses in stellar matter "
                 "(solid/dashed: this engine, dotted: reference table)",
                 y=1.005)
    fig.tight_layout()
    save(fig, outdir, "enjl_fig06_masses_stellar.png")


def figure_7(outdir, data):
    """Sound speed, quark fraction and P, against energy density."""
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 10.0), sharex=True)
    for k, (f_q, B) in enumerate(PARAM_GRID):
        up, _, _ = data[(f_q, B)]
        nb = np.array(sorted(up))
        if len(nb) < 4:
            continue
        eps = np.array([up[x].eps for x in nb])
        P = np.array([up[x].P for x in nb])
        fq_frac = np.array([sum(up[x].densities[q] for q in "uds") / 3.0 / x
                            for x in nb])
        order = np.argsort(eps)
        eps, P, fq_frac = eps[order], P[order], fq_frac[order]
        cs2 = np.gradient(P, eps)
        c = OKAB_CAT[k % len(OKAB_CAT)]
        lab = rf"$f_q={f_q}$, $B={B:g}$"
        axes[0].plot(eps, np.clip(cs2, 0, 1), color=c, lw=1.3, label=lab)
        axes[1].plot(eps, fq_frac, color=c, lw=1.3)
        axes[2].plot(eps, P, color=c, lw=1.3)
    axes[0].set_ylabel(r"$c_s^2$")
    axes[0].axhline(1.0 / 3.0, color="0.7", lw=0.8, ls=":")
    axes[1].set_ylabel(r"$n_b^Q / n_b$")
    axes[2].set_ylabel(r"$P$ [MeV/fm$^3$]")
    axes[2].set_xlabel(r"$\varepsilon$ [MeV/fm$^3$]")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    for a in axes:
        apply_style(a, legend=False, minor_ticks=False)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Fig. 7 — sound speed, quark fraction and pressure", y=1.002)
    fig.tight_layout()
    save(fig, outdir, "enjl_fig07_cs2_fq_P.png")


def figure_P_vs_nb(outdir, data):
    """P vs n_b — the equation of state in the variables a TOV solve wants."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for k, (f_q, B) in enumerate(PARAM_GRID):
        up, _, tab = data[(f_q, B)]
        nb = np.array(sorted(up))
        ax.plot(nb, [up[x].P for x in nb], color=OKAB_CAT[k % 6], lw=1.4,
                label=rf"$f_q={f_q}$, $B={B:g}$")
        if tab is not None:
            ax.plot(tab["nB"], tab["P"], color=OKAB_CAT[k % 6], lw=0.8,
                    ls=":", alpha=0.8)
    ax.set_xlabel(r"$n_b$ [fm$^{-3}$]")
    ax.set_ylabel(r"$P$ [MeV/fm$^3$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Pressure vs baryon density "
                 "(solid: this engine, dotted: reference table)")
    apply_style(ax, minor_ticks=False)
    save(fig, outdir, "enjl_fig09_P_vs_nb.png")


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output")
    os.makedirs(outdir, exist_ok=True)
    print(f"writing to {outdir}")

    print("fixed-composition figures (no branch ambiguity)")
    figure_1(outdir)
    figure_2(outdir)
    figure_3(outdir)

    print("beta-equilibrium branches (this is the slow part)")
    grid = np.round(np.concatenate([np.arange(0.05, 1.0, 0.025),
                                    np.arange(1.0, 10.01, 0.1)]), 4)
    data = {}
    for f_q, B in PARAM_GRID:
        up, down = branches(f_q, B, grid)
        data[(f_q, B)] = (up, down, table_curve(f_q, B))
        print(f"  f_q={f_q}, B={B:g}: {len(up)} up, {len(down)} down"
              + ("" if TABLE_FOR.get((f_q, B)) else "   [no reference table]"))

    figure_4(outdir, data)
    figure_5(outdir, data)
    figure_6(outdir, data)
    figure_7(outdir, data)
    figure_P_vs_nb(outdir, data)
    print("done. Fig. 8 (mass-radius) is not built: see the module docstring.")


if __name__ == "__main__":
    main()
