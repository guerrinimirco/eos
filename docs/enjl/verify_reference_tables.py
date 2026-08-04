"""
verify_reference_tables.py
==========================
Standalone audit of the Xia reference tables in `test/enjl/reference/`.

This script depends on NOTHING in `eos/` except the physical constants: it
rebuilds the extended-NJL mean field from the published parameters and the
tables' own columns, and reports how well each identity of
`docs/enjl/REFERENCE_TABLES.md` holds. Its purpose is to be the independent
oracle — if `eos/enjl/` and this script disagree, one of them is wrong and the
disagreement is diagnostic.

Run from the repository root:

    python docs/enjl/verify_reference_tables.py

Every number it prints was used to write the tolerance columns in
REFERENCE_TABLES.md. Re-run it if a table is ever replaced or extended.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "test", "enjl"))

from reference import (   # noqa: E402
    load_reference, solved_rows, baryon_potential, electron_potential,
    present, PARAMETER_SETS,
)

# --- constants and published parameters (paper Sec. II.B and Table I) --------
HC = 197.3269804
HC3 = HC ** 3
PI2 = np.pi ** 2

LAMBDA = 602.3
M0 = {"u": 5.5, "d": 5.5, "s": 140.7}
GS = 1.835 / LAMBDA ** 2
KAPPA = 12.36 / LAMBDA ** 5

A_S, B_S, N_S = 0.4413715, 0.4076285, 0.16
A_V, B_V, N_V = 3.566049, 1.062771, 0.214
A_TV, B_TV, N_TV = 0.5014459, 0.0117601, 0.1

F_LAMBDA = 1.0626
#: isospin-source normalization of the rho channel; see REFERENCE_TABLES.md 4c
RHO_FACTOR = 9.0

BARYONS = ("p", "n", "L")
QUARKS = ("u", "d", "s")
#: valence content N^q_i
NQ = {"u": {"p": 2, "n": 1, "L": 1},
      "d": {"p": 1, "n": 2, "L": 1},
      "s": {"p": 0, "n": 0, "L": 1}}
TAU = {"p": 1.0, "n": -1.0, "L": 0.0, "u": 1.0, "d": -1.0, "s": 0.0}
DEG = {"p": 2.0, "n": 2.0, "L": 2.0, "u": 6.0, "d": 6.0, "s": 6.0,
       "e": 2.0, "mu": 2.0}
CHARGE = {"p": 1.0, "n": 0.0, "L": 0.0,
          "u": 2.0 / 3.0, "d": -1.0 / 3.0, "s": -1.0 / 3.0,
          "e": -1.0, "mu": -1.0}
BNUM = {"p": 1.0, "n": 1.0, "L": 1.0,
        "u": 1.0 / 3.0, "d": 1.0 / 3.0, "s": 1.0 / 3.0, "e": 0.0, "mu": 0.0}
M_LEPTON = {"e": 0.511, "mu": 105.66}


# --- T = 0 kinetic integrals, Eqs. (11)-(13), natural units -----------------
def fermi_momentum(n_fm, g):
    """nu_i [MeV] from n [fm^-3]."""
    return (6.0 * PI2 * np.maximum(n_fm, 0.0) * HC3 / g) ** (1.0 / 3.0)


def _asinh(z):
    return np.arcsinh(np.maximum(z, 0.0))


def scalar_density(nu, M, g, cutoff=0.0):
    """n^s_i [MeV^3], Eq. (12). `cutoff` = Lambda for quarks, 0 otherwise."""
    x, y = nu / M, cutoff / M
    fx = x * np.sqrt(x * x + 1.0) - _asinh(x)
    fy = y * np.sqrt(y * y + 1.0) - _asinh(y)
    return (g * M ** 3 / (4.0 * PI2)) * (fx - fy)


def energy_density(nu, M, g, cutoff=0.0):
    """kinetic + vacuum energy density [MeV^4], Eq. (13)."""
    x, y = nu / M, cutoff / M
    fx = x * (2.0 * x * x + 1.0) * np.sqrt(x * x + 1.0) - _asinh(x)
    fy = y * (2.0 * y * y + 1.0) * np.sqrt(y * y + 1.0) - _asinh(y)
    return (g * M ** 4 / (16.0 * PI2)) * (fx - fy)


# --- density-dependent couplings, Eqs. (20)-(22) ----------------------------
def alpha_S(n_b):
    return A_S * np.exp(-n_b / N_S) + B_S


def d_alpha_S(n_b):
    """d alpha_S / d n_b, per MeV^3."""
    return -A_S * np.exp(-n_b / N_S) / (N_S * HC3)


def Gamma_omega(n_b):
    return 4.0 * GS * (A_V * np.exp(-n_b / N_V) + B_V)


def d_Gamma_omega(n_b):
    return -4.0 * GS * A_V * np.exp(-n_b / N_V) / (N_V * HC3)


def Gamma_rho(n_b):
    return RHO_FACTOR * 4.0 * GS * (A_TV * np.exp(-n_b / N_TV) + B_TV)


def d_Gamma_rho(n_b):
    return -RHO_FACTOR * 4.0 * GS * A_TV * np.exp(-n_b / N_TV) / (N_TV * HC3)


# --- mean field rebuilt from a table's own columns --------------------------
def mean_field(col, f_q, B_GeV):
    """Fields, rearrangement terms and mu_i from the table's densities/masses.

    Returns a dict with `mu` (per species), `SigmaR_b`, `SigmaR_q`,
    `gw` = g_omega omega, `gr` = g_rho rho, and the sources J_omega, J_rho.
    All natural units except mu (MeV).
    """
    n_b = col["nB"]
    f = {"p": 1.0, "n": 1.0, "L": F_LAMBDA,
         "u": f_q, "d": f_q, "s": f_q, "e": 0.0, "mu": 0.0}
    dens = {sp: col["n" + sp] * HC3 for sp in DEG}       # MeV^3
    ns_b = {b: col["ns" + b] * HC3 for b in BARYONS}     # MeV^3

    J_w = sum(f[b] * 3.0 * dens[b] for b in BARYONS) \
        + sum(f[q] * dens[q] for q in QUARKS)
    J_r = sum(f[sp] * TAU[sp] * dens[sp] for sp in BARYONS + QUARKS)
    gw = Gamma_omega(n_b) * J_w
    gr = Gamma_rho(n_b) * J_r

    shift = {b: sum(NQ[q][b] * (col["M" + q] - M0[q]) for q in QUARKS)
             for b in BARYONS}
    SigmaR_b = (0.5 * d_Gamma_omega(n_b) * J_w ** 2
                + 0.5 * d_Gamma_rho(n_b) * J_r ** 2
                + sum(d_alpha_S(n_b) * shift[b] * ns_b[b] for b in BARYONS))
    B_nat = B_GeV * 1000.0 / HC3
    SigmaR_q = (B_nat * sum(ns_b.values()) + SigmaR_b) / 3.0

    mu = {}
    for b in BARYONS:
        nu = fermi_momentum(col["n" + b], DEG[b])
        mu[b] = (np.sqrt(nu ** 2 + col["M" + b] ** 2)
                 + f[b] * (3.0 * gw + TAU[b] * gr) + SigmaR_b)
    for q in QUARKS:
        nu = fermi_momentum(col["n" + q], DEG[q])
        mu[q] = (np.sqrt(nu ** 2 + col["M" + q] ** 2)
                 + f[q] * (gw + TAU[q] * gr) + SigmaR_q)
    for l, m in M_LEPTON.items():
        nu = fermi_momentum(col["n" + l], DEG[l])
        mu[l] = np.sqrt(nu ** 2 + m ** 2)
    return dict(mu=mu, SigmaR_b=SigmaR_b, SigmaR_q=SigmaR_q,
                gw=gw, gr=gr, J_w=J_w, J_r=J_r)


# --- individual audits ------------------------------------------------------
#: rows that are non-converged Maple output and cannot be repaired. Keyed by
#: file, values are n_b [fm^-3] to be matched to 1e-6. See REFERENCE_TABLES.md
#: section 4b-bis for the evidence on each.
BAD_ROWS = {
    #   3.4 : self-consistent columns (`nB` right, epa*nB == E to 8e-8) but
    #         fails Eq. (23) by 9.6 MeV in the p channel, while the two rows
    #         below satisfy it to 1e-8 and the two above are stale-nB rows
    "Beta_fq0.7_B0.dat": (3.4,),
}


def bad_rows(col, filename):
    """Explicitly excluded non-converged rows, plus the stale-`nB` ones."""
    mask = stale_density_rows(col)
    for n_b in BAD_ROWS.get(filename, ()):
        mask = mask | (np.abs(col["nB"] - n_b) < 1.0e-6)
    return mask


def stale_density_rows(col):
    """Rows whose `nB` column disagrees with the sum of its own densities.

    Eight rows of Beta_fq0.7_B0.dat are like this — `nB` high by one grid step
    while `epa` = E / (sum of densities). Their other columns come from a
    mixture of iterations too, so they cannot be repaired, only excluded.
    See REFERENCE_TABLES.md section 4b-bis.
    """
    n_sum = (col["np"] + col["nn"] + col["nL"]
             + (col["nu"] + col["nd"] + col["ns"]) / 3.0)
    return np.abs(n_sum - col["nB"]) > 1.0e-5


def audit_bookkeeping(col, ok):
    n_b = col["nB"]
    quark_nb = (col["nu"] + col["nd"] + col["ns"]) / 3.0
    charge = sum(CHARGE[sp] * col["n" + sp] for sp in DEG)
    quark_total = col["nu"] + col["nd"] + col["ns"]
    quark_rich = quark_total > 0.1 * np.maximum(n_b, 1.0e-30)
    baryons = col["np"] + col["nn"] + col["nL"]
    has_baryons = baryons > 1.0e-8
    mu_q = col["muu"] + 2.0 * col["mud"]
    return {
        "n_b sum rule (Eq. 7)":
            np.nanmax(np.abs((col["np"] + col["nn"] + col["nL"] + quark_nb
                              - n_b)[ok])),
        "charge neutrality (Eq. 24), absolute":
            np.nanmax(np.abs(charge[ok])),
        "  relative to n_u+n_d+n_s  [quarks present]":
            (np.nanmax((np.abs(charge)
                        / np.maximum(quark_total, 1.0e-30))[ok & quark_rich])
             if (ok & quark_rich).any() else float("nan")),
        "epa * n_b == E":
            np.nanmax(np.abs((col["epa"] * n_b - col["E"])[ok])),
        "fq == n_b^Q / n_b":
            np.nanmax(np.abs((col["fq"] - quark_nb / n_b)[ok])),
        "munr == mun  [baryons present]":
            np.nanmax(np.abs((col["munr"] - col["mun"])[ok & has_baryons])),
        "munr == muu + 2 mud  [baryons gone]":
            (np.nanmax(np.abs((col["munr"] - mu_q)[ok & ~has_baryons]))
             if (ok & ~has_baryons).any() else float("nan")),
    }


def audit_gap_equation(col, ok):
    """Eq. (5) fed with the Sigmaq columns, skipping chirally restored rows."""
    nbar = {q: col["Sigma" + q] * HC3 for q in QUARKS}
    prod = nbar["u"] * nbar["d"] * nbar["s"]
    out = {}
    for q in QUARKS:
        restored = np.abs(col["M" + q] - M0[q]) < 1.0e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = M0[q] - 4.0 * GS * nbar[q] + 2.0 * KAPPA * prod / nbar[q]
        rel = np.abs(gap - col["M" + q]) / col["M" + q]
        rel = np.where(restored | ~ok, np.nan, rel)
        out[f"M_{q} from Eq. (5), relative"] = np.nanmax(rel)
        out[f"  rows skipped (M_{q} == m_{q}0)"] = int((restored & ok).sum())
    return out


def audit_scalar_densities(col, ok):
    """nsq = medium only;  Sigmaq = medium + vacuum + alpha_S sum. See section 3."""
    a = alpha_S(col["nB"])
    out = {}
    for q in QUARKS:
        restored = np.abs(col["M" + q] - M0[q]) < 1.0e-9
        nu = fermi_momentum(col["n" + q], 6.0)
        medium = scalar_density(nu, col["M" + q], 6.0, 0.0) / HC3
        vacuum = -scalar_density(np.zeros_like(nu), col["M" + q], 6.0,
                                 LAMBDA) / HC3
        cluster = a * sum(NQ[q][b] * col["ns" + b] for b in BARYONS)
        e_ns = np.abs(col["ns" + q] - (medium + cluster))
        e_sig = np.abs(col["Sigma" + q] - (medium - vacuum + cluster))
        use = ok & ~restored
        out[f"ns{q} == medium + alpha_S sum"] = np.nanmax(e_ns[use])
        out[f"Sigma{q} == medium+vacuum + alpha_S sum"] = np.nanmax(e_sig[use])
    for b in BARYONS:
        use = ok & (col["n" + b] > 1.0e-8)
        nu = fermi_momentum(col["n" + b], 2.0)
        calc = scalar_density(nu, col["M" + b], 2.0, 0.0) / HC3
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(calc - col["ns" + b]) / np.abs(col["ns" + b])
        out[f"ns{b} == Eq. (12), Lambda=0 (relative)"] = np.nanmax(rel[use])
    return out


def audit_beta_equilibrium(col, ok):
    """Eq. (23), enforced only on species that are actually present.

    mu_b is `munr`, not `mun`, and mu_e comes from `electron_potential` —
    see REFERENCE_TABLES.md sections 2 and 4.
    """
    mu_b = baryon_potential(col)
    mu_e = electron_potential(col)
    out = {}
    for sp in BARYONS + QUARKS:
        use = ok & present(col, sp)
        resid = np.abs(col["mu" + sp]
                       - (BNUM[sp] * mu_b - CHARGE[sp] * mu_e))
        out[f"mu_{sp} = B munr - q mu_e  [n_{sp} > 1e-4 n_b]"] = \
            np.nanmax(resid[use]) if use.any() else float("nan")
    return out


def audit_pressure(col, ok):
    mu_dot_n = sum(col["mu" + sp] * col["n" + sp]
                   for sp in ("u", "d", "s", "p", "n", "L", "e"))
    mu_dot_n = mu_dot_n + col["mumu"] * col["nmu"]
    resid = (mu_dot_n - col["E"] - col["P"])[ok]
    return {"P = sum mu_i n_i - E (Eq. 19), absolute": np.nanmax(
                np.abs(resid)),
            "  relative to P": np.nanmax(
                np.abs(resid) / np.maximum(col["P"][ok], 1.0e-3))}


def audit_mean_field(col, f_q, B_GeV, ok):
    mf = mean_field(col, f_q, B_GeV)
    out = {}
    for sp in BARYONS + QUARKS:
        use = ok & (col["n" + sp] > 1.0e-8)
        if not use.any():
            out[f"mu_{sp} rebuilt (Eqs. 14-15)"] = float("nan")
            continue
        err = np.abs(mf["mu"][sp] - col["mu" + sp])[use]
        out[f"mu_{sp} rebuilt (Eqs. 14-15)"] = np.nanmax(err)
    low = ok & (col["nB"] < 1.0)
    errs = [np.abs(mf["mu"][sp] - col["mu" + sp])[low & (col["n" + sp] > 1e-8)]
            for sp in BARYONS + QUARKS]
    errs = [e for e in errs if e.size]
    out["  worst of the above below n_b = 1 fm^-3"] = \
        max(np.nanmax(e) for e in errs)
    return out


def audit_E0(col, f_q, B_GeV, ok):
    """E_0 of Eq. (13) extracted as the density-independent offset."""
    mf = mean_field(col, f_q, B_GeV)
    kin = np.zeros_like(col["nB"])
    for b in BARYONS:
        nu = fermi_momentum(col["n" + b], 2.0)
        kin = kin + energy_density(nu, col["M" + b], 2.0, 0.0)
    for q in QUARKS:
        nu = fermi_momentum(col["n" + q], 6.0)
        kin = kin + energy_density(nu, col["M" + q], 6.0, LAMBDA)
    for l, m in M_LEPTON.items():
        nu = fermi_momentum(col["n" + l], DEG[l])
        kin = kin + energy_density(nu, m, DEG[l], 0.0)
    nbar = {q: col["Sigma" + q] * HC3 for q in QUARKS}
    cond = 2.0 * GS * sum(nbar[q] ** 2 for q in QUARKS) \
        - 4.0 * KAPPA * nbar["u"] * nbar["d"] * nbar["s"]
    vec = 0.5 * mf["gw"] * mf["J_w"] + 0.5 * mf["gr"] * mf["J_r"]
    E0 = ((kin + cond + vec - col["E"] * HC3) / HC3)[ok]
    return {"E_0 [MeV/fm^3] (mean)": round(float(np.nanmean(E0)), 4),
            "  spread, relative": float(
                (np.nanmax(E0) - np.nanmin(E0)) / abs(np.nanmean(E0)))}


def coexistence_rows(col):
    """The Maxwell coexistence windows. See REFERENCE_TABLES.md section 5.

    Off-grid densities are the transition rows. Each first-order transition
    shows up as a run of them at one constant pressure, so grouping by pressure
    collapses the f_q = 0.5 interpolated plateaus to their two endpoints and
    leaves the other files' endpoint pairs untouched.

    Returns [(P, n_b_low, n_b_high, mu_b_low, mu_b_high, n_rows), ...]. mu_b is
    `munr` where it is filled in and `mun` on the mixed-phase rows where it is
    not (there the two agree, since baryons are still present).
    """
    n_b = col["nB"]
    off = np.abs(n_b * 100.0 - np.round(n_b * 100.0)) > 1.0e-6
    mu_b = np.where(np.isfinite(col["munr"]), col["munr"], col["mun"])
    out = []
    for P in np.unique(np.round(col["P"][off], 3)):
        grp = off & (np.abs(np.round(col["P"], 3) - P) < 1.0e-9)
        lo, hi = int(np.argmax(grp)), int(len(grp) - 1 - np.argmax(grp[::-1]))
        out.append((float(col["P"][lo]), float(n_b[lo]), float(n_b[hi]),
                    float(mu_b[lo]), float(mu_b[hi]), int(grp.sum())))
    return sorted(out, key=lambda r: r[1])


# --- driver -----------------------------------------------------------------
def audit_file(filename, f_q, B_GeV):
    col = load_reference(filename)
    ok = solved_rows(col)
    interpolated = int((~ok).sum())
    bad = bad_rows(col, filename)
    ok = ok & ~bad
    note = ""
    if interpolated:
        note += f", {interpolated} interpolated rows excluded"
    if bad.any():
        note += f", {int(bad.sum())} non-converged rows excluded"
    print(f"\n{'=' * 74}\n{filename}   f_q = {f_q}, B = {B_GeV} GeV/fm^3, "
          f"{int(ok.sum())} solved rows{note}\n{'=' * 74}")
    for title, block in [
        ("bookkeeping", audit_bookkeeping(col, ok)),
        ("gap equation, Eq. (5)", audit_gap_equation(col, ok)),
        ("scalar densities, Eqs. (6), (12)", audit_scalar_densities(col, ok)),
        ("beta equilibrium, Eq. (23)", audit_beta_equilibrium(col, ok)),
        ("pressure, Eq. (19)", audit_pressure(col, ok)),
        ("mean field, Eqs. (14)-(18)",
         audit_mean_field(col, f_q, B_GeV, ok)),
        ("vacuum constant, Eq. (13)", audit_E0(col, f_q, B_GeV, ok)),
    ]:
        print(f"\n  -- {title}")
        for key, value in block.items():
            if isinstance(value, int):
                print(f"     {key:<52s} {value:d}")
            elif abs(value) > 1.0:
                print(f"     {key:<52s} {value:.4f}")
            else:
                print(f"     {key:<52s} {value:.3e}")
    windows = coexistence_rows(col)
    if windows:
        print("\n  -- Maxwell coexistence windows (off-grid densities)")
        print(f"     {'P':>12s} {'n_b low':>10s} {'n_b high':>10s} "
              f"{'mu_b low':>11s} {'mu_b high':>11s} {'rows':>6s}")
        for P, lo, hi, mu_lo, mu_hi, count in windows:
            print(f"     {P:12.4f} {lo:10.6f} {hi:10.6f} "
                  f"{mu_lo:11.4f} {mu_hi:11.4f} {count:6d}")


def main():
    for filename, (f_q, B_GeV) in PARAMETER_SETS.items():
        audit_file(filename, f_q, B_GeV)
    print("\nSee docs/enjl/REFERENCE_TABLES.md for what each number should be.")


if __name__ == "__main__":
    main()
