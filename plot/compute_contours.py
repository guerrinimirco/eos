#!/usr/bin/env python
"""Precompute 68% / 95% confidence contours for NS-EOS constraint plots.

Reads raw posterior samples from  eos/plot/data/samples/  and writes ready-to-plot
CSV contours to  eos/plot/data/contours/  so plotting scripts never rebuild KDEs
on the fly.

    python compute_contours.py

Everything is deliberately kept as small, single-purpose functions so individual
constraints can be regenerated or added without touching the rest.

Physics note: "68%" / "95%" here are the *enclosed-probability* contours of the
2D posterior (1σ / 2σ credible regions), not per-axis error bars.  For the
mass-only sources they are the plain 1σ / 2σ Gaussian intervals.
"""
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")            # headless: we only harvest contour paths, never show
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "data" / "samples"
# The generated CSVs live INSIDE the package (shipped as package-data) so that
# eos.general.observational_constraints can read them from an installed wheel.
# The raw posterior samples stay out here: they are ~160 MB and only this
# script needs them.
CONTOURS = HERE.parent / "eos" / "general" / "data" / "contours"

FRACTIONS = (0.68, 0.95)         # 1σ / 2σ enclosed probability
KDE_NMAX = 8000                  # subsample cap: gaussian_kde is O(n_samples * n_grid)
GRID = 180                       # KDE evaluation grid per axis


# ─────────────────────────────────────────────────────────── core KDE machinery
def _subsample(cols, n_max=KDE_NMAX, weights=None, rng=None):
    """Return columns thinned to <= n_max rows.  If weights are given, draw with
    probability ∝ weight so the weight information is baked into the sample and a
    plain (unweighted) KDE can be used downstream."""
    rng = rng or np.random.default_rng(0)          # fixed seed → reproducible contours
    n = cols[0].size
    if weights is not None:
        idx = rng.choice(n, size=min(n_max, n), replace=False,
                         p=weights / weights.sum())
    elif n > n_max:
        idx = rng.choice(n, size=n_max, replace=False)
    else:
        idx = slice(None)
    return [c[idx] for c in cols]


def _kde_grid(x, y, gridsize=GRID, pad=0.15):
    """2D gaussian KDE of (x, y) evaluated on a regular grid spanning the data
    plus a `pad` fractional margin.  Returns (X, Y, Z) meshes."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    kde = gaussian_kde(np.vstack([x, y]))          # weights already resampled in
    dx, dy = np.ptp(x) * pad, np.ptp(y) * pad
    xs = np.linspace(x.min() - dx, x.max() + dx, gridsize)
    ys = np.linspace(y.min() - dy, y.max() + dy, gridsize)
    X, Y = np.meshgrid(xs, ys)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def _density_levels(Z, fractions):
    """Density thresholds enclosing each probability fraction.  Grid cells have
    equal area, so summing Z from the peak down ∝ the enclosed posterior mass."""
    z = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(z)
    csum /= csum[-1]
    return [float(z[np.searchsorted(csum, f)]) for f in fractions]


def extract_2d_contour(x, y, weights=None, levels=FRACTIONS, gridsize=GRID):
    """{fraction: (N,2) path in (x, y)} for the enclosed-probability contours of a
    2D KDE of (x, y).  Weighted samples are resampled to an effective unweighted
    set first."""
    x, y = _subsample([np.asarray(x, float), np.asarray(y, float)], weights=weights)
    X, Y, Z = _kde_grid(x, y, gridsize)
    dlev = _density_levels(Z, levels)
    order = np.argsort(dlev)                        # contour() needs increasing levels
    cs = plt.figure().gca().contour(X, Y, Z, levels=[dlev[i] for i in order])
    plt.close("all")
    out = {}
    for pos, i in enumerate(order):
        segs = cs.allsegs[pos]
        # ponytail: keep the longest segment — these posteriors are unimodal, so
        # one closed blob per level.  NaN-join the segments if a multi-modal
        # source ever needs every island.
        out[levels[i]] = max(segs, key=len) if segs else np.empty((0, 2))
    return out


def _save_contour(name, paths, header):
    """Write each fraction's path as <name>_68.csv / <name>_95.csv."""
    for frac, path in paths.items():
        f = CONTOURS / f"{name}_{round(frac * 100)}.csv"
        np.savetxt(f, path, delimiter=",", header=header, comments="")
        print(f"  wrote {f.name}  ({len(path)} pts)")


# ────────────────────────────────────────────────────────────── data categories
# 1. Mass-only Gaussian sources (no samples): 1σ / 2σ intervals.
MASS_TARGETS = {                 # name: (mean, sigma)  [M_sun]
    "J0952-0607": (2.24, 0.17),  # rotation-corrected mass, Romani et al. 2022
    "J0740":      (2.08, 0.07),  # Shapiro-delay mass, Fonseca et al. 2021 (NANOGrav)
    "GW190814":   (2.59, 0.05),  # secondary compact object, Abbott et al. 2020
}

def compute_mass_bounds():
    rows = []
    for name, (mu, sig) in MASS_TARGETS.items():
        for frac, k in ((68, 1.0), (95, 2.0)):       # 1σ / 2σ
            rows.append((name, frac, mu - k * sig, mu + k * sig))
    out = CONTOURS / "mass_bounds.csv"
    with open(out, "w") as fh:
        fh.write("name,level,lower,upper\n")
        for name, frac, lo, hi in rows:
            fh.write(f"{name},{frac},{lo:.6f},{hi:.6f}\n")
    print(f"  wrote {out.name}  ({len(rows)} rows)")


# 2. Mass-Radius 2D posteriors.  spec: (file, col_R, col_M, col_weight|None)
MR_SOURCES = {
    "J0030": ("J0030.txt", 0, 1, None),   # header-documented: R, M
    "J0740": ("J0740.txt", 0, 1, 2),      # header-documented: R, M, weight
    "HESS":  ("HESS.txt",  0, 1, None),   # header "# R_NS (km) M_NS (Msun)"
    "J0614": ("J0614.dat", 1, 0, None),   # header-less: col0=M, col1=R
}

def compute_mr_contours():
    for name, (fname, cR, cM, cW) in MR_SOURCES.items():
        a = np.loadtxt(SAMPLES / fname)               # '#' comments skipped by default
        R, M = a[:, cR], a[:, cM]
        W = a[:, cW] if cW is not None else None
        paths = extract_2d_contour(R, M, weights=W)   # x=R, y=M → save (R, M)
        _save_contour(name, paths, header="R_km,M_sun")
        print(f"{name}: {len(a)} samples → contours")


# 3. GW170817 mass-tidal.  header-less; cols: 2=M1 3=M2 4=Λ1 5=Λ2 (verified by
#    column stats: M1>M2, Λ ~ hundreds).
def _chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

def _lambda_tilde(m1, m2, l1, l2):
    return (16.0 / 13.0) * ((m1 + 12 * m2) * m1**4 * l1 +
                            (m2 + 12 * m1) * m2**4 * l2) / (m1 + m2) ** 5

# spec: name -> (file, cols=(cM1,cM2,cΛ1,cΛ2)).  gw170817.dat carries the tidal
# params at cols 2..5; the GW190425 extract (from fetch_gw190425.py) is a plain
# 4-column m1,m2,Λ1,Λ2 table, so its cols are 0..3.
# Both from official LVK releases via fetch_gw1708NN.py: low-spin prior,
# source-frame masses, layout m1,m2,Λ1,Λ2 → cols 0..3.
TIDAL_SOURCES = {
    "GW170817": ("gw170817_extracted_table.txt",  (0, 1, 2, 3)),
    "GW190425": ("gw190425_extracted_table.txt",  (0, 1, 2, 3)),
}

def compute_tidal_contour():
    for name, (fname, (cM1, cM2, cL1, cL2)) in TIDAL_SOURCES.items():
        if not (SAMPLES / fname).exists():
            print(f"  {name}: {fname} missing — run fetch_gw190425.py; skipping")
            continue
        a = np.loadtxt(SAMPLES / fname)
        m1, m2, l1, l2 = a[:, cM1], a[:, cM2], a[:, cL1], a[:, cL2]
        Mc = _chirp_mass(m1, m2)
        Lt = _lambda_tilde(m1, m2, l1, l2)
        paths = extract_2d_contour(Mc, Lt)            # x=Mchirp, y=Λ̃
        _save_contour(name, paths, header="Mchirp_Msun,LambdaTilde")
        print(f"{name}: {len(a)} samples → M-Λ̃ contours")


# 3b. GW170817 mass-radius: header-less; cols 0=M1 1=M2 2=Λ1 3=Λ2 4=R1 5=R2
#     (verified by column stats).  One contour per NS component (the two share
#     the same EoS → same R(M) band, so they get the same colour downstream).
def compute_gw170817_mr_contour():
    a = np.loadtxt(SAMPLES / "GW170817_MR.txt")
    for tag, cM, cR in (("GW170817MR_1", 0, 4), ("GW170817MR_2", 1, 5)):
        paths = extract_2d_contour(a[:, cR], a[:, cM])   # x=R, y=M → save (R, M)
        _save_contour(tag, paths, header="R_km,M_sun")
    print(f"GW170817MR: {len(a)} events → 2 component M-R contours")


# 4. Nuclear bands: local files are already (rho, lower, upper) — re-emit as csv.
#    chiral_eft.txt is a χEFT PNM energy band in the same 3-column format.
BAND_FILES = ("FOPI_2016_eSM.txt", "FOPI_2016_PSM.txt", "DLL_2002_PSM.txt",
              "ASYEOS_2016_Esym.txt", "chiral_eft.txt")

def convert_band_files():
    for fname in BAND_FILES:
        a = np.loadtxt(SAMPLES / fname)
        out = CONTOURS / (Path(fname).stem + ".csv")
        np.savetxt(out, a, delimiter=",",
                   header="rho_fm3,lower,upper", comments="")
        print(f"  wrote {out.name}  ({len(a)} rows)")


def fetch_nucleardatapy_bands():
    """χEFT (microscopic) and HIC bands straight from the nucleardatapy API.
    Best-effort: if the package/tables are missing the pipeline still finishes.
    See https://jeromemargueron.github.io/nucleardatapy/ ."""
    try:
        import nucleardatapy as nda
    except ImportError:
        print("  nucleardatapy not installed — skipping χEFT/HIC API bands")
        # TODO: pip install nucleardatapy  (or download its data tables manually)
        return
    # χEFT E/A bands: neutron matter (PNM) and symmetric matter (SNM).
    for matter, tag in (("NM", "chiEFT_PNM_E"), ("SM", "chiEFT_SNM_E")):
        try:
            b = nda.setupMicroBand(matter=matter)     # central ± 1σ across models
            arr = np.column_stack([b.den, b.e2a_int - b.e2a_std,
                                   b.e2a_int + b.e2a_std])
            out = CONTOURS / f"{tag}.csv"
            np.savetxt(out, arr, delimiter=",",
                       header="rho_fm3,lower,upper", comments="")
            print(f"  wrote {out.name}  ({len(arr)} rows)  [{b.models}]")
        except Exception as e:                        # noqa: BLE001 - keep pipeline alive
            print(f"  χEFT {matter} band unavailable: {e}")
            # TODO: some models need their data tables downloaded — see nucleardatapy docs.


# ───────────────────────────────────────────────────────────────────── driver
def main():
    CONTOURS.mkdir(parents=True, exist_ok=True)
    print("Mass-only bounds:");     compute_mass_bounds()
    print("Mass-radius contours:"); compute_mr_contours()
    print("Tidal contour:");        compute_tidal_contour()
    print("GW170817 M-R contour:"); compute_gw170817_mr_contour()
    print("Nuclear bands (local):"); convert_band_files()
    print("Nuclear bands (nucleardatapy):"); fetch_nucleardatapy_bands()
    print(f"\nDone → {CONTOURS}")


# ── tiny self-check: a known isotropic Gaussian's 1σ contour must enclose ~68%
# of its own samples and be ~1σ in radius.  Fails loudly if the level math breaks.
def _selfcheck():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(6000), rng.standard_normal(6000)
    paths = extract_2d_contour(x, y)
    for frac, r_expect in ((0.68, 1.515), (0.95, 2.486)):   # 2D χ radii: √(-2 ln(1-f))
        r = np.hypot(*paths[frac].T).mean()
        assert abs(r - r_expect) < 0.35, f"{frac}: contour r={r:.2f} vs {r_expect:.2f}"
    print("selfcheck OK")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
