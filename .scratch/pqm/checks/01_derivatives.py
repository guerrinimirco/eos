"""Gate of .scratch/pqm/issues/01-closed-form-derivatives.md.

Runs the prototype's definitions -- `notebooks/csc_bag_map.py` up to its first
driving cell, so no plot and no NJL solve beyond the on-disk cache -- and
checks every bullet of that ticket's Gate.

    python3 .scratch/pqm/checks/01_derivatives.py
"""
import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "notebooks" / "csc_bag_map.py").read_text()


def _run(text, ns):
    exec(compile(text, "csc_bag_map.py", "exec"), ns)


NB = {"__name__": "csc_bag_map"}
_run(SRC[:SRC.index("FIT_PAR = replace(")], NB)          # every definition
i = SRC.index("FIT_PAR = replace(")
_run(SRC[i:SRC.index("# ------", i)], NB)                # FIT_PAR, TERM_SETS
i = SRC.index("#: The term set the rows above justify")
_run(SRC[i:SRC.index("# %% [markdown]", i)], NB)         # TERMS

basis_terms, pressure = NB["basis_terms"], NB["pressure"]
charges, entropy = NB["charges"], NB["entropy"]
PATTERNS, TERMS, FIT_PAR = NB["PATTERNS"], NB["TERMS"], NB["FIT_PAR"]
S_MASSES = NB["S_MASSES"]

fits = {p: NB["fit_phase"](NB["sample"](FIT_PAR, p), p, TERMS)
        for p in PATTERNS}
GRID = list(itertools.product(NB["GRID_MU_B"], NB["GRID_MU_C"],
                              NB["GRID_MU_S"]))
HOT = [(1400.0, -40.0, 0.0, 10.0), (1400.0, -40.0, 0.0, 30.0),
       (1600.0, 40.0, 80.0, 50.0), (1000.0, 0.0, -80.0, 20.0),
       (1800.0, -80.0, 0.0, 80.0)]
fail = []


def check(name, worst, tol):
    ok = worst <= tol
    fail.append(name) if not ok else None
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} {worst:9.2e} <= {tol:.0e}")


def state(f, mu_B, mu_C, mu_S, T):
    """P, the three densities, s, and eps and f built from them."""
    P = pressure(f["coeff"], mu_B, mu_C, mu_S, T, f["pattern"],
                 f["Delta_star"], f["sigma"])
    n = charges(f, mu_B, mu_C, mu_S, T)
    s = entropy(f, mu_B, mu_C, mu_S, T)
    mu_n = mu_B * n[0] + mu_C * n[1] + mu_S * n[2]
    eps = -P + mu_n + T * s
    return P, n, s, eps, eps - T * s, mu_n


# --------------------------------------------------------------------------
print("\n=== the identities (CLAUDE.md section 8), asserted ===")
print("  structural given eps := -P + sum_a mu_a n_a + T s -- what makes them")
print("  hold is that one Omega supplies P, every n_a and s; the content is")
print("  the FD gate below, which is what says the slopes ARE that Omega's.")
worst = {"euler": 0.0, "f=eps-Ts": 0.0, "f=-P+sum mu n": 0.0}
for pattern in PATTERNS:
    f = fits[pattern]
    for mu_B, mu_C, mu_S in GRID:
        for T in (0.0,):
            P, n, s, eps, fe, mu_n = state(f, mu_B, mu_C, mu_S, T)
            if not np.isfinite(P) or n[0] <= 0:
                continue
            sc = max(abs(eps), abs(P), 1e-30)
            worst["euler"] = max(worst["euler"],
                                 abs(eps + P - T * s - mu_n) / sc)
            worst["f=eps-Ts"] = max(worst["f=eps-Ts"], abs(fe - (eps - T * s)) / sc)
            worst["f=-P+sum mu n"] = max(worst["f=-P+sum mu n"],
                                         abs(fe - (-P + mu_n)) / sc)
    for mu_B, mu_C, mu_S, T in HOT:
        P, n, s, eps, fe, mu_n = state(f, mu_B, mu_C, mu_S, T)
        sc = max(abs(eps), abs(P), 1e-30)
        worst["euler"] = max(worst["euler"], abs(eps + P - T * s - mu_n) / sc)
        worst["f=eps-Ts"] = max(worst["f=eps-Ts"], abs(fe - (eps - T * s)) / sc)
        worst["f=-P+sum mu n"] = max(worst["f=-P+sum mu n"],
                                     abs(fe - (-P + mu_n)) / sc)
print(f"  {len(GRID)} T = 0 points + {len(HOT)} T > 0, each in "
      f"{len(PATTERNS)} patterns")
for name, w in worst.items():
    check(name, w, 1e-12)

# --------------------------------------------------------------------------
print("\n=== closed form against the central difference it replaces ===")
print("  a chain-rule bug does NOT shrink with h; truncation does, as h^2.")


def fd(f, mu_B, mu_C, mu_S, T, h):
    out = []
    for axis in range(4):
        hi = [mu_B, mu_C, mu_S, T]
        lo = [mu_B, mu_C, mu_S, T]
        hi[axis] += h
        lo[axis] -= h
        out.append((pressure(f["coeff"], *hi, f["pattern"], f["Delta_star"],
                             f["sigma"])
                    - pressure(f["coeff"], *lo, f["pattern"],
                               f["Delta_star"], f["sigma"])) / (2.0 * h))
    return out


# Away from a massive-gas threshold, where a difference straddles a kink in
# the exact function and measures the kink instead of the slope.
def clear_of_threshold(f, mu_B, mu_C, mu_S, h):
    mu_s = (NB["flavour_mu"](mu_B, mu_C, mu_S)[2] if f["pattern"] != "CFL"
            else (mu_B + mu_S) / 3.0)
    return all(abs(mu_s - m) > 4.0 * max(h, 1.0) for m in S_MASSES)


TEST = [(1200.0, -40.0, 0.0), (1500.0, 0.0, 80.0), (1700.0, 40.0, -80.0),
        (1000.0, -80.0, 0.0), (1650.0, 0.0, 0.0)]
fd_err = {}                      # the h = 1 MeV truncation, per pattern, T = 0
for T in (0.0, 30.0):
    print(f"  T = {T:g} MeV" + (" (s = dP/dT has a row only at T > 0)"
                               if T else ""))
    for pattern in PATTERNS:
        f = fits[pattern]
        errs = {}
        for h in (4.0, 2.0, 1.0, 0.5):
            e = 0.0
            for mu_B, mu_C, mu_S in TEST:
                if not clear_of_threshold(f, mu_B, mu_C, mu_S, h):
                    continue
                exact = list(charges(f, mu_B, mu_C, mu_S, T)) + \
                    [entropy(f, mu_B, mu_C, mu_S, T)]
                approx = fd(f, mu_B, mu_C, mu_S, T, h)
                for x, a in zip(exact, approx):
                    if abs(x) > 1e-6:
                        e = max(e, abs(a - x) / abs(x))
            errs[h] = e
        if T == 0.0:
            fd_err[pattern] = errs[1.0]
        ratio = errs[4.0] / max(errs[0.5], 1e-300)
        print(f"    {pattern:9s} " +
              "  ".join(f"h={h:g}: {e:.2e}" for h, e in errs.items()) +
              f"   h=4 / h=0.5 = {ratio:.0f}x  (h^2 would be 64x)")
        check(f"FD shrinks with h, {pattern}, T={T:g}", 1.0 / max(ratio, 1e-30),
              1.0 / 30.0)

# --------------------------------------------------------------------------
print("\n=== the CFL locking, out of the derivative and not out of a filter ===")
f = fits["CFL"]
w_C = w_S = 0.0
for mu_B, mu_C, mu_S in GRID:
    for T in (0.0, 30.0):
        n_B, n_C, n_S = charges(f, mu_B, mu_C, mu_S, T)
        if n_B <= 0:
            continue
        w_C = max(w_C, abs(n_C) / n_B)
        w_S = max(w_S, abs(n_S / n_B - 1.0))
check("CFL |n_C| / n_B", w_C, 1e-12)
check("CFL |n_S/n_B - 1|", w_S, 1e-12)

# --------------------------------------------------------------------------
print("\n=== refit: closed-form rows against the FD rows they replace ===")


def fit_fd(records, pattern, terms, h=1.0):
    """`fit_phase` as it was: the density rows by central difference."""
    Delta_star, sigma = NB["gap_powerlaw"](records)
    use = NB["usable_terms"](terms, pattern)

    def vector(mu_B, mu_C, mu_S, T):
        X = NB["basis"](mu_B, mu_C, mu_S, T, pattern, Delta_star, sigma)
        return np.array([X[t] for t in use])

    P_scale = max(abs(r["P"]) for r in records)
    n_scale = max(abs(r["n_B"]) for r in records)
    A, b = [], []
    for r in records:
        mu = (r["mu_B"], r["mu_C"], r["mu_S"], r["T"])
        A.append(vector(*mu) / P_scale)
        b.append(r["P"] / P_scale)
        for axis, target in enumerate(("n_B", "n_C", "n_S")):
            hi = list(mu); hi[axis] += h
            lo = list(mu); lo[axis] -= h
            A.append((vector(*hi) - vector(*lo)) / (2.0 * h) / n_scale)
            b.append(r[target] / n_scale)
    A, b = np.array(A), np.array(b)
    Xinf = NB["basis"](NB["MU_CONFORMAL"], 0.0, 0.0, 0.0, pattern, Delta_star,
                       sigma)
    P_free_inf = 3.0 * (NB["MU_CONFORMAL"] / 3.0) ** 4 / (4.0 * NB["PI2"]
                                                          * NB["HC3"])
    A = np.vstack([A, NB["CONFORMAL_WEIGHT"]
                   * np.array([Xinf[t] for t in use]) / P_free_inf])
    b = np.concatenate([b, [NB["CONFORMAL_WEIGHT"]]])
    A = np.vstack([A, NB["RIDGE"] * np.eye(len(use))])
    b = np.concatenate([b, np.zeros(len(use))])
    from scipy.optimize import lsq_linear
    res = lsq_linear(A, b, bounds=([NB["TERM_BOUNDS"][t][0] for t in use],
                                   [NB["TERM_BOUNDS"][t][1] for t in use]))
    return dict(zip(use, res.x))


# What the refit is allowed to move by is the h = 1 MeV truncation of the rows
# it replaces, measured above, per pattern. The comparison is on the EoS the
# coefficients PREDICT and not on the coefficients themselves: a4, a4l and a6
# are nearly collinear over any finite mu range, so a single coefficient is
# free to move inside the ridge's valley without the pressure moving at all
# (map.md, "a small Tikhonov ridge is REQUIRED"). Both are printed.
for pattern in PATTERNS:
    recs = NB["sample"](FIT_PAR, pattern)
    old = fit_fd(recs, pattern, TERMS)
    new = fits[pattern]["coeff"]
    f_old = dict(fits[pattern], coeff=old)
    dP = dn = 0.0
    P_scale = max(abs(r["P"]) for r in recs)
    n_scale = max(abs(r["n_B"]) for r in recs)
    for r in recs:
        mu = (r["mu_B"], r["mu_C"], r["mu_S"], r["T"])
        dP = max(dP, abs(pressure(new, *mu, pattern, f_old["Delta_star"],
                                  f_old["sigma"])
                         - pressure(old, *mu, pattern, f_old["Delta_star"],
                                    f_old["sigma"])) / P_scale)
        dn = max(dn, abs(charges(fits[pattern], *mu)[0]
                         - charges(f_old, *mu)[0]) / n_scale)
    drift = {t: abs(new[t] - old[t]) for t in old}
    worst_t = max(drift, key=lambda t: drift[t] / max(abs(old[t]), 1e-12))
    print(f"  {pattern:9s} ({len(recs)} pts)  a4 = {new['a4']:.4f}   "
          f"worst coefficient {worst_t}: {old[worst_t]:+.5g} -> "
          f"{new[worst_t]:+.5g} ({drift[worst_t] / max(abs(old[worst_t]), 1e-12):.1e} rel)"
          f"\n{'':12s}predicted P moves {dP:.2e}, n_B {dn:.2e} of scale; "
          f"the FD rows it replaces were good to {fd_err[pattern]:.2e}")
    check(f"refit moves P by less than the FD error, {pattern}", dP,
          fd_err[pattern])
    check(f"refit moves n_B by less than the FD error, {pattern}", dn,
          fd_err[pattern])

print("\n" + ("FAILED: " + ", ".join(fail) if fail else "all gates pass"))
sys.exit(1 if fail else 0)
