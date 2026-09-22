"""Ticket 10, step 1: quadrature error of each RG pass, in isolation.

At converged CFL and 2SC states of the pinned configuration, each of the three
passes of `rg_pair_block` -- hot at Lambda_UV, vacuum at Lambda_UV, vacuum at
Lambda -- is evaluated on the SHIPPED panel layout at N nodes per panel and
compared with the same layout at 64. The combined block is compared too, both
with all three passes at N and with the hot pass held at 24 while only the
vacuum pair moves: whether the vacuum half can be cheapened on its own depends
on whether the hot and vacuum tail errors cancel, which only the combination
shows.

    PYTHONPATH=. python3 .scratch/njl-speed/t10_quad_accuracy.py
"""
import numpy as np

from eos import njl
from eos.general.pairing import pair_block, pair_nodes
from eos.njl.thermodynamics import RG_PANEL_RATIO

par = njl.Parameters.named("rg_njl1")
FLAGS = njl.SpeciesFlags(csc=True)
NS = (6, 8, 10, 12, 16, 20, 24, 32)
REF = 64
FIELDS = ("delta_omega", "delta_n", "delta_rho_s", "gap_kernel")


def hot(M, mu, D, N):
    rule = pair_nodes(M, mu, 0.0, par.Lambda_medium, N, RG_PANEL_RATIO,
                      Delta=D)
    return pair_block(M, mu, D, 0.0, par.Lambda_medium, backend="fast",
                      quadrature=rule), rule[0].size


def vac(M, D, k_max, N):
    zero = np.zeros(9)
    rule = pair_nodes(M, zero, 0.0, k_max, N, RG_PANEL_RATIO)
    return pair_block(M, zero, D, 0.0, k_max, backend="fast",
                      quadrature=rule), rule[0].size


def combine(h, a, b):
    return {f: np.atleast_1d(getattr(h, f) - getattr(a, f) + getattr(b, f))
            for f in FIELDS}


def err(x, ref):
    """max |x - ref| / max |ref|, per field: relative to the field's size."""
    out = {}
    for f in FIELDS:
        r = np.atleast_1d(ref[f])
        out[f] = float(np.max(np.abs(np.atleast_1d(x[f]) - r))
                       / max(np.max(np.abs(r)), 1e-300))
    return out


def fmt(e):
    return "  ".join(f"{f[6:] if f.startswith('delta_') else f[:6]:>6s} "
                     f"{e[f]:8.1e}" for f in FIELDS)


for pattern, densities in (("CFL", (0.70, 1.00, 1.149, 1.50)),
                           ("2SC", (0.55, 0.90, 1.30))):
    for n_B in densities:
        r = njl.eos_point(par, "beta_eq_neutrinoless", FLAGS, n_B=n_B, T=0.0,
                          patterns=(pattern,), backend="fast")
        st = r.point._state
        M, mu, D = np.array(st.M), np.array(st.mu_star), np.array(st.Delta)
        print(f"\n=== {pattern} n_B={n_B}  conv={r.ok} realised="
              f"{r.point.pattern_realised} gapless={r.point.gapless} "
              f"M={np.round(M, 1)} Delta={np.round(D, 2)}")
        H = {N: hot(M, mu, D, N) for N in NS + (REF,)}
        A = {N: vac(M, D, par.Lambda_medium, N) for N in NS + (REF,)}
        B = {N: vac(M, D, par.Lambda, N) for N in NS + (REF,)}
        ref = combine(H[REF][0], A[REF][0], B[REF][0])
        print(f"  nodes at 24: hot {H[24][1]}, vac(UV) {A[24][1]}, "
              f"vac(L) {B[24][1]}")
        for N in NS:
            e_hot = err({f: getattr(H[N][0], f) for f in FIELDS},
                        {f: getattr(H[REF][0], f) for f in FIELDS})
            e_vuv = err({f: getattr(A[N][0], f) for f in FIELDS},
                        {f: getattr(A[REF][0], f) for f in FIELDS})
            e_vl = err({f: getattr(B[N][0], f) for f in FIELDS},
                       {f: getattr(B[REF][0], f) for f in FIELDS})
            e_all = err(combine(H[N][0], A[N][0], B[N][0]), ref)
            e_vac = err(combine(H[24][0], A[N][0], B[N][0]), ref)
            print(f"  N={N:2d} hot   {fmt(e_hot)}")
            print(f"       vacUV {fmt(e_vuv)}")
            print(f"       vacL  {fmt(e_vl)}")
            print(f"       ALL@N {fmt(e_all)}")
            print(f"       hot24+vac@N {fmt(e_vac)}")
