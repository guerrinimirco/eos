"""Ticket 19: is the in-medium pass's 12-node failure a LARGE-T effect?

The collar panels are [k_F - 25T, k_F] and [k_F, k_F + 25T], and the Fermi
factor's poles sit pi T from k_F, so the pole-to-panel ratio does not depend
on T. If that is what limits Gauss here, 12 nodes fail at T = 1-5 MeV as they
do at 20-30. Same points as ticket 10 section 4 (Parameters.default(), one
pattern held, reference backend), hot 12 against hot 24 on the landed vacuum.

    PYTHONPATH=. python3 .scratch/njl-speed/t19_smallT.py
"""
import eos
from eos import njl

print(eos.__file__)
par = njl.Parameters.default()
flags = njl.SpeciesFlags(csc=True)
for pattern, n_B in (("2SC", 1.4), ("CFL", 1.2)):
    for T in (1.0, 2.0, 5.0, 10.0, 20.0):
        pts = {}
        for nodes in (24, 12):
            r = njl.eos_point(par, "beta_eq_neutrinoless", flags, n_B=n_B, T=T,
                              patterns=(pattern,), backend="reference",
                              pair_nodes_per_panel=nodes)
            pts[nodes] = r
        a, b = pts[24].point, pts[12].point
        print(f"{pattern} n_B={n_B} T={T:4.1f}: ok {pts[24].ok}/{pts[12].ok} "
              f"realised {a.pattern_realised}/{b.pattern_realised}  12 vs 24: "
              f"dP/P {abs(b.P / a.P - 1):.1e}  ds/s {abs(b.s / a.s - 1):.1e}",
              flush=True)
