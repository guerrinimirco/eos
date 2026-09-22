"""Ticket 19: is the reference backend's 3.5e-9 in deep CFL the rule or the solve?

At the grid density where reference-landed moved P by 3.5e-9 against every
other arm (fast-landed moved 5e-11), solve CFL on the reference backend, landed
rule, from three seeds: cold, the fast-landed root, and the reference-control
root. Report M_u, P, the solver's error, and the reference residual (scaled)
at the fast-landed root.

    PYTHONPATH=. python3 .scratch/njl-speed/t19_ref_deep_cfl.py
"""
import numpy as np
import eos
from eos import njl
from eos.njl import solver
import eos.njl.thermodynamics as th
import eos.njl.backends.jacobian as jac

print(eos.__file__)
par = njl.Parameters.named("rg_njl1")
flags = njl.SpeciesFlags(csc=True)
n_B = float(np.linspace(0.5, 1.55, 200)[177])
print(f"n_B = {n_B}")


def arm(vac):
    th.VACUUM_NODES_PER_PANEL = jac.VACUUM_NODES_PER_PANEL = vac


def solve(backend, x0=None):
    return njl.solve(par, "beta_eq_neutrinoless", n_B, 0.0, flags,
                     patterns=("CFL",), backend=backend, x0=x0)


arm(12)
fast = solve("fast")
arm(24)
ref_ctl = solve("reference")
arm(12)
for label, x0 in (("cold", None), ("seed fast-landed root", fast.x),
                  ("seed ref-control root", ref_ctl.x)):
    p = solve("reference", x0)
    print(f"reference landed, {label:24s}: M_u {p.M[0]:.12f}  P {p.P!r}  "
          f"error {p.error:.1e}  dP/P vs fast-landed {p.P / fast.P - 1:+.2e}")
print(f"fast landed                                : M_u {fast.M[0]:.12f}  P {fast.P!r}  error {fast.error:.1e}")
print(f"reference control (24/24)                  : M_u {ref_ctl.M[0]:.12f}  P {ref_ctl.P!r}  error {ref_ctl.error:.1e}")
