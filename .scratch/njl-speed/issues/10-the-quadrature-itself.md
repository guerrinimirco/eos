# The pairing quadrature is 77% of a good solve: what is the gate paying for?

Type: prototype
Status: open
Blocked by: 03
Parent: ../map.md

## Question

[Where do the 443 ms of a CFL solve actually go?](03-profile-one-cfl-solve.md)
found that a converged warm-started CFL solve is **77.3% jitted pairing
quadrature** and only 4.6% Python. Compilation is therefore spent
(ticket 06 is worth at most 1.3x) and the dead branches are ticket 05's.
What is left, and it is the whole of a good solve, is the quadrature itself:
**fewer passes, or cheaper ones.**

Two mechanisms, measured, in one ticket because one session measures both
with the same harness and the same gate:

### 1. The RG triple-pass, and a cache that misses two thirds of the time

`rg_pair_block` runs three passes — hot, vacuum at Lambda_UV, vacuum at
Lambda — because the nested form is what keeps the mass residual converging
(the shell form agrees to 1.8e-16 in delta_omega and leaves a 2SC solve
stalled at 9.0e-9). Measured: hot 288 nodes / 4.75 ms, each vacuum pass 192
nodes / 2.75 ms, so the subtraction is **54% of an uncached
`rg_pair_block`**; 11.75 ms cache-cold against 6.15 ms cache-hot.

`_vacuum_pair_block` is keyed on (M, Delta) bytes and **both move every Newton
step**: over the CFL table, 19906 calls for 12872 misses, a **35% hit rate**.
So the vacuum subtraction is **25.6% of the whole CFL build and 14.0% of a
converged solve**, and the same triple-pass is paid again inside
`rg_pair_jacobian`.

The question is not whether to drop it — CLAUDE.md section 2 and the
docstring both say what the scheme costs and why. It is whether the vacuum
half, which sits at mu* = 0 with no Fermi surface in it, needs the same node
count, the same panel rule and the same exactness as the hot pass it is
subtracted from.

### 2. The 288-node rule nobody has measured the gate against

`pair_nodes_per_panel` is already an argument all the way down from
`eos_point`, and `NODES_PER_PANEL` has never been varied against the
convergence gate. The solve terminates at `tol=1e-13` on the SCALED rows and
the map's correctness gate is P to 1e-8 — those are four orders apart, and
the quadrature rule was chosen for the tighter one.

**How much of the 288 nodes is the 1e-8 gate actually buying?**

## Scope, and the line it must not cross

- **Exact solves only** (the map's settled rule). This is a quadrature-accuracy
  question, not interpolation: every delivered point is still a converged
  solve of the same equations, and nothing is fitted between points.
- A coarser rule that converges to a DIFFERENT root, or that loses a gapless
  state, is a wrong answer and not a faster one. The map has already lost
  gapless CFL to a metastable 2SC once, through exactly this quadrature.
- The node rule is `eos/general/pairing.py`'s, shared with every other model
  that pairs. A change to the DEFAULT is a change to all of them; a per-call
  argument already exists and is the cheap answer if the default must stand.

## Gate

- ms/pt on the pinned benchmark, single-pattern CFL and 2SC, with the same
  attribution ticket 03 took, at each node count tried.
- P to 1e-8 and the same realised pattern at every benchmark density, worst
  point reported — and the gapless densities checked by name, since they are
  where a coarse rule fails first.
- A statement of what the RG vacuum half needs, separate from the hot pass:
  either a measured cheaper rule for it, or the measurement showing it needs
  the same one.
- The multiple this gives on a converged solve, against the 80 ms/pt that
  ticket 03 measured and the map's 1 ms/pt destination.
