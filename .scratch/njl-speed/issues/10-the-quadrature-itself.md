# The pairing quadrature is 77% of a good solve: what is the gate paying for?

Type: prototype
Status: closed
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

## Resolution, 2026-09-22

**At T = 0 the shipped 24 nodes per panel buy nothing the gate can see, and
the RG vacuum half needs half its nodes at every temperature. But the hot
pass's 24 is what FINITE T needs, so the shared default stands: the lever is
a T = 0 rule and a vacuum-only rule, and it is worth 1.67x on a converged CFL
solve at 12 nodes (2.16x at 8). No coarse rule lost a gapless state. The rule
that did was the SHIPPED one, through a root-selection lottery that ticket
16's bound makes systematic ([ticket 18](18-bound-loses-gapless-cfl.md)).**

Stack: python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0, HEAD `e2729bd`, AC
power. **The machine was not quiet**: eight foreign 3.14 multiprocessing
workers (16-18 h old) and a Jupyter kernel held ~80% each throughout, and
loadavg ran 19-164 (cpu/wall 0.59-0.94). Only ratios inside one interleaved
run are quoted. The absolute ms/pt are not comparable to tickets 02/03.

Harness (`.scratch/njl-speed/t10_*.py`, logs and JSON beside them). An ARM
`H/V` is H Gauss-Legendre nodes per panel for the hot pass (the
`pair_nodes_per_panel` argument, which only reaches the pairing block) and V
for the two vacuum passes. V is set by re-pointing the module globals that
`rg_pair_block` and `rg_pair_jacobian` look the vacuum blocks up through, so
the residual and its Jacobian always see the same vacuum rule. `24/24` is the
shipped rule. `t10_tables.py time` runs the arms INTERLEAVED, repeat by
repeat, so they share one load window. `t10_tables.py attr` rebuilds ticket
03's self-time wrapper stack, and reproduces ticket 03's CFL counts exactly
at 24/24: 9953 hot passes, 12872 vacuum misses, 35% hit rate. `t10_gate.py`
matches each delivered row to its GRID density (the row carries the solved
n_B, which moves at 1e-10, and keying on it silently compares nothing).

### 1. Each pass alone (`t10_quad_accuracy.log`, `t10_vac_layout.log`)

Relative error of each pass against the SAME panel layout at 64 nodes, at
converged fast-backend states (CFL at 0.70/1.00/1.149/1.50, 2SC at
0.55/0.90/1.30 fm^-3); delta_rho_s shown, the slowest field everywhere (the
gap kernel is 1e-13 or better from N = 8):

| pass | N = 6 | 8 | 10 | 12 | 24 (shipped) |
|---|---|---|---|---|---|
| hot, Lambda_UV | 1.5e-9 .. 4.9e-8 | 4e-12 .. 2.1e-10 | <= 2.3e-12 | <= 2e-12 | 2e-13 .. 1.6e-12 |
| vacuum, Lambda | <= 8.5e-11 | <= 6.5e-14 | floor | floor | floor |
| vacuum, Lambda_UV, CFL | <= 1.1e-10 | <= 2.8e-12 | floor | floor | floor |
| vacuum, Lambda_UV, 2SC at 1.30 (M_u = 10.7 MeV) | 2.3e-7 | **1.6e-8** | 5.3e-10 | 2.4e-11 | 2.7e-12 |

The hot pass is converged by N = 10-12 and the Lambda vacuum by N = 8. The
Lambda_UV vacuum is the slow one, and only in 2SC at high density, which is a
**layout effect**: its geometric panels stop at Lambda_UV/2^7 = 47 MeV, so its
lowest panel holds the light-quark mass, while the Lambda vacuum's lowest
panel is [0, 4.7 MeV]. With breakpoints at the three constituent masses ON
TOP of the shipped edges, N = 8 reaches 2.8e-12 there, **88 nodes against
192**. (Not through `panel_nodes`: a breakpoint raises the floor of its
geometric tail and deletes every panel below it, which made the first attempt
at this measurement 1e-6 WORSE.)

### 2. The pinned tables at T = 0: the gate

Arms 24/24 (control), 16/16, 12/12, 10/10, 8/8, and vacuum-only 24/16, 24/12,
24/8; single pattern, pinned grid, both backends.

| | rows | realised mismatches | worst \|dP\|/P, where |
|---|---|---|---|
| 2SC fast, all 8 arms | 200/200 | 0 | 6.4e-10 .. 9.9e-10, **all at 0.7796**, where the CONTROL is the outlier |
| CFL fast, all 8 arms | 188/188 | 0 | 2.9e-9 .. 4.6e-9, all at the branch onset 0.5633-0.6108 |
| 2SC reference, 24/16/12/10 and 24/12 | 200/200 | 0 | <= 1.1e-10 vs the reference control; <= 8.3e-10 vs the fast one |
| CFL reference, 24/12/10 and 24/12 | 188/188 | 0 | <= 3.5e-9 vs the reference control; <= 3.6e-9 vs the fast one |

Both worst points are solver-path noise, not quadrature. At 0.7796 every
arm agrees with every other to 1e-10 (M_u = 18.81731025x) and the control
alone sits at ...021. At the CFL onset, the vacuum-only 24/16 arm, which
moves no block output past 1e-11, shows 3.4e-9 too. Every arm's three timing
repeats are **bit-identical**. The rule changes no Newton count: 9.2-9.4
residuals and 3.8-3.95 Jacobians per 2SC point at every arm, and
9953-10485 / 933-968 over the CFL table.

What the gate cannot see, the masses can. P is stationary in M and Delta, so
a quadrature error in a gap equation reaches P only at second order. **Vacuum
N = 8 leaves |dM|/M = 2.2e-8 at 2SC n_B = 1.4234**, the M_u-panel effect of
section 1, gone at vacuum N = 10 (2.0e-9, the solver floor). CFL 8/8 moves
Delta by 5.6e-10 against <= 7e-11 for every other arm.

### 3. Gapless states, by name

**The pinned tables deliver no gapless row at all**, in either pattern on
either backend (the flag checked on every row). So the gapless check was
taken where this model's gapless ground state is documented: `fixed_YC`,
Y_C = 0.1, leptons, T = 0, 45 densities over 0.5-1.6, where gapless CFL is
the ground state at **0.775-1.6 (34 densities, named in
`t10_gapless_*.json`)**:

- **Held CFL** (`T10_PATTERNS=CFL`, every point own-seeded): at every arm, all
  34 densities realise gapless CFL, |dP|/P <= 6.3e-10 at each, by name. **The
  rule does not move a gapless state.** Below 0.775 the held CFL layout has
  several roots and the arms differ row by row, the vacuum-only 24/12
  included, so that region measures the sweep's path, not the rule.
- **The default enumeration** is where a state can be LOST, and there **the
  shipped rule loses it**: HEAD fast 24/24 delivers metastable 2SC at 13 of
  the 34 densities, 0.775-1.075, up to 37.7 MeV/fm^3 above the gapless CFL
  state, with P non-monotone across 1.075 -> 1.100. The coarse rules don't
  (16/16, 10/10, 8/8, 24/12 and 24/8 keep all 34; 12/12 loses one, 0.775). A FINER rule does
  too (32/32 loses 7), and on the reference backend a vacuum-only 24/12 loses
  7. Root selection at this onset is a lottery that any perturbation enters,
  so it cannot be charged to a quadrature rule in either direction. It is
  ticket 18's: `428cd66`, before the bound, gets the shipped table right, and
  from the unpaired cross seed the bounded ladder reaches gapless CFL at 0 of
  3 densities where rung D (the differenced repeat, which the bound removed)
  reaches it at 3 of 3.

### 4. Finite T: why the default stands (`t10_finiteT_probe.log`)

The pinned benchmark is T = 0 and the rule does not transfer. At T > 0 the
panels around each Fermi momentum become +-25 T thermal collars, several
hundred MeV wide, and the Fermi-Dirac tail across them costs nodes.
`Parameters.default()`, single pattern, reference backend, against 24:

| point | 16 | 12 | 10 |
|---|---|---|---|
| 2SC, T = 20, n_B = 1.4: dP/P, ds/s | 1.9e-9, 6.0e-9 | **1.3e-7**, 9.4e-7 | **1.3e-6**, 1.1e-5 |
| CFL, T = 30, n_B = 1.2 | 5.8e-10, 2.6e-8 | **1.7e-7**, 6.4e-7 | **2.3e-7**, 2.3e-5 |
| CFL, T = 20, n_B = 1.2 | 4.2e-10, 3.5e-10 | 2.3e-9, 1.4e-6 | **4.1e-7**, 9.6e-6 |

and 24 against 48 is <= 1.3e-11 in P, <= 2.2e-12 in s (20 is <= 8e-12 and
1e-10). **12 nodes fails the P gate at finite T**, and 16 passes it with s at
3e-8. The vacuum passes are evaluated at T = 0 by construction whatever the
state's temperature, so **their rule is T-independent**. Hot pass at 24 and
vacuum at 12 or 10, at the same three points: P <= 1.2e-9 fast / 4.3e-11
reference, s <= 1.5e-12, M <= 8e-10. Vacuum 8 leaves M at 2e-8 in 2SC here
too (`t10_finiteT_vacuum.log`).

### 5. What it buys: the pinned benchmark, n = 3, median, interleaved

cpu ms/pt, with the multiple against 24/24 in the same window. The per-repeat
ratios agree to a few percent (CFL 12/12: 1.66/1.66/1.85, 10/10:
1.97/1.91/1.91).

| arm | 2SC table | x | CFL table | x | **CFL converged solve** | x |
|---|---|---|---|---|---|---|
| 24/24 | 67.8 | 1.00 | 775.5 | 1.00 | 88.3 | 1.00 |
| 16/16 | 52.8 | 1.28 | 599.5 | 1.29 | 64.5 | 1.37 |
| **12/12** | 44.7 | **1.52** | 511.3 | 1.52 | 53.0 | **1.67** |
| 10/10 | 40.7 | 1.67 | 453.0 | 1.71 | 46.1 | 1.91 |
| 8/8 | 37.7 | 1.80 | 427.6 | 1.81 | 40.8 | 2.16 |
| 24/12 (vacuum only) | 55.2 | 1.23 | 659.6 | 1.18 | 72.1 | 1.22 |
| 24/8 (vacuum only) | 49.6 | 1.37 | 619.8 | 1.25 | 67.8 | 1.30 |

Windows: 2SC at loadavg 25-33, cpu/wall 0.64-0.83; CFL at 19-164,
cpu/wall 0.59-0.94. The reference backend, one loaded run each: 2SC
1.23/1.56/1.68x at 16/12/10 and 1.13x vacuum-only; CFL 1.66/2.05x at 12/10
and 1.35x vacuum-only.

**Against ticket 03's 80 ms/pt converged: ~48 ms/pt at 12/12, ~42 at 10/10,
~37 at 8/8.** The map's 1 ms/pt destination is still 40-50x away, and the
single-pattern CFL table is still 89.5% dead points below the branch.

### 6. Where the time goes now: ticket 03's attribution at each rule

% of the whole table (2SC: every point converges, so this is also the
converged split):

| | 2SC 24/24 | 12/12 | 8/8 | 24/8 | CFL 24/24 | 12/12 | 8/8 | 24/8 |
|---|---|---|---|---|---|---|---|---|
| hot pass | 17.9 | 13.4 | 10.2 | 22.0 | 33.5 | 26.0 | 21.1 | 43.0 |
| vacuum passes | 18.3 | 12.2 | 10.4 | 6.8 | 24.7 | 18.8 | 15.4 | 10.6 |
| hot Hessian | 9.2 | 7.9 | 5.3 | 11.6 | 4.4 | 3.2 | 2.6 | 5.6 |
| vacuum Hessian | 12.2 | 8.6 | 7.4 | 4.8 | 4.2 | 3.1 | 2.6 | 1.8 |
| **`gapless_momenta`** | 12.1 | 16.7 | **20.0** | 15.9 | 22.8 | 35.2 | **42.8** | 28.2 |
| unpaired references, panel builds | 17.2 | 22.9 | 25.4 | 20.4 | 5.9 | 7.6 | 8.3 | 6.0 |
| jitted / numpy / Python | 57.9 / 29.3 / 12.8 | 42.5 / 39.6 / 17.9 | 33.7 / 45.4 / 20.9 | 45.6 / 36.3 / 18.2 | 66.9 / 28.8 / 4.3 | 51.3 / 42.8 / 5.9 | 41.8 / 51.2 / 7.0 | 61.1 / 34.3 / 4.6 |

And over the 188 CONVERGED CFL solves alone, ticket 03's second column
(`t10_attr_CFL_conv.json`, the session's quietest window: loadavg 18-28,
cpu/wall 0.98; the wrappers cost a few percent):

| % of the converged solves | 24/24 | 16/16 | 12/12 | 10/10 | 8/8 | 24/12 | 24/8 |
|---|---|---|---|---|---|---|---|
| hot pass | 25.2 | 22.7 | 21.1 | 20.0 | 18.3 | 30.6 | 32.7 |
| vacuum passes | 14.1 | 12.7 | 11.8 | 11.1 | 10.2 | 8.6 | 6.2 |
| hot Hessian | 19.0 | 17.3 | 16.1 | 15.1 | 13.8 | 23.0 | 24.7 |
| **vacuum Hessian** | **19.2** | 17.5 | 16.2 | 15.2 | 14.0 | 11.8 | 8.5 |
| `gapless_momenta` | 10.3 | 14.4 | 17.0 | 19.1 | 21.8 | 12.3 | 13.4 |
| unpaired references, panel builds | 7.7 | 9.7 | 10.9 | 11.9 | 13.3 | 8.6 | 9.0 |
| jitted / numpy / Python | **77.6** / 18.1 / 4.3 | 70.2 / 24.0 / 5.8 | 65.3 / 28.0 / 6.7 | 61.5 / 31.0 / 7.5 | 56.4 / 35.1 / 8.4 | 74.1 / 20.9 / 5.0 | 72.2 / 22.5 / 5.3 |
| ms per converged solve, wrapped | 86.4 | 63.8 | 50.7 | 44.7 | 39.3 | 70.5 | 66.5 |

At 24/24 this reproduces ticket 03 (77.6% jitted against 77.3%, 4.3%
Python against 4.6%, vacuum passes 14.1% against 14.0%), and it separates
what ticket 03 did not: **the vacuum HESSIAN is another 19.2% of a converged
solve**, because `_vacuum_pair_hessian` hits its cache 5% of the time (one
call per Jacobian, and (M, Delta) move every step), against 35% for the
vacuum block. So the vacuum half is **37.7% of a converged CFL solve**, and
still 34% at 12/12.

**Over whole tables the vacuum half is 41% of 2SC and 32% of CFL at the
shipped rule** (passes, Hessians, their unpaired references and panel
builds). Its cache hit rate does not move with the rule (26% 2SC, 35% CFL).
**On the whole table, once the quadrature is cut, `gapless_momenta` is
the wall**: 48 scan
momenta, refinement and Brent per call, whatever the node count, 43% of the
CFL table at 8/8, in a pattern with no gapless state anywhere on the grid.

### 7. The baseline (`t10_baseline_precheck.log`)

The four paired points `test/baseline` freezes for njl (`Parameters.default()`,
reference backend), flattened by `generate_baseline.row` and compared as the
test compares them:

- 24/24 reproduces `njl.npz` exactly: 0 of 139 keys (the control);
- **vacuum-only 12: 9 of 139 outside 1e-10.** Six sit at the solver's own
  resolution: mu_3 ~= 0 in 2SC (stored -3.6e-9 MeV), and CFL's mu_C =
  -mu_3 = 0.51 MeV taken separately (their sum is 2.9e-11, and mu_8 and `x`
  follow). Three are physical, at 1.1-2.9e-10 (2SC n_B = 1.4887: M, mu_e,
  mu_C);
- 12/12: 32 of 139, the T = 20 point carrying s, Y_C and n_e at ~1e-6
  (section 4); at T = 0 the physical keys move <= 2.4e-10.

So any default change regenerates `njl.npz` in its own commit with the deltas
quoted (test_baseline's rule 2). The solver-resolution keys are a hygiene
finding of their own, the same class `row()` already drops for mu_S: they
would fail under any change to the Newton path.

### Verdict

- **What the RG vacuum half needs.** Not the hot pass's node count: 12 per
  panel on its shipped layout reproduces every delivered quantity to the
  solver floor, table-verified at T = 0 on both backends and point-verified at
  T = 20 and 30. 10 does too (M at 8e-10 at T = 20). 8 leaves M at 2e-8 in
  2SC, and 8 plus breakpoints at the three masses reaches the floor in
  isolation (88 nodes against 192, not table-verified). Nor does it need to
  SHARE the hot pass's nodes: the decoupled rules (24/12, 24/8) take the same
  Newton steps and deliver the same rows. So whatever the nested form buys
  over the shell form, it is not node coincidence in the tail. It does need
  the residual and the Jacobian on the same vacuum rule, which is what the
  harness enforces.
- **What the hot pass needs.** 10-12 at T = 0, where the kinks sit on panel
  edges. At T > 0, >= 16 for P and ~20 for s at 1e-10: the shipped 24 is the
  finite-T rule.
- **The shared default stands.** `NODES_PER_PANEL` is
  `eos/general/fermi_integrals.py`'s, and every model's unpaired Fermi
  integrals use it as well as the pairing block. ccdm's pairing (no RG split,
  finite T) was not measured, and at finite T a coarser rule is wrong here.
- **Landable, as its own ticket:** (a) the vacuum passes at 12, the
  T-agnostic cut, **1.22x** on a converged CFL solve and 1.23x on the 2SC
  table; (b) a T = 0 hot rule of 12, T-aware inside `rg_pair_block` or per
  call through the existing `pair_nodes_per_panel`. Together that is 12/12,
  **1.67x**. Both regenerate `njl.npz`. Neither is done here.
- **What is left.** On a converged solve the jitted quadrature is still 65%
  at 12/12: the passes are cheaper, not fewer, and the vacuum half (passes
  and Hessian) is still 34%. The node rule stops paying below ~10 because
  `gapless_momenta` does not scale with it. It is 17% of a converged solve
  and 35% of the whole CFL table at 12/12 (the dead points have more
  crossings to hunt), scanning 48 momenta for crossings a gapped state does
  not have. Fewer passes, meaning a vacuum Hessian that does not miss its
  cache 95% of the time, and a cheaper no-crossing verdict are the next two
  places to look.
