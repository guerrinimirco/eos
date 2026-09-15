# Generate the finite-T training set, and measure what it costs BEFORE running it

Type: task
Status: open
Blocked by: --
Parent: ../map.md

## Question

The model is **T = 0 and nothing else**, and it is T = 0 in a way that is easy
to miss: `aT2` and `aT4` are defined in `basis()`
(`notebooks/csc_bag_map.py:168-169`) and have entries in `TERM_BOUNDS`, but they
are **NOT in `TERMS`** (`csc_bag_map.py:709-710`). They are dead basis entries.
`GRID_T = (0.0,)` (`:266`), so even if they were in the term set their columns
would be identically zero and resolved only by the ridge.

The fourth acceptance benchmark -- T = 0-100 MeV, `Y_C` = 0.01-0.5, `mu_S` = 0,
the one that serves merger and supernova tables -- cannot be attempted until
this data exists. This ticket is on the frontier with ticket 01 and should
START EARLY, because it is the only expensive thing on the map and it blocks
ticket 05 alone.

### The grid

Extend the shipped one (`csc_bag_map.py:263-266`) on the temperature axis only:

| axis | value | why unchanged |
|---|---|---|
| `mu_B` | 900-1800 MeV, 28 points | **Measured: no branch reaches 1 n_sat.** Swept in 50 MeV steps on four gluon-exchange sets, every branch ends at its own terminus, 1.5-2.4 n_sat at mu_B = 850-1150 MeV, below which the T = 0 quark phase does not exist (the chiral condensate is still there). The reachable window is ~2-13 n_sat. |
| `mu_C` | {-80, -40, 0, 40} MeV | Widening it is safe only with `scan_root`-style bracketing, not a fixed one -- see the trap below. |
| `mu_S` | {-80, 0, 80} MeV | |
| `T` | **(0, 10, 25, 50, 75, 100) MeV** | the new axis. 0 exactly, because that is where every published comparison lives and a uniform 4-D sample would place almost no points there. |
| CFL | the `mu_B` line only | CFL is 1-D; see the trap below. |

That is 28 x 4 x 3 x 6 = **2016 points per unpaired/2SC pattern** and
**168 for CFL**, per NJL parameter set.

**Use `vector_form="gluon_exchange"` for the training set -- but the reason is
NOT the one the prior art gives.** `notebooks/csc_dataset.py:107-113` builds its
25-set grid with `constant` and justifies it -- *"that is the form the published
RG-NJL sets (rg_njl1/Kunkel, eta_D = 1.45, eta_V = 0.7) are quoted in"* -- and
`docs/csc_bag_mapping.md:391-403` contradicts it with the Appendix A.1 table:
unpaired rms in P of 2.1% / 9.8% / 13.8% at eta_V = 0 / 0.5 / 1 with `constant`,
against 3.8% for `gluon_exchange`.

**That table no longer says what it is cited for.** It was measured on the BAG
SUB-BASIS -- Appendix A's setup bounds `a4`, `B` and `m_s` and nothing else, so
the basis was `(mu^4, mu^2, const)` -- and its stated failure mode is that such
a basis "absorbs `~G_V n_q^2 ~ mu^6` only by leaving the physical window". That
is a complaint about a MISSING TERM, and the shipped basis now has it: `a6` is
in `TERMS` (`csc_bag_map.py:833`). The table is also on the ABANDONED
`mu_B` = 1500-2600 MeV grid, which this map calls excellent and useless, and its
best row (2.1% at eta_V = 0) is the degenerate no-vector case, which shows that
NO vector sector is easy rather than that a constant one is mappable. Three
reasons the number is stale; do not lean on it.

**The argument that does survive is the shape of `a6`.**

    a6              sum_f mu_f^6/(mu_f^2 + M_V^2)         -> mu^4 above M_V
    gluon_exchange  G_V = G_V0/[1 + 8 k_F^2/(9 M_g^2)]    -> G_V n_q^2 ~ mu^4
    constant        G_V = eta_V G_S                       -> G_V n_q^2 ~ mu^6

`a6` IS the gluon-exchange form written as a basis function: same saturation,
same asymptote, and `csc_bag_map.py:48-58` says so where the term is declared.
A constant `G_V` is the one member of the family that never saturates, so `a6`
can track it below `M_V` and must miss it above. And it collides with ticket 02
head-on: the basis is built so that `P/P_free -> a4 + a6` and `c_s^2 -> 1/3`,
while a constant-`G_V` NJL runs to `c_s^2 -> 1/5`. **The conformal limit cannot
be imposed on a target that violates it.** That argument is structural and does
not depend on a grid. Secondary: the Bayesian axis is `G_V0_over_GS`, which only
`gluon_exchange` reads, while `eta_V` is read only by `constant`
(`eos/njl/parameters.py:71,79`) -- mapping both is two surfaces, not one.

**So measure it before writing it off, because it is cheap.** The
reproducibility cost is real -- Kunkel's published set is quoted in the constant
form -- and unpaired is the 10 ms/pt pattern, so the check is minutes, not hours:

> Before the 15-25 set run, refit the UNPAIRED pattern with the CURRENT `TERMS`
> (`a6` included) on the SHIPPED 900-1800 MeV grid at `vector_form="constant"`,
> `eta_V` = 0, 0.5, 1, and report rms in P together with whether `B` or `m_s^2`
> pegs at its bound.

If it lands within target, `constant` becomes a SECOND SURFACE AXIS and A.1 is
superseded in `docs/csc_bag_mapping.md` rather than cited. If it does not, the
`docs/DEFERRED.md` entry saying the published `constant`-form sets are not
reproducible by this model is written WITH A CURRENT NUMBER behind it instead of
a superseded one. Either way the training set itself is `gluon_exchange`; this
check does not block it and can run alongside.

### The generator, and the five traps it must keep handling

The entry point is the phase adapter, potentials in / thermo out, **one pattern
DECLARED**, so a fit never sees a first-order transition
(`csc_bag_map.py:285-296`, contract at `eos/mixed/adapters.py:1144-1388`):

```python
phase = njl_phase(par, njl.SpeciesFlags(csc=True), patterns=(pattern,),
                  backend="fast")
th = phase.thermo(mu_B, mu_C, mu_S, T)
```

1. **CFL is one-dimensional whatever the grid does.** Locking gives `n_C = 0`
   and `n_S = n_B` identically, so P depends only on `mu_B + mu_S` -- measured
   identical to every digit at (1900,0,0), (1900,-60,0), (1800,0,100),
   (1840,0,60). Sample the line, not the box; the full grid there is one line
   sampled many times over.
2. **A nonzero gap does NOT identify the CFL phase.** Measured at `mu_B` = 967
   and 1000 MeV, a CFL-layout solve near the terminus converged on an UNLOCKED
   state: `n_C = 0.21 fm^-3`, `n_S = 0` -- no strange quarks at all -- while
   carrying a 138 MeV gap. Filter on the LOCKING
   (`|n_C| < 1e-6 n_B` and `|n_S/n_B - 1| < 1e-6`), not on the gaps. Dropping
   those 2 of 24 points improved the CFL fit by a factor of **FORTY** (rms in
   P 9.9e-3 -> 2.4e-4, in n_B 2.0e-2 -> 1.5e-3).
3. **Layout collapse.** Require `max|Delta_i| > 1 MeV` for a paired pattern, or
   the candidate has fallen onto a rival's root.
4. **Read `pattern_realised`, never `pattern`.** The solved-layout column is
   not stable: `.scratch/njl-speed/issues/05-cheap-pre-screen.md:154-159`
   measured `free` winning the `min(f)` tie at 40 densities where the state was
   CFL. And `realised_pattern` can return names outside the enumeration (uSC,
   dSC, usSC) when a CFL-layout solve collapses -- never index a dict on it.
5. **Sweep DOWNWARD in `mu_B`, warm started, per (mu_C, mu_S, T) line.** The
   shipped sampler iterates `itertools.product` with no `x0` at all
   (`csc_bag_map.py:292-296`), which is both slow and exposed to the root
   stitch: *"an ASCENDING CFL sweep stitches two roots together and comes back
   with mu_B(n_B) non-monotone"* (`notebooks/quark_timing.py:885-890`, and the
   library's own `branch_ladder` at `eos/njl/table.py:386-412` walks DOWN for
   this reason -- "every pattern exists at the top of a density range and the
   paired ones end somewhere below"). Warm starting is also most of the speed.

**Plus one the ledger warns about and the filters above do not catch.**
`docs/DEFERRED.md` records that a `fixed_Y_C = 0.1` table once reported
metastable 2SC at 98 of 100 points because the gapless CFL that minimised `f`
could not reach the gate, and closes with: *"`converged = True` on every row of
a table is therefore not evidence the table is the ground state."* Here each
pattern is declared separately so there is no ranking to get wrong -- but a
gapless state landing in the training data as an ordinary CFL row IS a
problem, because the ansatz has no gapless branch. **Record `th.fields`'
gapless flag per row** so ticket 07 can ask how many there are; do not filter
on it yet.

### The cost, which is the actual question

The only projection in the repo is `notebooks/csc_dataset.py:228-239`, using
`COST_PER_SOLVE = {"unpaired": 0.01, "2SC": 0.15, "CFL": 0.60}` seconds
(`:116`): 25 sets x 3 patterns x 1024 Sobol points = 76,800 solves, **5.4
CPU-hours**, 0.7 h on 8 shards. Those constants are fixed-`n_B` costs; a
fixed-mu internal solve closes a smaller system and should be cheaper, and
warm starting should cut it again.

**Do not run the full set on an unmeasured constant.** Time ONE parameter set,
all three patterns, the full T grid, and report the measured per-point and
per-set cost first. Then size the (eta_D, G_V0) map from it -- a quadratic
surface in two variables needs 6 points and the prototype used 15
(`MAP_ETA_D = (0.75, 1.00, 1.25, 1.45, 1.65)` x
`MAP_G_V0 = (0.25, 0.50, 0.75)`), so 15-25 sets is the range.

Cache as the prototype does: md5 over `(asdict(par), pattern, grids)`, pickled
under `output/csc_map_cache/` (`csc_bag_map.py:271-283`), so a re-run is free
and a partial run resumes.

## Gate

- **A measured wall-clock per parameter set**, all three patterns, full T grid,
  stated with n, cpu beside wall, and the interpreter's numpy and scipy
  versions -- BEFORE the full sweep is launched.
- The full dataset cached, with a per-pattern row count and the count of points
  DROPPED by each filter (unlocked CFL, collapsed layout, non-convergence)
  reported separately -- a filter that drops nothing and a filter that drops
  half both deserve a look.
- **Zero unlocked rows in the CFL set**, asserted, not assumed.
- The gapless flag carried on every row, for ticket 07.
- `vector_form` is `gluon_exchange` throughout the training set, and the
  `constant`-form contradiction in `csc_dataset.py:107-113` is resolved in
  writing -- against a re-measured number, not against Appendix A.1.
- **The `constant`-form refit measured**: unpaired, current `TERMS`, shipped
  900-1800 MeV grid, eta_V = 0 / 0.5 / 1, rms in P and which bounds peg.
  Outcome recorded either as a second surface axis or as a `docs/DEFERRED.md`
  entry, and A.1 marked superseded in `docs/csc_bag_mapping.md` either way.
- A statement of how many points the descending warm start saved against the
  shipped cold `itertools.product`, which is the cheapest possible check that
  the sweep direction was actually applied.
