# Does the `methods` bound pay inside `eos/mixed`?

Type: prototype
Status: closed
Blocked by: 16
Parent: ../map.md

## Question

[Bound the rescue ladder below a branch](13-bound-the-rescue-ladder.md)
measured `lm` at 29.2% of a 200-point njl build, entered by exactly the 13
candidates with no root and rescuing none of them, and
[16](16-land-the-bounded-ladder.md) gives `solve_system` a `methods` argument
to decline it. **Does the same rung cost the same nothing inside the mixed
loop, and can the same argument decline it there?**

### Why it is not the same question

`njl_phase.thermo` is a different code path and ticket 13 does not reach it:

- it calls `eos.njl.thermodynamics.thermo_from_mu`, not `solve_pattern`, so
  `attempt`'s ladder -- the Newton rung, rung C, rung D, the cold retry -- is
  not in it at all. What it runs is `solve_system(..., tol=1e-13)` with **no
  Jacobian**: `hybr`, then `lm`, then the polish;
- its candidates are seeded from their OWN previous root or from cold, never
  cross-seeded, so the discriminator ticket 13 found -- *where did this seed
  come from* -- does not exist here. A pattern with no root at those
  potentials looks exactly like one that simply has not been reached yet.

So the finding transfers but the rule does not, and the rule is the part that
has to be re-derived.

### What to measure first

The counts, not the clock -- [ticket 04](04-count-the-mixed-loop.md)'s
instrument already prices one hybrid row at ~41 `thermo` calls and ~123 NJL
internal solves. Of those solves, how many enter `lm`, what does it cost, and
how many does it rescue? If the answer matches ticket 13's (entered only by
rootless candidates, rescues none), the bound is `methods=('hybr',)` on the
adapter's call and the question becomes whether anything downstream of a
declined `lm` changes.

**Trap ticket 04 already left here:** the mixed cost is **flat in count and 2x
in per-call cost across the window**, expensive at the QUARK end rather than
at the onset. A rootless-candidate story predicts the opposite shape, so if
the `lm` share does not concentrate where the cost does, this is not where the
mixed time is going and the ticket should say so rather than shave it.

### Gate

- The mixed gate, not njl's: the located `window` (n_onset, n_offset), chi,
  and the per-phase charge decomposition unchanged; P to 1e-8.
- Counts before and after from ticket 04's instrument, wall quoted beside cpu
  with the machine's load stated.

### Handed in from ticket 16 (2026-09-21): the prior is now AGAINST

The path this ticket would bound has **no Jacobian**, and ticket 16 measured
the one Jacobian-free path this effort has tried the bound on. There, `lm` was
not dead weight: it was what found the CFL branch where it begins. On njl's
reference backend, bounding the cross-seeded candidates to `hybrd` alone lost
**30 of 30 CFL rows** over the pinned grid's first 60 densities. `hybrd`
stalled at 1e-6 at the branch's first density, 0.5686, then the sweep
captured a second CFL root about 1 MeV/fm^3 higher in f, and the onset was
never found. That is why `50b3b7f` bounds only where there is an analytic
Jacobian. This ticket's system is different (fixed potentials, own seeds), so
the result does not transfer as a verdict. It does move the burden: "`lm`
rescues none of them" has to be MEASURED here, per candidate, and not
inherited from ticket 13.

---

## Resolution, 2026-09-22

**Yes, and exactly where the build throws its work away -- not where the
rows spend it.** Inside the mixed loop `lm` is **1.7-4.2% of the NJL work in a
table row and 18-24% of it in `locate_window`**, entered almost only at trial
potentials of mixed solves that fail in both arms (probes below the onset,
the walk's failing step, the failed exact onset). Declined, **the window is
bit-identical and fifteen mixed rows are bit-identical in every field on both
backends**; the hinted locator's NJL work falls **24.5%**. **Landed in
`66d713c`**, in the adapter, for the enumeration only: a held pattern keeps
the full ladder.

### The discriminator, re-derived

Not ticket 13's seed origin -- here every candidate is seeded the same way,
from `MixedCtx.phase_seed`, one constant per mixed solve. **It is whether the
candidate has rivals.** Enumerating, a candidate that misses the gate is
dropped and the max-P rival is the block, so a failure is an ordinary
answer. Held to one pattern the candidate IS the phase, a failure is a raise,
and nothing measured here licenses taking `lm` away from it. `njl_phase`
already draws that line for its seed (`branch_declared`); the bound uses the
same flag.

### How it was measured -- counts, and an exact counterfactual

`t17_count_lm.py`, layered on ticket 04's `count_mixed.py` from outside
(nothing in `eos/` edited for the census): every `internal_residual`
evaluation is attributed to the rung it was spent in, and the per-evaluation
cost is the pattern's quadrature, so evaluation shares are cost shares.
**The counterfactual is exact, not modelled**: with `methods=('hybr',)` a
call runs the identical `hybr` and then polishes from ITS iterate, and since
the seed is constant within a mixed solve there is no path dependence inside
a call -- so for every candidate that entered `lm`, the real `newton_polish`
was re-run from `hybr`'s iterate and its outcome, layout and P recorded.
The instrument reproduces ticket 04 to the call (cold row 47 residuals / 147
NJL solves; deep rows 289 / 912; hinted locator 1825 / 5700).

python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0, ticket 04's configuration:
DID + NJL `rg_njl1`, csc, patterns `("unpaired", "2SC", "CFL")`,
`beta_eq_neutrinoless`, T = 0, eta = 0, grid `linspace(0.08, 1.60, 200)`,
backend `fast` for the census, HEAD `63a6dfa`.

| where | NJL evals | `lm` share | entered `lm` | bounded call differs |
|---|---|---|---|---|
| one cold row, n_B = 1.000 | 6,264 | 4.2% | 1 (2SC) | 0 |
| 7 onset rows, 0.859-0.905 | 32,853 | 2.3% | 7 (2SC) | 0 |
| 7 deep rows, 1.004-1.050 | 35,484 | 1.7% | 2 (2SC) | 0 |
| `locate_window`, hinted | 350,866 | **23.7%** | 316 | **1** |
| `locate_window`, unhinted | 564,051 | **18.4%** | 371 | **3** |

In every row entry `lm` failed too and the polish converged the candidate
from `hybr`'s own iterate, so the bounded call returns it bit for bit. In the
locator, `lm` is 30-32% of the scan's evaluations, 21-38% of the exact
refines', 18% of the walk's retry ladder and **0% of `bisect`'s**; nearly all
of it is `fail(lm)` -- a CFL candidate at trial potentials where it has no
root, grinding the 400-evaluation cap and then the polish.

**The ticket's own trap, answered: the `lm` share does NOT concentrate where
the row cost does** (1.7% deep, where ticket 04 put the expensive rows, 2.3% at
the onset). So `lm` is not where the ROW time goes and nothing here shaves the
rows. It is where the LOCATOR's doomed solves go, and the locator is 76% of a
build, which is what made it worth landing.

**The four candidates the bound loses**, all at trial points of mixed solves
that fail in BOTH arms: hinted, one CFL candidate inside the walk's failing
step at 0.851453 (block CFL P 285.26 -> 2SC 259.69 at that trial point);
unhinted, three inside doomed scan probes (the block at one moves CFL
0.124 -> unpaired ~0, at another 2SC -> a CFL collapsed onto the same state,
dP/P 6e-9). None reaches a result: the solves they sit in fail either way.
**The same CFL point converges with `hybr` alone from a cold start**, on both
backends -- the `lm` rescue was a property of the per-solve seed, not of the
point, so it does not make a regression test for the held-pattern carve-out
either. That carve-out is argued (the candidate IS the phase), not measured.

### Gate -- PASS on counts; the clock half is not quotable

- **The window, hinted**: `0.8533622658748288 -> 1.2450284086872276`,
  bit-identical, `lm` declined against HEAD; every converging solve takes the
  SAME number of mixed residuals (scan 321, walk 59, bisect 175, offset
  refine 60). Only the doomed solves move, and they still fail: scan probes
  84,367 -> 45,204 NJL evals, the failed onset refine 39,012 -> 21,519, the
  walk's failing step 12,702 -> 8,181.
- **Counts before and after, ticket 04's instrument, hinted**: 1825 -> 1807
  mixed residuals, 5700 -> 5646 NJL solves, **350,866 -> 264,807 NJL evals**.
- **The rows, P to 1e-8 and more**: the cold row and ticket 04's 7 onset and
  7 deep rows, HEAD against `lm` declined, **15 of 15 bit-identical in every
  field** -- P, eps, chi, the realised CFL pattern, every potential, and each
  phase's (n_B, n_C, n_S, P) (`t17_gate.py`, `t17_gate_head.json` /
  `t17_gate_nolm.json`). **Ticket 16's trap checked on BOTH backends**: on
  `backend="reference"` (njl_phase's default), HEAD from a `63a6dfa` worktree
  against the landed tree, 15 of 15 bit-identical again, while the two
  backends differ from each other by 6.9e-10 in P -- so the reference arm is
  a genuinely different numerical path, not the same one twice.
- **The unhinted locator on the landed tree**: see ticket 11, which lands in
  the same commit and gates both.
- **Tests on the landed tree**, python.org 3.14.2 / numpy 2.3.5 / scipy
  1.17.0, one directory at a time, `eos/*.py` fingerprinted either side
  (`ea9e8a346f28ee58` both): `test/njl` **132**, `test/mixed` **272**,
  `test/baseline` **20**, `test/test_imports.py` **221**, all passed. Those
  are the reachable suites (the committed tree differs from the tested one
  by a comment-only rewording of the adapter's measurement note, after which
  `test/test_imports.py` was re-run, 221 passed): `thermo_from_mu`'s new
  argument defaults to the
  old ladder for every other caller (its only other one is
  `njl/verify/run_full_check.py`), the adapter is reached only through
  `eos/mixed`, and baseline pins every model at rtol = 1e-10.
- **Wall beside cpu: NOT quotable, and not quoted.** Two BayEoS
  multiprocessing pools (~20 workers) and a video call held loadavg at
  70-430 for the whole session, the laptop ran on battery until it died and
  slept overnight, and cpu/wall ran 0.4-0.6. Indicative only, hinted locator
  cpu 3917 s at HEAD against 2393 s with `lm` declined. A quiet AC window is
  still owed for a clock number.

### What landed

`eos/njl/thermodynamics.py`: `thermo_from_mu(..., methods=("hybr", "lm"))`,
passed to `solve_system`; the default is the old ladder.
`eos/mixed/adapters.py`: `njl_phase` computes `branch_declared` once, up
front, and hands `thermo_from_mu` `("hybr",)` when enumerating and the full
ladder when held; the comment carries the measurement.

### Artifacts (in `.scratch/njl-speed/`)

`t17_count_lm.py` (the census), `t11_trace_locator.py` (the per-solve trace
it feeds, with `T11_VARIANT=nolm` for the patched arm), `t17_gate.py` and its
`t17_gate_*.json` / `.log`, `t17_cold.log`, `t17_onset.log`, `t17_warm.log`,
`t11_hinted_{head,nolm,landed}.log`, `t11_unhinted_{head,landed}.log`.
