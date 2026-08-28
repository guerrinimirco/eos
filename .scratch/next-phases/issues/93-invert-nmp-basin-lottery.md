# `invert_nmp`'s verdict is decided in the target's last bits

Type: task
Status: resolved (2026-08-28)
Blocked by: -
Parent: ../map.md

## Question

Split out of [ticket 67](67-dd2-t0-adoption.md), which found this while
measuring something else and could not fix it from where it stood.
[Ticket 47](47-dd2-nmp-inversion.md) found the same floor from the other side:
47 is the STENCIL noise in the forward map, this is the SOLVER's basin
selection in the inverse map, and the fix candidates are different.

### What was measured

The 5x5 default closure at DD2's OWN nuclear-matter parameters, target scaled
by (1 + eps), no code change, python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0:

    eps      isoscalar residual        couplings
    0        6.7e-11                   moved 3.9%
    1e-14    2.2007e-3                 moved 0.0 -- the seed, bit for bit
    1e-13    2.2007e-3                 moved 0.0
    1e-12    2.3e-08                   moved 3.9%
    1e-10    2.0e-3 (partial)          moved 0.27%

Three findings, none of which is about any one refactor:

1. **`root(method="hybr")` returns the published seed unmoved** on a target
   perturbed in its fourteenth digit, and `InversionStatus.ok` is True when it
   does, because ISO_GATE = 2e-2 admits the seed's own 2.2007e-3 cross-row
   violation. Not new for the 6x6 closure: `nmp.py:70-78` documents the failure
   mode, `test_api.py` documents hybr stalling with status 5 at DD2's own NMPs,
   and that test already carried the seed comparison
   (`assert inv.b_sigma != par.b_sigma`) this ticket would otherwise propose.
   **What is new is that it reaches the 5x5 closure**, which the same docstring
   describes as converging to 6.7e-11 without qualification -- it does, at
   eps = 0, and returns the seed at eps = 1e-14.

2. **`test_inversion_is_deterministic` gives false comfort.** Two stuck runs
   agree perfectly, so the test passes hardest exactly when the solver has done
   nothing.

3. **The converged branch is not injective at DD2's own point.** Where the
   solve does move, it lands on couplings 3.9% from the published set that
   reproduce the same six NMPs. "Converged" and "recovered DD2" are different
   statements and the return value does not separate them.

And across the (K_sat, Q_sat) plane, the 6x6 closure's 32-restart landing
residual is chaotic in the target's last bits: eight targets over
K_sat in {210, 220, 230, 240} and Q_sat in {300, 350}, each at
eps in {0, 1e-13, 1e-11}, and NOT ONE holds its verdict across its own three
-- residuals scattering 3e-5 to 0.15 with no relation to eps.

### What to decide

1. **Does `ok` learn to say "never moved"?** The cheap version is the
   comparison `nmp.py:70-78` already tells the reader to make by hand:
   `InversionStatus` carries whether the recovered couplings differ from the
   seed, and `ok` is False when they do not. That converts a silent wrong
   answer into a reported failure, which is what section 6 requires of a public
   boundary.
2. **Is ISO_GATE = 2e-2 defensible** when the seed's own cross-row violation is
   2.2e-3, an eighth of it? A gate that admits the starting point cannot
   certify an answer.
3. **Restart coverage.** `test_dd2_m8` gave up asserting on 32 restarts at a
   single target (ticket 67, decision 3) because that verdict is a coin flip.
   The property worth testing is the one `nmp.py:84-96` already records --
   7/68/115 of 187 cells inverting at 0/32/64 restarts -- a count over a grid,
   stable where any single cell is not. Slow; decide whether it is a test or a
   `verify/` entry.
4. **Does the same lottery reach `from_nmp` and the scan callers**, or is it
   confined to targets sitting on their own seed? Partly answered already: it
   reached `test/tov/test_solver_fast_robustness.py`, whose three cases needed
   a soft Delta-rich parametrization and obtained it through
   `build_parametrization` on a hand-tuned NMP target. Ticket 74 had already
   re-tuned that sample once for this reason; ticket 67 froze the resulting
   parametrization as data instead, which removes the prerequisite but does
   not fix the map.

5. **Dropping Q_sat is NOT the workaround it looks like.** The 5x5 keeps the
   system square by PINNING c_omega, so excluding Q_sat constrains the
   isoscalar sector more than imposing it does -- the 6x6 solves for c_omega as
   a sixth unknown. Measured at (K_sat = 220, L_sym = 30) with DD2's other
   NMPs: the 6x6 converges (Q_sat 299.65) where the 5x5 fails at 5.9e-2. Any
   proposal to retire the 6x6 has to answer what reaches those targets
   instead.

---

## Resolution (2026-08-28)

**The defect was not the verdict. It was that the stall SUPPRESSED THE
RESTARTS that would have solved the problem**, and the verdict merely failed to
notice. `_restart_loop` ran on `best_res >= ISO_GATE`; a stall carries the
published couplings' own 2.201e-03 cross-row violation, which sits UNDER the
2e-2 gate, so the 32 restarts never fired. They were never needed to be many:
at DD2's own nuclear-matter parameters the **FIRST** jittered restart drives the
5x5 to 6.8e-08 and recovers K_sat to 1e-4 MeV. So the same condition that makes
`ok` honest, fed into the restart trigger, converts the silent wrong answer into
a **correct** answer rather than into a reported failure — more than §6 asked
for.

All numbers python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0.

### 1. `ok` learns to say "never moved" — YES, and it drives the restarts too

`InversionStatus.coupling_shift` (max relative distance from the seed) is
reported, and `_stalled(x, seed, res)` — the seed returned **bit for bit** with
a residual above `STALL_RES = 1e-5` — is read in both places `ISO_GATE` was
read. Bit-for-bit rather than a tolerance because the stall is hybr giving up on
its FIRST step (`status=5`, "not making good progress", 23 evaluations), not a
small final move.

The residual guard is not decoration: re-seeding a solve at its own answer
returns `shift = 0.0` with `res = 1.8e-08` and `success=True`. An unmoved seed
is legitimate when the seed WAS the root. Five orders separate that from the
2.2e-03 stall; `STALL_RES` sits in the middle.

**The 5x5 lottery is gone.** At DD2's own NMPs, target scaled by (1 + eps) for
eps in {0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8} — all seven converge,
residual 2.4e-10 .. 9.1e-08, `coupling_shift` **3.948e-02 in all seven**, K_sat
recovered to 1e-4 MeV. Before: four of those seven returned the seed with
`ok=True`.

### 2. ISO_GATE = 2e-2 — the premise is right, the remedy is not. Gate unchanged.

**It cannot be tightened to catch the stall.** Over K_sat = 200–300 at three
perturbations each, a **moved and ACCURATE** 5x5 solve landed at
max|residual| = **1.944e-03** with K_sat recovered to 0.0095 MeV — against the
stall's 2.201e-03. Stalled and converged residuals **overlap**; no threshold on
the residual alone separates them at any value. The ticket's sentence "a gate
that admits the starting point cannot certify an answer" is correct, and what
follows from it is that the certificate stops being a gate reading — not that
the gate needs a smaller number.

Also tried and rejected: `root()`'s own `success` flag, which DOES report the
stall (status 5). It also reports status 5 at K_sat = 200 and 260, which land at
8.5e-08 and 2.6e-08 with K_sat recovered to 1e-4 MeV. `success` rejects good
answers; the coupling shift does not.

**What the gate still admits, quantified.** Iterating the 5x5 round trip eight
times, iteration 5 lands off-branch at res 5.13e-03 (b_sigma 0.671 against the
0.6094 branch) and `ok=True`. Iterations 6–8 return to the branch but carry a
permanent 2.12e-02 MeV K_sat drift. So a residual lottery remains, its cost is
0.02 MeV of K_sat out of 243 — 1e-4 relative — and that is what a 2e-2 gate buys.

### 3. Restart coverage — a `verify/` entry, not a test

`eos/dd2/verify/run_full_check.py::_check_restarts_extend_the_basin`. Nine cells
over K_sat in {200, 240, 280} x Q_sat in {0, 200, 400}, 0 vs 32 restarts:
**0/9 -> 4/9**, ~10 s. `verify/` because 10 s is real cost in a suite that runs
on every commit touching solver internals while `verify/` runs on demand, and
because sfho already carries its forward/inverse agreement check there.

NOT the 187-cell 7/68/115 counts: those are the number to quote and they stay in
the module docstring; nine cells is what is affordable to re-run. And what is
asserted is that the restarts **change the answer** — the loop keeping the best
of N is monotone by construction and asserting monotonicity would assert the
loop's own logic.

### 4. It reaches `from_nmp` and the scan callers — measured, and fixed at one point

Before the fix, `from_nmp` on DD2's own six returned `gamma_sigma == 10.686681`,
the published value **bit for bit**, and `build_parametrization` reported
`stage='ok'` on it. Both route through `invert_nmp`, so both inherit the fix
with no second edit.

Two things found while measuring it, and they are why two dd2 tests moved:

- **A whole `compute_nmp` dict carries `Q_sat`, so `from_nmp(compute_nmp(par))`
  — the natural round trip — selects the 6x6**, not the shipped 5x5.
  `test_roundtrip_recovers_couplings` and `test_idempotent` were both 6x6 tests
  that read as 5x5 ones.
- **`test_roundtrip_recovers_couplings` passed only because the solver never
  moved**: it asserted the recovered couplings sit within 1e-3 of the published
  table, and compared the published couplings with themselves. `nmp.py`'s own
  docstring has always denied its premise — *"Do NOT read a 5x5 round trip as a
  test that recovers published couplings: no seed recovers them, because they
  are not a root."* Corrected to assert what the closure delivers: reaches a
  root (res < 1e-5), reports `coupling_shift > 1e-3`, reproduces the six imposed
  NMPs to < 5e-4. A correction, not a loosening — it measures a different,
  better-chosen quantity, and is strictly stronger on the NMPs.
- `test_idempotent` failed for the routing reason alone. On the six imposed keys
  the 5x5 is idempotent to **3.8e-08**, three orders inside the 1e-3 it asks for.

Also documented (not changed): **`from_nmp` returns `None` on a failed
inversion** and its docstring never said so, so a caller who skips
`return_status` gets `'NoneType' object has no attribute 'kernel_masses'` from
inside `solver.py`. zl's `from_nmp` raises instead — two models answering §6
differently is [ticket 103](103-nmp-closures-four-models.md)'s question, not
this one's.

### 5. Dropping Q_sat is not a workaround — unchanged, and nothing here proposes it

### What this hands to [ticket 105](105-dd2-isoscalar-conditioning.md), sharper

After the fix the 6x6 at DD2's own NMPs **converges instead of stalling** — to
max|residual| = 1.408e-02, under ISO_GATE by a factor 1.4, `ok=True` — while
imposing Q_sat only to **1.585 MeV** and K_sat to 1.9e-03. It saturates there:
64 and 128 restarts find nothing better. That is no longer a verdict defect (the
solver now reports honestly what it reached); it is the closure, and it is 105's
arithmetic — 515 x the 1.5e-03 stencil floor.

**The session's framing was right that the candidates must be judged against
amplified noise, and the measurement splits the two closures further than
expected**: in the 5x5 the noise never mattered at all — its own floor is
2e-10 .. 1e-07 and the whole defect was the suppressed restarts — while in the
6x6 the noise is the entire story and no change to the verdict can rescue it.
The 6x6 needs 105's reparametrisation or its analytic Q_sat; there is nothing
left here for a gate to do.

### Landed

- `eos/dd2/nmp.py` — `STALL_RES`, `_relative_shift`, `_stalled`;
  `InversionStatus.coupling_shift`; the stall as a miss in `_restart_loop`;
  the stall as `ok=False` in `invert_nmp`; module docstring corrected (its
  "property of the SciPy version" claim was wrong — it is the target's last
  bits); `from_nmp`'s None documented.
- `eos/dd2/verify/run_full_check.py` — `_check_restarts_extend_the_basin`.
- `test/dd2/test_dd2_m8.py` — two tests corrected, one added
  (`test_a_stalled_solve_is_reported_not_certified`).
- `eos/dd2/dd2.md`, `eos/dd2/dd2.tex` — §11 owes the new returned quantity.
- `docs/DEFERRED.md` — the solver half of the dd2 entry retired, the closure
  half restated and handed to 105.

### Gate

python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0.

- Full suite: **1812 passed, 23 skipped, 0 failed** (1835 collected, 46:30,
  exit 0). No failure added against `output/_audit/pytest_before.txt`.
- `test/dd2`: 211 passed (was 210, +1 for
  `test_a_stalled_solve_is_reported_not_certified`).
- dd2 `run_full_check`: **PASS**, ten checks including the new
  `restarts extend the basin` (0/9 cells at 0 restarts, 4/9 at 32). Golden
  SNM(0.16) **1.40e-05** and CompOSE HS(DD2) **2.83e-05** both UNMOVED, so no
  golden reference moved (§12).
