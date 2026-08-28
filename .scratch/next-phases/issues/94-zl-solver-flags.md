# `zl.solver` takes `flags`, and `include_photons` goes

Type: task
Status: resolved
Blocked by: [90](90-solver-signature-and-units-sweep.md)
Parent: ../map.md

## Question

Split out of [ticket 90](90-solver-signature-and-units-sweep.md), which was
ruled by [ticket 81](81-second-default-solver-kwargs.md) §4 and gated on
"**no value moves anywhere**". That gate is false for this model, so the work
leaves ticket 90 rather than the gate being weakened — the split
[ticket 89](89-dd2-honours-species-flags.md) was given, applied to the three
models ticket 81 wrongly believed would move zero rows.

`eos/zl/solver.py` carries `include_photons: bool = True` on
`solve_beta_eq_neutrinoless`, `solve_fixed_yc`,
`solve_beta_eq_neutrino_trapped` and on `_finish`, and takes no
`SpeciesFlags` at all. `eos/zl/species.py:32` already carries
`photons: bool = False`, and `eos/zl/table.py:46` is the only thing that
translates one into the other — the ticket-89 shape exactly, one layer lower:
a solver's own callers get a photon gas whatever the flags say.

**Why values move.** `test/baseline/generate_baseline.py:468` (`case_zl`)
calls the raw solvers and names `params=` and `include_electrons=` only. Once
`solve` reads `flags.photons`, the generator picks up the §4 default, which is
`False` since [ticket 65](65-species-flag-defaults.md). Ticket 81 predicted
"zl moves **zero** rows"; it moves every row at T > 0.

`photons` is the ONLY live flag here. `SpeciesFlags.__post_init__` raises on
`hyperons`, `deltas`, `thermal_mesons`, `muons` and `thermal_neutrinos`, so
there is no second sector for the solver to honour or to forget.

## Work

1. `flags: SpeciesFlags` becomes a required argument of every solver in
   `eos/zl/solver.py`, placed after `par` (ticket 90 has already put `par`
   first and required, and spelled it `par`). `solve` and `_finish` read
   `flags.photons`; `include_photons` is deleted from both and from every
   wrapper.
2. `include_electrons` -> `leptons`, a SEPARATE named argument — §5, and
   already ruled by [ticket 70](70-leptons-on-a-beta-mode.md). It is not a
   species flag and never enters `SpeciesFlags`.
3. `eos/zl/table.py:46` stops translating and passes the flags through.
4. The 19 call sites outside `solver.py` name their flags. A site whose P,
   eps and s are DISCARDED (a warm-start seed, a potentials-only extraction)
   takes `photons=False` and says so in a comment — ticket 89 found three such
   sites its own work list had missed, so enumerate them from the source
   before editing rather than trusting this count.
5. Regenerate `test/baseline/zl.npz` under
   [ticket 65](65-species-flag-defaults.md)'s measure-then-regenerate gate.

## Gate

- **Measure BEFORE regenerating.** Re-evaluate `case_zl()` against the frozen
  file and diff key by key at rtol = 1e-10. Record moved, unmoved, and the
  residue.
- **Every moved key is P, eps or s at T > 0, and moves by exactly one photon
  gas.** Photons carry no conserved charge, so every density, every potential,
  every Y_C and Y_S is unchanged. A moved composition key is a bug, not a
  regeneration.
- **Zero keys move at T = 0.** `TEMPERATURES = (0.0, 10.0, 30.0)`, so the
  T = 0 third of the file is the control the same run carries with it.
- No other `test/baseline/*.npz` moves at rtol = 1e-10 — `mixed.npz`
  ASSERTED, not assumed, if any zl code is on its path.
- Full suite green, zero added failures against
  `output/_audit/pytest_before.txt`; `zl` `verify/` green.

## Resolution

Executed on python.org 3.14.2 (numpy 2.3.5, scipy 1.17.0). `eos/zl/solver.py`
takes a required `flags: SpeciesFlags`; `include_photons` is gone from
`_finish` and from all three solvers, `include_electrons` is `leptons`, and
`table.py` stops translating and passes the flags through.

### The signature is sfho's, not the one this ticket wrote

Work item 1 says "placed after `par`". Taken literally that gives
`(par, flags, n_B, ...)`, which is a FOURTH argument order — the three models
that already carry a required `flags` put it after the mode's own conditions:

    sfho, did   solve_beta_eq_neutrinoless(par, n_B, flags, T, ...)
                solve_fixed_yc(par, n_B, Y_C, flags, T, leptons=...)
                solve_beta_eq_neutrino_trapped(par, n_B, Y_Le, flags, T, ...)

§13 exists so "a physicist who has read one model can read the next without a
translation table", and [ticket 81](81-second-default-solver-kwargs.md) §4 is
the ruling that made `solver.py` uniform in the first place. ZL now matches
sfho and did exactly. `initial_guess` is NOT renamed to sfho's `x0` and `T`
keeps no default: ticket 90 left both alone in every model and this ticket is
not the place to widen its sweep.

`leptons` keeps its `True` default. [Ticket 91](91-leptons-default-and-drift-checks.md)
owns the flip to False, and moving it here would have moved rows this
ticket's gate does not allow for.

### The measurement, taken BEFORE regenerating

`zl.npz` has **1356 keys**. Re-evaluating `case_zl()` against the frozen file:

    control, no edits applied         0 moved of 1356, all bit-identical
    moved at rtol = 1e-10           144   (48 points x P, eps, s)
    unmoved                        1212   every one BIT-identical
    moved keys that are not P/eps/s   0
    moved keys at T = 0               0 of 216
    residue vs. one photon gas    0.000e+00  EXACT, at every one of the 144

The control is worth as much as the delta. The eight edits were held back and
`case_zl()` re-evaluated against the frozen file first: **0 of 1356 keys
moved, all 1356 bit-identical.** So the generator is deterministic on this
stack and the whole delta is attributable to the change and nothing else —
the null hypothesis measured at exactly zero, as in
[ticket 65](65-species-flag-defaults.md).

The 48 moved points are the entire T > 0 population: 18 `beta` (9 densities
at T = 10, 9 at T = 30), 6 `trapped`, 12 `yc.lep` and 12 `yc.nolep`, all at
T = 10. Not one density, potential, `Y_C` or `Y_S` moved anywhere, which is
the check that the sector removed carries no conserved charge.

The residue is stronger than ticket 89's. There the photon gas was ADDED to
an already-summed total and the move landed within 0.89 ulp; here it is not
added at all, so what remains is bit-for-bit the sum that preceded the
`result.P_total += gamma.P`, and `frozen - gamma == new` holds exactly at
all 144 keys.

`mixed.npz` and `zlvmit.npz` were **ASSERTED, not assumed**: `test/baseline`
is 20 passed at rtol = 1e-10 with only `zl.npz` regenerated, and all twelve
other `.npz` are BYTE-identical to a pre-change snapshot.

### Call sites: two the work list did not have

Work item 4 said 19 sites and told the next session to enumerate from source
rather than trust the count. Two of what it finds are not edits:

- **`eos/zlvmit/mixed_phase_eos.py:2386-2390` was already dead.** It calls the
  zl solvers in the PRE-[ticket 90](90-solver-signature-and-units-sweep.md)
  argument order — `solve_pure_H_beta(n_B_est, T, zl_params)`, so `par`
  receives a float and `default_guess` dies on `par.m_p`. All three sit inside
  a bare `except: pass`, so since ticket 90 they have raised silently and the
  routine has always fallen through to its hardcoded guesses. **Left alone and
  reported.** Repairing the order would start producing warm guesses this
  legacy solver has not had for two tickets, which can move `zlvmit.npz` — and
  this ticket's gate forbids that. `zlvmit` is §1-exempt legacy; restoring
  those three calls is a behaviour change that wants its own ruling.
- **`eos/mixed/adapters.py:912 `zl_phase(params=None)` takes no flags at all.**
  `eos/mixed/species.py` states the rule — every adapter hands its phase
  `photons=False` because radiation is counted once at the mixture level, and
  "the one exception is a phase's `wing_sweep`", which carries the caller's own
  `photons`. `zl_phase` is built from parameters alone and has no caller flags
  to carry, so it cannot obey that sentence. Given `photons=False` throughout,
  which agrees with the mixture whenever the flag is at its all-False default
  and is identical at T = 0, where the pairing is tested. Before this change
  the wing took the bare `include_photons=True` and carried radiation even at
  `MixedFlags(photons=False)`; it now agrees with the window in the default
  case instead of disagreeing in every case. `eos/mixed/species.py` says so
  where it described the exception.

**`vmit_phase` and `alphabag_phase` have the same shape** — both are
`(params=None)` and neither can carry a caller's `photons` either. Today they
take the `include_photons=True` default and so carry radiation
unconditionally, which is worse than what zl now does but is
[ticket 95](95-vmit-solver-flags.md)'s and [96](96-alphabag-solver-flags.md)'s
to move. **Whether the three flagless adapters should grow a `flags=`
parameter is one ruling, not three**, and it should be made once those two
land rather than invented separately in each.

### The verify suite keeps its photons

`zl/verify` gains a module-level `GAMMA = SpeciesFlags(photons=True)` and
passes it at all eleven solver calls. That is deliberate rather than
inertia: `_matter_only` subtracts exactly one photon gas off every state
before checking Euler and the free energy, so photons=True is what makes that
subtraction a test — the gas must be added and removable rather than woven
into the equations. Zero verify numbers move.

### The coverage gap ticket 81 named is closed

Ticket 81 recorded that `test/baseline` "does not exercise the `SpeciesFlags`
-> solver wiring" for zl, vmit and alphabag, and asked whether the gap should
be closed by a baseline case or a `verify/` entry. For zl it is now BOTH: the
generator passes a flags object, and `test/zl/test_zl_modes.py` gains
`test_the_photon_flag_reaches_the_solver`, which asserts the difference
between `photons=True` and `photons=False` through the RAW solver is exactly
one photon gas in P, eps and s and exactly zero in `n_p`, `n_n` and `mu_e`.
Before this ticket that difference was identically zero at every value of the
flag, so the check is non-vacuous by construction.

### Gate

    interpreter   python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0 / numba 0.63.1

    before  output/_audit/pytest_before_ticket94_py314.txt
            1812 passed, 23 skipped, 0 failed   (1835 collected, 21:04)
    after   output/_audit/pytest_after_ticket94_py314.txt
            1816 passed, 23 skipped, 0 failed   (1839 collected, 30:10)

**Zero added failures**, and unambiguously so: zero failures on both sides.
`test/baseline` 20 passed, `test/zl` + `test/mixed/test_phase_pairs.py` 61
passed, `zl/verify` PASS on all ten checks. Key-by-key diff kept at
`output/_audit/baseline_diff_ticket94_py314.txt`.

### A CONCURRENT SESSION held the tree throughout, and the denominator says so

Not a clean run, and the next session should not read 1835 -> 1839 as this
ticket's arithmetic. Another session wrote `eos/enjl/{table.py, enjl.md,
enjl.tex, verify/run_full_check.py}`, `eos/mixed/{boundaries.py,
construction.py, __init__.py}`, `docs/DEFERRED.md` and three issue files
between 16:13 and 16:29 — DURING the before-run (16:05-16:26) and across the
edits here. So:

- **The before-image is contaminated** as a control. Its 1835/1812/0 happens
  to reproduce ticket 102's recorded state exactly, but it was measured on a
  tree changing under it.
- **The +4 splits 3/1 and the split is checkable from mtimes.** Theirs:
  `test/enjl/test_enjl_construction.py` (16:23:56),
  `test/mixed/test_enjl_pair.py` (16:24:07),
  `test/mixed/test_locate_maxwell.py` (16:26:10). Mine: **exactly +1**,
  `test_the_photon_flag_reaches_the_solver` in `test/zl/test_zl_modes.py`
  (16:29:39); `test/zl/test_zl_interaction.py` was edited, not extended.
- **The two diffs touch disjoint files**, verified with `git diff --stat` over
  each set, so neither session's work can be hiding inside the other's. Every
  write here used an explicit pathspec.
- The runtime 21:04 -> 30:10 is CPU contention between the two, not this
  change: the 144 moved keys are the removal of an addition.

What carries the claim is therefore NOT the full-suite delta but the
attributable subset, which is disjoint from the other session's files and was
run separately and green: `test/zl`, `test/baseline`,
`test/mixed/test_phase_pairs.py`, and `zl/verify`.
