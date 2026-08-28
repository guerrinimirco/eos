# `vmit.solver` takes `flags`, and `include_photons` goes

Type: task
Status: resolved
Blocked by: [90](90-solver-signature-and-units-sweep.md)
Parent: ../map.md

## Question

Split out of [ticket 90](90-solver-signature-and-units-sweep.md) for the same
reason as [ticket 94](94-zl-solver-flags.md): ticket 90's gate is "**no value
moves anywhere**" and vMIT moves rows.

`eos/vmit/solver.py` carries `include_photons: bool = True` on all four
solvers — `solve_beta_eq_neutrinoless:188`, `solve_fixed_yc:287`,
`solve_fixed_yc_ys:416`, `solve_beta_eq_neutrino_trapped:540` — and takes no
`SpeciesFlags`. `eos/vmit/species.py:33` already carries
`photons: bool = False`; `eos/vmit/table.py:49` is the only translator.

**Why values move.** `test/baseline/generate_baseline.py:499` (`case_vmit`)
calls the raw solvers naming `params=` and `include_electrons=` only, so once
`solve` reads `flags.photons` the generator picks up the §4 default `False`
([ticket 65](65-species-flag-defaults.md)). Ticket 81 predicted zero moved
rows; every row at T > 0 moves.

`photons` is the ONLY live flag: `SpeciesFlags.__post_init__` raises on
`hyperons`, `deltas`, `thermal_mesons`, `muons` and `thermal_neutrinos`.

## Work

1. `flags: SpeciesFlags` required, after `par` (ticket 90 has already put
   `par` first and spelled it `par`). The four solvers read `flags.photons`;
   `include_photons` deleted from all four and from any helper that carries it
   through.
2. `include_electrons` -> `leptons`, a SEPARATE named argument (§5,
   [ticket 70](70-leptons-on-a-beta-mode.md)), never a species flag.
3. `eos/vmit/table.py:49` stops translating and passes the flags through.
4. The 21 call sites outside `solver.py` name their flags — including
   whatever `eos/mixed` reaches for, since vMIT is the shipped quark side of
   the DD2+vMIT front door. Enumerate from the source: a site whose P, eps and
   s are discarded (warm-start seed, potentials-only extraction, a frozen
   block whose P and eps ARE the frozen sound speed) takes `photons=False`
   with the intent stated, which is the trap
   [ticket 89](89-dd2-honours-species-flags.md) hit three times.
5. Regenerate `test/baseline/vmit.npz` under ticket 65's
   measure-then-regenerate gate.

## Gate

- **Measure BEFORE regenerating**: re-evaluate `case_vmit()` against the
  frozen file, key by key at rtol = 1e-10.
- **Every moved key is P, eps or s at T > 0 and moves by exactly one photon
  gas.** No density, no potential, no Y_C, no Y_S.
- **Zero keys move at T = 0.**
- **`mixed.npz` ASSERTED unmoved** at rtol = 1e-10, not assumed — vMIT is on
  its path, which zl is not. If a mixed row moves, a seed was mis-translated.
- Full suite green, zero added failures against
  `output/_audit/pytest_before.txt`; `vmit` and `mixed` `verify/` green.

---

## Note from [ticket 94](94-zl-solver-flags.md) (2026-08-28)

This model's `eos/mixed` adapter takes **no flags object** — it is
`(params=None)` — so its `wing_sweep` cannot carry the caller's own `photons`
the way `eos/mixed/species.py` says a wing must. Ticket 94 hit the same thing
in `zl_phase` and gave it `photons=False` rather than invent an API three
times. **Do the same here and do NOT add a `flags=` parameter**: that is
[ticket 109](109-flagless-mixed-adapters.md), which is blocked by this ticket
and is one ruling covering all three adapters.

Two more findings from 94 that apply directly:

- **The signature is sfho's and did's**, `(par, n_B, [fraction], flags, T)`,
  not the literal "after `par`" these tickets were written with — that would
  be a fourth argument order against §13. Do not rename `initial_guess` to
  `x0` or give `T` a default; ticket 90 left both alone everywhere.
- **`leptons` keeps whatever default it has.**
  [Ticket 91](91-leptons-default-and-drift-checks.md) owns the flip to False,
  and moving it here moves rows the measure-then-regenerate gate does not
  allow for.

## Resolution

Executed on python.org 3.14.2 (numpy 2.3.5, scipy 1.17.0). `eos/vmit/solver.py`
takes a required `flags: SpeciesFlags`; `include_photons` is gone from all four
solvers, `include_electrons` is `leptons`, and `table.py` stops translating and
passes the flags object through.

### The before-image IS the post-ticket-100 state, and that is measured

The question this ticket was opened with. `test/baseline/` is gitignored, so
git cannot answer it; the tree answers it directly. Re-evaluating `case_vmit()`
against the on-disk `vmit.npz` with `SpeciesFlags(photons=True)` — the sector
configuration the deleted kwarg default gave — moves **0 of 1119 keys**. So the
frozen file already carries [ticket 100](100-vmit-point-Y_S-never-assigned.md)'s
regeneration: all 42 `.Y_S` keys are non-zero (`beta.T0.n0.45.Y_S =
0.8402368306`, the number 6c4d9bd quotes), and none of them is in this ticket's
delta. The two diffs are disjoint by construction, not by assertion.

That run is also the null control, and a stronger one than
[ticket 94](94-zl-solver-flags.md)'s: 94 held its edits back and re-ran the old
code, which measures the generator's determinism; this one runs the NEW code
with the sector switched back on, which measures determinism AND isolates the
flag as the only cause of the delta.

### The signature is njl's and ccdm's, not zl's

Ticket 94 found the hadronic order `(par, n_B, [fraction], flags, T)` and warned
that "after `par`" taken literally would invent a fourth. vMIT is a QUARK model
and its siblings order it the other way — `njl` and `ccdm` are
`(par, n_B, [fraction], T, flags, ...)`, with `T` before the flags:

    njl, ccdm   solve_beta_eq_neutrinoless(par, n_B, T, flags, x0=None)
                solve_fixed_yc(par, n_B, Y_C, T, flags, leptons=...)

vMIT now matches those exactly. This is the same ruling 94 made, applied to the
right neighbours: §13 asks that a physicist reading one model can read the next,
and for a quark model the next one is `njl`, not `zl`. It is also the smaller
diff — vmit's `T` was already the third positional, so every existing call site
gains an argument rather than reordering.

`initial_guess` is NOT renamed to `x0` and `T` keeps no default (ticket 90 left
both alone everywhere). `leptons` keeps its `True` default;
[ticket 91](91-leptons-default-and-drift-checks.md) owns the flip.

### `two_flavour` went into the flags object too, and had to

Not in this ticket's work list, and it is one line of physics rather than
scope creep: `SpeciesFlags.two_flavour` already existed AND the four solvers
carried a parallel `two_flavour: bool = False` kwarg. Leaving it would have
left the solver accepting "BOTH a SpeciesFlags and a parallel kwarg for a
sector the flags object carries" — exactly what
[ticket 91](91-leptons-default-and-drift-checks.md)'s second drift check
forbids, and what §4 means by a sector controlled by its flag. It is
value-neutral: both spellings defaulted False and the only caller that passed
it (`table.solve_at`) read it off `species` anyway. `alphabag` has the same
shape and [ticket 96](96-alphabag-solver-flags.md) does the same thing.

The internals keep the plain bool — `default_guess`, `warm_start`,
`thermo_from_mu*`, `two_flavour_state`. Those are not solvers, and threading a
flags object into `thermodynamics.py` would put a species-flag object on the
side of the §5 boundary that must not know which mode it is in.

### The measurement, taken BEFORE regenerating

`vmit.npz` has **1119 keys**.

    control, photons=True (the old default)     0 moved of 1119
    moved at rtol = 1e-10                     108   (36 points x P, eps, s)
    unmoved                                  1011
    moved keys that are not P/eps/s             0
    moved keys at T = 0                         0
    residue vs. one photon gas          EXACT at all 108

The 36 moved points are the entire T > 0 population: 12 `beta` (6 densities at
T = 10, 6 at T = 30), 18 `yc` (3 Y_C x 3 n_B x lep/nolep, all T = 10), 3 `ycys`
and 3 `trapped`. Not one density, potential, `Y_C` or `Y_S` moved anywhere,
which is the check that the sector removed carries no conserved charge. The
residue is bit-exact rather than within an ulp, for ticket 94's reason: the gas
is not added at all, so what remains is the sum that preceded
`result.P_total += gamma.P`, and `frozen - gamma == new` holds exactly.

Key-by-key diff at `output/_audit/baseline_diff_ticket95_py314.txt`.

**`mixed.npz` ASSERTED unmoved, and every other file with it.** md5 over all
fourteen `test/baseline/*.npz` before and after: only `vmit.npz` differs.
That is stronger than the rtol=1e-10 the gate asked for — the other thirteen
are BYTE-identical.

### Call sites: what the enumeration found

Work item 4 said 21 sites and told this session to enumerate from source. Two
things the list did not have:

- **`eos/mixed/adapters.py`'s `vmit_phase` takes no flags object**, exactly as
  ticket 94 found for `zl_phase`. Given `photons=False` throughout, through a
  module constant `_VMIT_MATTER_ONLY` whose comment states the reason. **No
  `flags=` parameter was added**; that is
  [ticket 109](109-flagless-mixed-adapters.md), one ruling for all three
  adapters. `eos/mixed/species.py` now names `vmit_phase` beside `zl_phase`
  where it describes the exception.
- **A call reached only through an alias.** `test/mixed/test_hybrid_modes.py:116`
  calls `solve_beta_eq_neutrino_trapped` imported as `vmit_trapped`, and a grep
  for the solver names does not see it. It survived to the test run and failed
  there (1 failed, 302 passed), which is the enumeration working rather than
  the list being right. A source enumeration for this repository has to grep
  the `as`-aliases too.
- **`eos/zlvmit/mixed_phase_eos.py:2460-2464` was already dead**, the same way
  94 found zl's three to be: `solve_pure_Q_beta(n_B_est, T, vmit_params)` is
  the PRE-ticket-90 argument order, so `par` receives a float and the call dies
  inside a bare `except:`. It died before this change and dies after; **left
  alone and reported**. `zlvmit` is §1-exempt legacy and repairing the order
  would start feeding it warm guesses it has not had for two tickets, which can
  move `zlvmit.npz` — which this gate forbids.

### The verify suite keeps its photons

`vmit/verify` gains a module-level `GAMMA = SpeciesFlags(photons=True)` and
passes it at all eleven solver calls, for ticket 94's reason: `_quark_only`
subtracts exactly one photon gas off every state before Euler and the free
energy are tested, so `photons=True` is what makes that subtraction a test. Its
comment "Photons are on by default in every solver call this suite makes" is
now true because the suite says so rather than because a kwarg did. Zero verify
numbers move.

### The coverage gap ticket 81 named is closed for vmit

`test/vmit/test_equilibrium.py` gains
`test_the_photon_flag_reaches_the_solver`: the difference between
`photons=True` and `photons=False` through the RAW solver is exactly one photon
gas in P, eps and s and exactly zero in `n_u`, `n_d`, `n_s` and `mu_e`. Before
this ticket that difference was identically zero at every value of the flag, so
the check is non-vacuous by construction.

### Gate

    interpreter   python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0

    test/vmit + test/mixed      303 passed, 0 failed  (9:14)
    test/baseline               19 passed, 0 failed   (alphabag DESELECTED,
                                see below)
    vmit/verify                 PASS, all nine checks
    other .npz                  13 of 13 BYTE-identical

**One honest caveat on the denominator.** `test/baseline` was run with
`-k "not alphabag"`. This session works [ticket 95](95-vmit-solver-flags.md),
[96](96-alphabag-solver-flags.md) and [91](91-leptons-default-and-drift-checks.md)
back to back at the user's request, and ticket 96's `eos/alphabag` edits were
already in the tree when this gate ran; `case_alphabag` therefore could not
have run against a signature it predates. The alphabag baseline is measured and
gated under ticket 96, and the full suite is run once at the end of the three.
Nothing in `test/baseline`'s other nineteen entries touches alphabag, and the
md5 comparison above covers `alphabag.npz` as an unmoved FILE.

