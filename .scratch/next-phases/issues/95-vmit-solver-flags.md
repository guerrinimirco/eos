# `vmit.solver` takes `flags`, and `include_photons` goes

Type: task
Status: open
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
