# Should `zl_phase`, `vmit_phase` and `alphabag_phase` take a `flags`?

Type: grilling
Status: resolved (2026-08-29)
Blocked by: [95](95-vmit-solver-flags.md), [96](96-alphabag-solver-flags.md)
Parent: ../map.md

## Question

Surfaced by [ticket 94](94-zl-solver-flags.md), which hit it in one adapter and
declined to invent an API three times.

`eos/mixed/species.py` states the rule: photons and thermal neutrinos are
PHASE-COMMON and counted once at the mixture level, "which is why every shipped
adapter hands the phase it wraps a flag object with `photons=False`. ... The one
exception is a phase's `wing_sweep`, whose rows are stitched into the hybrid
table as they stand, with no mixture layer above them to add the radiation:
those carry the caller's own `photons`."

**Three adapters cannot obey the exception, because they are handed no flags at
all.** `dd2_phase(par, flags)` and `sfho_phase(par, flags)` take one;
`zl_phase(params=None)`, `vmit_phase(params=None)` and
`alphabag_phase(params=None)` take parameters and nothing else. So their wings
have no caller `photons` to carry, and each has had to pick a value:

    dd2_phase       wing follows `flags.photons`          the rule, obeyed
    sfho_phase      takes flags; check what its wing does
    zl_phase        `photons=False` since ticket 94       agrees with the
                                                          mixture at its default
    vmit_phase      `include_photons=True` default        radiation
    alphabag_phase  `include_photons=True` default        unconditionally

The two `True` rows are ticket 95's and 96's to move and will land on the same
choice zl faced. **That is why this is one ruling and not three**: whichever way
it goes, the three adapters should agree, and a hybrid table at T > 0 whose
wings and window disagree about the radiation is the defect underneath all of
it.

**The candidates.**

1. **Give all three a `flags=None` parameter**, defaulting to that model's
   all-False `SpeciesFlags`. The wing then follows the caller, `species.py`'s
   sentence becomes true again, and `zl_phase()` / `vmit_phase()` keep working.
2. **Rule the exception dead**: a wing carries `photons=False` in EVERY adapter,
   including `dd2_phase`, and `eos.mixed.hybrid` adds the radiation once when it
   stitches. Fewest places for the two to disagree, and it moves dd2's wing —
   [ticket 89](89-dd2-honours-species-flags.md) deliberately put it the other
   way and its reasoning has to be answered, not skipped.
3. **Leave it per-adapter and document why.** Costs nothing and keeps a rule
   `species.py` states but the code only half-keeps.

Whichever is chosen, say whether a hybrid table's wing and window must agree
about the radiation at T > 0, because that is the physics question under the
API one. `test/mixed/test_hybrid_modes.py` is what asserts the dd2 answer today.

**Not a baseline blocker.** `mixed.npz` is T = 0 throughout and moves under no
answer here.


## Resolution

**Candidate 1, with the `default_pair` forwarding that the ticket's own
wording did not include.** `zl_phase`, `vmit_phase` and `alphabag_phase` take
a `flags=None` defaulting to that model's all-False `SpeciesFlags`; their
wings follow it; and `default_pair` forwards `photons` to the vMIT phase, so
one flags object through the documented front door reaches all three sites.
Ruled by the user after the round below; candidate 3 fell with Q1.

### sfho's row, which the ticket flagged as unverified

`sfho_phase`'s wing passes the caller's `flags` straight to the four mode
solvers, exactly as `dd2_phase`'s does. Measured at `fixed_YC`, Y_C = 0.3,
T = 30 MeV, each wing row against the pure model's own solve with photons off
and then on. **Every digit below reproduces on BOTH stacks** — anaconda 3.9.7
(numpy 1.26.4 / scipy 1.13.1) and python.org 3.14.2 (numpy 2.3.5 / scipy
1.17.0) — so nothing here is a stack artifact:

    dd2        wing P = 32.5583428496   off 32.5352215181   on 32.5583428496
    sfho       wing P = 24.4893158932   off 24.4661945617   on 24.4893158932
    zl         wing P = 24.7324149993   off 24.7324149993   on 24.7555363307
    vmit       wing P = 241.7225433380  off 241.7225433380  on 241.7456646695
    alphabag   wing P = 183.4505746167  off 183.4505746167  on 183.4736959482

Every on-off gap is **0.0231213315 MeV/fm^3**, one photon gas at T = 30 — the
number [ticket 89](89-dd2-honours-species-flags.md) measured. So sfho's row is
"the rule, obeyed", and the ticket's table is otherwise correct as written.

### The defect was reachable through the documented front door

`PAIR = default_pair(PAR, FLAGS)` then `hybrid_table(PAIR, "fixed_YC", FLAGS,
...)` — the call `test/mixed/test_hybrid_modes.py` already makes, at T = 30
instead of 0. One `eos.dd2.SpeciesFlags` in, and:

    photons=False   mix -> Q at n_B 1.0631 -> 1.0995:  dP = +23.76249142
    photons=True    mix -> Q at n_B 1.0631 -> 1.0995:  dP = +23.73937009   <-- short

Turning the flag on added the gas to every `H` and every `mix` row and to
**none** of the `Q` rows: `480.65522175` was bit-identical across the two runs.
After the change the same two lines read `+23.76249142` and `+23.76249142`,
and the Q rows move to `480.67834308` — one gas, like everything else.

### The physics question the ticket demanded be answered

**A hybrid table's wing and window MUST agree about the radiation at T > 0.**
The photon gas is a function of T alone and the table is at one T; at n_offset
chi = 1, so the last window row and the first wing row describe the same
matter. A wing short of the gas puts a step of P_gamma = pi^2 T^4/45(hc)^3
there — 0.023 MeV/fm^3 at T = 30, three times that in eps — in a table
CLAUDE.md section 8 requires be monotone before it reaches TOV. Nothing in the
physics puts it there.

### Why candidate 1 over candidate 2, and what candidate 1 does NOT buy

The ticket counts five adapters; there are **eight** that carry a wing.
`did_phase(par, flags)` takes a required one, and **`njl_phase(par,
flags=None, patterns=None)` and `ccdm_phase(par, flags=None, ...)` already
have candidate 1's exact shape** — `flags=None` defaulting to the model's
all-False `SpeciesFlags`, passed straight into the wing solvers. So candidate
1 copies two adapters in the same file rather than inventing an API three
times, which is what [ticket 94](94-zl-solver-flags.md) declined to do.

It also moves no frozen value: the default is all-False, which is what the
three hardcoded, so `mixed.npz` cannot move. And it keeps
`test_hybrid_modes.py`'s stated strongest check — "a wing row must equal the
pure model's own solve at the same conditions to round-off, because it IS that
solve" — literally true. Candidate 2 would have made a wing row a model solve
PLUS a term the model did not add, and would have reversed
[ticket 89](89-dd2-honours-species-flags.md) across eight wings.

**Candidate 1 does not make disagreement unrepresentable, and that is
recorded rather than glossed.** A hand-built pairing — `zl_phase(p)` +
`alphabag_phase(p)` + `species=X` — can still be given three inconsistent
photon flags. Candidate 1 buys agreement for the shipped front door and leaves
it as caller discipline elsewhere; candidate 2's one advantage was exactly
that, and it was weighed and lost on the three counts above.

### Three sectors REFUSED rather than forwarded

Adding a `flags` argument opens sectors the wing can solve and the window
cannot match — the same defect as the photon step, wearing another name. Per
CLAUDE.md section 4 each raises rather than becoming a wing-only gas, on the
precedent of `dd2_phase`'s existing `sigma_star` refusal:

- **`alphabag_phase(gluons=True)`** — the adapter's `thermo` calls
  `_ab_from_mu` and adds no gluon gas, and the mixture has no gluon term at
  all (the engine's six flags do not name one).
- **`alphabag_phase(two_flavour=True)` and `vmit_phase(two_flavour=True)`** —
  both `thermo` surfaces are written for three flavours, and for vMIT the
  analytic quark Jacobian (`eos/mixed/backends/jacobian.py:113`) builds the
  s-flavour kappa unconditionally, so honouring it in the wing alone would
  give a Jacobian inconsistent with its own residual.
- **`alphabag_phase(thermal_neutrinos=True)`** — refused at the mixture level
  by `eos.mixed.species.SpeciesFlags` for every pairing anyway.

The other five sectors need no refusal: `hyperons`, `deltas`,
`thermal_mesons` and `muons` already raise in all three models'
`__post_init__`, and `thermal_neutrinos` raises in zl's and vmit's.

### The cold starts keep their photons off, deliberately

A cold start reads potentials and discards P, eps and s, so the radiation is
dead weight — the reason `dd2_phase` strips it from its own. zl's and
alphabag's now use an explicit `cold_flags = replace(flags, photons=False)`
beside a wing that follows the caller, which is the same shape dd2 has. This
is value-neutral in any case: a photon gas moves no density and no potential
(ticket 89 measured that).

### Q4, ruled IN scope by the user: the frozen blocks

`sfho_phase`'s `frozen_thermo` said "matter only — the same convention as the
DD2 block" in its comment and passed the caller's `flags` through unstripped,
where `_dd2_frozen_block` does `replace(flags, photons=False)` — which
[ticket 89](89-dd2-honours-species-flags.md) added deliberately, calling that
site "NOT a seed: its P and eps ARE the frozen sound speed". **`did_phase`'s
`frozen_thermo` has the identical shape and the identical comment**, so the
fix went to both rather than to the one the ticket named: one `replace(flags,
photons=False)` each. zl's, vmit's and alphabag's frozen blocks were already
clean (they call from_n surfaces that take no flags). The move is unmeasurable
in the baseline because nothing freezes `mixed.npz` at T > 0 with photons on;
it is a correctness fix, not a regeneration.

### The check that holds it (Q3)

`test/mixed/test_hybrid_modes.py` was **entirely T = 0**, which is why this
survived two tickets: the answer ticket 89 recorded had never been exercised
where photons exist. Three tests added, on one module-scoped fixture that
builds the same fixed-Y_C hybrid at T = 30 twice (2.7 s each):

- `test_photons_move_every_segment_by_exactly_one_gas` — the on-off delta in
  P and in eps is one photon gas on the `H`, `mix` AND `Q` rows alike, and no
  boundary moved.
- `test_hot_quark_wing_carries_the_callers_photons` — a hot Q row equals
  vMIT's own solve with `photons=True` to rel=1e-12.
- `test_a_wing_only_sector_raises_rather_than_becoming_a_gas` — the four
  refusals above.

**Proved able to fail.** Rebuilding the pre-fix wiring through the new API —
`(dd2_phase(PAR, f), vmit_phase(None, VMITFlags(photons=False)))` against a
mixture carrying photons — reproduces it exactly:

    PRE-fix   H dP = 0.02312133   mix dP = 0.02312133   Q dP = 0.0
    POST-fix  H dP = 0.02312133   mix dP = 0.02312133   Q dP = 0.02312133

### Gate

**The first gate run named the wrong interpreter and is reported as such.**
`python` on this machine is anaconda 3.9.7, not the canonical python.org
3.14.2 that [ticket 57](57-canonical-interpreter.md) ruled. That run —
`test/mixed test/baseline test/zl test/vmit test/alphabag test/sfho test/did`
— gave **4 failed, 576 passed in 27:34**, and the four are
`test_baseline[ccdm]`, `[enjl]`, `[njl]`, `[zlvmit]`: **exactly** the set
`map.md` records as the 3.9-stack cross-comparison ("miss on lepton-pressure
keys at ~1e-9; the `.npz` files were regenerated on 3.14 by ticket 62 ... not
a regression"). `mixed.npz`, `zl.npz`, `vmit.npz`, `alphabag.npz`, `sfho.npz`,
`did.npz` and `dd2.npz` all PASSED there. It is kept as the wide-coverage run;
the canonical gate below is the 3.14 one.


- **One shape across all five adapters**: every one now takes a flags object
  and its wing follows the caller's `photons`, verified by the five-row
  measurement above re-run after the change (all five read FOLLOWS).
- **`mixed.npz` unmoved** — see the suite line below; nothing in this change
  can move it, since every new default reproduces the old hardcoded value and
  `mixed.npz` is T = 0 throughout in any case.
- No call site broke: every existing caller of the three uses the no-argument
  or single-argument form (`zl_phase()`, `vmit_phase()`, `alphabag_phase()`,
  `vmit_phase(vmit_params)`), so the new second parameter is additive.

**The canonical run, python.org 3.14.2 (numpy 2.3.5 / scipy 1.17.0):**

    test/mixed  test/baseline  test/did
    373 passed, 0 failed, 0 skipped   (479.44 s)
    output/_audit/pytest_ticket109_py314.txt

Zero failures, so `mixed.npz` and all fourteen other baselines are unmoved on
the canonical stack. **`test/sfho` is deliberately NOT in the 3.14 run**: a
concurrent session is rewriting `eos/sfho/nmp.py` and `eos/sfho/parameters.py`
(and `CLAUDE.md`, `eos/dd2/nmp.py`, `map.md`) in this same checkout, so any
failure there would be theirs with no clean way to attribute it. The 3.9 run
above covered `test/sfho` green, and the two diffs touch disjoint files — mine
are `eos/mixed/{adapters,species,thermodynamics}.py`, `eos/mixed/mixed.{md,tex}`,
`docs/DEFERRED.md` and `test/mixed/test_hybrid_modes.py`.
