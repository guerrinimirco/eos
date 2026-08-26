# eos/mixed has no species flags, so its photon gas cannot be turned off

Type: task
Status: resolved
Assignee: session 7bd72c3a
Parent: ../map.md

## Question

`eos/mixed/` has **no `species.py`**. `thermodynamics.py:84` calls
`photon_thermo(T)` unconditionally whenever T > 0, so the composite engine always
returns a photon contribution to `eps`, `P` and `s` and offers the caller no way
to say otherwise.

CLAUDE.md §4 requires every degree of freedom beyond the nucleons to be an
explicit named boolean, with identical names across models, and forbids a sector
being on implicitly. §5's list of what a composite engine carries —
`adapters.py`, `api.py`, `responses.py`, `verify/`, its own `<name>.tex` — does
**not** include `species.py`, and a composite engine "is not a model and does not
take this list". So it is genuinely unclear whether §4 binds `mixed` at all.

This is a **different finding** from
[ticket 28](28-photons-silent-ignore.md), which was dd2 declaring a flag and not
reading it. Here there is no flag to ignore. Ticket 08's report framed the two as
one bug; they need different fixes, which is why they are separate tickets.

The design question underneath: **how should a composite engine take species
flags at all?** A pairing is two `Phase` objects, each closing over its own
model's parameters and therefore its own `SpeciesFlags`. Photons, though, are
phase-common — they belong to neither phase, like the eta-split leptons and the
trapped neutrinos in the same function. So a photons flag on `mixed` is not a
per-phase flag and cannot simply be delegated.

Three ways out:

- **Give `mixed` its own flag set** for the phase-common sectors (photons,
  thermal neutrinos), separate from the two phases' own flags. Most faithful to
  §4; adds a `species.py` that §5's engine list does not mention.
- **Take them as explicit keyword arguments** on the engine's entry points,
  matching how `include_photons` already threads through
  `adapters.py:224,537,583` and `scan.py:264`. Smaller, but leaves `mixed` the
  one place where a sector is a kwarg rather than a flag.
- **Record it in `docs/DEFERRED.md`** as a known gap with the reasoning — the
  engine always carries photons at T > 0 and says so — matching how the ledger
  already handles `mixed`'s `Y_Lmu` refusal.

Whichever way, §11's requirement that every returned quantity be documented means
`mixed.md`/`mixed.tex` must state the photon treatment either way; today they do
not.

## Ruling

Agreed with the user: **`mixed` takes a §4 flag set and passes the per-phase
sectors to the models it couples**, consuming the phase-common ones itself.

**Measured: there is no double-counting risk, because the code is already built
this way.** `adapters.py:225,538,584` already pass `include_photons=False` into
every phase, and `mixed/thermodynamics.py:84` adds `photon_thermo(T)` once at
the mixture level. The separation between per-phase and phase-common sectors
already exists in the implementation — what is missing is only the **flag**.
Adding it makes the adapters' hardcoded `False` correct by construction rather
than correct by accident.

So: `mixed` gains a `species.py` carrying the six §4 names; `hyperons`,
`deltas`, `muons` and `thermal_mesons` delegate to the two `Phase` objects;
`photons` and `thermal_neutrinos` are consumed at the mixture level, where the
eta-split leptons already are.

**[Ticket 65](65-species-flag-defaults.md) is what makes this urgent.** Once
`photons` defaults to `False` everywhere, an engine that switches photons on
unconditionally is the only place in the package where a sector is enabled
implicitly — §4's exact words.

§5's engine list (`adapters.py`, `api.py`, `responses.py`, `verify/`, the
`.tex`) does not mention `species.py`; it is a list of what an engine HAS, not a
prohibition, and it gains the entry via [ticket 85](85-claudemd-sentences-owed.md).

Executed; see the Resolution below.

## Binding ruling from ticket 65

[Ticket 65](65-species-flag-defaults.md) is **resolved**: all six of §4's flag
names default to `False` in every model, `enjl` excepted (it fixes every flag
and raises on any move). **Whatever `species.py` this ticket gives `eos/mixed`
must default all six to `False`.**

No edit is owed today: `mixed` has no `species.py` and reuses `dd2`'s flags, so
it already inherited the change. The obligation is on the module 29 creates.

`test/test_imports.py::test_the_six_species_flags_all_default_to_off` is the
check that enforces it, and it iterates `eos.MODELS` minus a named `exempt`
dict — so it covers `mixed` automatically as soon as `mixed` is in the
iterated set, and a stale exemption turns it red rather than passing quietly.
Note the engine's own flags are not covered by that rule: `chi` and anything
else that is the composite engine's physics defaults where 29 says it does,
the way `phi_field`, `gluons` and `csc` do in the models.

## Sharpened by [ticket 84](84-vmit-params-in-the-plumbing.md)

`mixed` does not merely LACK a `species.py`. `eos/mixed/api.py:49` reads
`from eos.dd2.species import SpeciesFlags` — the engine borrows a MODEL's flag
class as its own, and its public docstrings say `species : dd2 SpeciesFlags`.
So a caller pairing SFHo with NJL is still asked for a DD2-typed object, and
`import eos.mixed` still pulls DD2 in.

That is why this ticket and 81 are one problem seen from two sides, and it makes
this ruling's `species.py` the first half of 81's answer rather than a tidy-up.


## Resolution

Executed on `main` at `8bb546c`. `eos/mixed/species.py` exists and carries
CLAUDE.md §4's six names, all defaulting to `False` — ticket 65's binding rule,
adopted rather than reinvented. `eos/mixed/api.py` no longer imports
`eos.dd2.species`, and no public docstring says `dd2 SpeciesFlags`.

### The split, and why it needed no new bookkeeping

The ruling's measurement held: `adapters.py` already passed
`include_photons=False` into every phase and `thermodynamics.py` already added
`photon_thermo(T)` once, at the mixture level. What was missing was only the
flag, and adding it made those hardcoded `False` values correct BY
CONSTRUCTION. Demonstrated rather than asserted, in the new test: switching
`photons` on at T = 20 MeV, n_B = 0.65 moves P, eps and s by exactly
`photon_thermo(T)` to round-off (3.6e-15 on P), which is only true if no phase
carries a photon of its own. At T = 0 the flag is bit-for-bit inert.

- per-phase — `hyperons`, `deltas`, `thermal_mesons`, `muons`: delegated, each
  `Phase` closing over its own model's flags.
- phase-common — `photons`, `thermal_neutrinos`: consumed at the mixture level,
  where the eta-split leptons already are, and counted once.

`thermal_neutrinos` is carried and RAISES, exactly as `dd2` does with the same
name: the flavours a mode does not track are not wired in the engine. It is
NOT the trapped electron neutrino of `beta_eq_neutrino_trapped`, which is
matter composition and comes from the mode, never from a flag. This is the one
line of the ruling that could have been read as "wire it"; it is not wired, and
`mixed.md` / `mixed.tex` say so in the Not-implemented section.

### `muons=` retired — as one object, not two booleans

The `muons=None` keyword argument is gone from all four public entry points, as
the ticket required. It was also threaded through `solver`, `boundaries`,
`table`, `hybrid` and `responses` to reach the mixture level, and rather than
run a second bool alongside it for `photons`, that whole chain now threads the
FLAG OBJECT as `species=`. So `photons` needed no plumbing of its own, and
`thermal_neutrinos` (or anything §4 adds later) needs none either. The leaf
helpers that genuinely take a bool — `charged_leptons`, `neutral_phase`,
`total_pressure`, `locate_maxwell`, `sound_speed_frozen_quark` — keep it; the
engine threads flags, the leaves take booleans.

`mixture_flags(flags)` reads the six names off whatever object a caller gave:
the engine's own `SpeciesFlags`, or a coupled model's, since every model
carries the same six. That is what lets the DD2 + vMIT front door hand ONE
`eos.dd2.SpeciesFlags` to both the hadronic phase and the mixture, and it is
why no test call site had to grow a second flags argument.

### Left for ticket 86, deliberately

`species` on the four entry points is no longer a DD2 type, which is the half
86 was blocked on. The front door's *default* flags — what `species=None` with
`par=` falls back to — moved OUT of `api.py` and into
`adapters.default_flags()`, the layer that is allowed to import a model. So
`api.py` is clean today and 86 deletes `default_flags` along with the rest of
the front door when `phases=` becomes the parameter argument. Not done here,
by instruction: `charges.py`'s `hadronic_qn` import, `responses.py`'s
`warm_start`, `__init__.py`'s re-export, and the `vmit_params` retirement are
all 86's.

`eos/mixed/__init__.py` was NOT touched: it was dirty with ticket 87's work at
the time, and 86 owns the re-export anyway. `docs/DEFERRED.md` was NOT touched
for the same concurrency reason — it carried another session's uncommitted
edits, and staging it would have swept them into this commit. The
`thermal_neutrinos` gap is stated in `mixed.md` / `mixed.tex`; a DEFERRED row
for it is owed and is the one loose end this ticket leaves.

### Gate

Run on **python.org 3.14.2** (numpy 2.3.5, scipy 1.17.0), never prefixed with
`timeout`, in an isolated `git archive HEAD` copy — the working tree carried a
concurrent session's in-flight edits throughout. Gated as an isolated-copy
PAIR: a HEAD control and a HEAD+29 copy, with the concurrent session's
non-`mixed` edits copied into BOTH so the only difference between them is this
ticket.

    test/mixed + test/baseline/test_baseline.py    282 collected
      HEAD control   274 passed, 1 failed
      HEAD + 29      281 passed, 1 failed        (+7 = this ticket's new tests)

The single failure is `test_baseline[ccdm]`, IDENTICAL in the control and
therefore not this ticket's — it is not in `eos/mixed` and was not touched.
`test_baseline[mixed]` passes at rtol = 1e-10: the mixed baseline is a T = 0
sweep and the photon term is a finite-T one, so no number moved.
`eos/mixed/verify/run_full_check.py`: **PASS**, all ten `[ok ]`.

HEAD moved mid-ticket — the concurrent session landed tickets 87 and 75 — so
the pair was rebuilt on the new HEAD and re-run; the numbers above are from
that second run.

### Files

- `eos/mixed/species.py` — new: `SpeciesFlags` (six names, all False),
  `mixture_flags`
- `eos/mixed/thermodynamics.py` — `assemble(..., photons=False)`
- `eos/mixed/solver.py` — `MixedCtx.species` in place of `MixedCtx.muons`;
  `build_mixed_ctx(species=)`; `solve`/`sweep` take `species=`
- `eos/mixed/boundaries.py`, `table.py`, `hybrid.py`, `responses.py` — the
  `muons=` chain becomes `species=`; `TableSpec.muons` becomes
  `TableSpec.species`
- `eos/mixed/backends/jacobian.py` — `ctx.species.muons`
- `eos/mixed/adapters.py` — `default_flags()`
- `eos/mixed/api.py` — the dd2 import gone; `muons=` gone from all four entry
  points; the trapped-mode `neutrinos` check narrowed to the front door, where
  it is the DD2 wing's requirement (the adapter already enforces it for a
  `Phase` pair)
- `eos/mixed/mixed.md`, `eos/mixed/mixed.tex` — §11: the photon treatment,
  the per-phase / phase-common split, the corrected signatures, and
  `thermal_neutrinos` in Not-implemented. The old text said the opposite in
  three places.
- `test/mixed/test_species_flags.py` — new, 6 tests (untracked, §11)
- `test/mixed/test_finite_temperature.py` — its docstring claimed photons
  always enter the totals (untracked, §11)
- `test/mixed/test_{phase_pairs,njl_pair,ccdm_pair,enjl_pair}.py` — the
  `muons=True` call sites become `species=MU` (untracked, §11)
