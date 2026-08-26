# eos/mixed has no species flags, so its photon gas cannot be turned off

Type: task
Status: open
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
prohibition, and it gains the entry via [ticket 22](22-phase5-claudemd.md).

Open for execution.

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
