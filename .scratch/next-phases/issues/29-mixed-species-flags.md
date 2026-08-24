# eos/mixed has no species flags, so its photon gas cannot be turned off

Type: grilling
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
