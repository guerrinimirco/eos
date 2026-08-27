# The CLAUDE.md sentences the post-22 rulings owe

Type: task
Status: open
Blocked by: 29, 70, 84
Parent: ../map.md

## Question

[Ticket 22](22-phase5-claudemd.md) was the vehicle for CLAUDE.md amendments and
**is resolved**. Rulings made after it closed still say "§N gains the sentence
via ticket 22", so those amendments have no home. This is the successor.

Collected, each with the ruling that owes it:

1. **§3 — `leptons=` on a beta-equilibrium mode**
   ([ticket 70](70-leptons-on-a-beta-mode.md)). `leptons=False` on a beta mode
   RAISES; `leptons=True` is accepted and ignored as redundant. §3 currently
   says only that the flag "applies to `fixed_YC` and `fixed_YC_YS`", which
   leaves the beta-mode case to six models to answer three ways. State the rule
   and the reason: in a beta mode the leptons are constitutive, not
   unimplemented, so §4's "a flag a model does not implement RAISES" does not
   govern `True`.

2. **§5 — the composite engine's file list gains `species.py`**
   ([ticket 29](29-mixed-species-flags.md)). The list
   (`adapters.py`, `api.py`, `responses.py`, `verify/`, its own `.tex`) is a
   list of what an engine HAS, and the ruling gives `mixed` its own flag set for
   the phase-common sectors.

3. **§5 — the DD2+vMIT front door sentence is RETIRED**
   ([ticket 84](84-vmit-params-in-the-plumbing.md)). §5 currently reads "with
   the plain `(par, flags, vmit_params)` signatures remaining the DD2+vMIT front
   door". Ticket 81 retires that signature entirely: `phases` becomes the
   parameter argument in fact as well as in the sentence before it. Delete the
   clause; the DD2+vMIT convenience survives as `adapters.default_pair`, which
   is a call, not a privileged position.

4. **§1 — `mixed/scan.py` disappears from the astro carve-out**
   ([ticket 84](84-vmit-params-in-the-plumbing.md), ticket 84). §1 currently
   reads "`mixed/hybrid.py` and `mixed/scan.py` import `eos.astro.tov`".
   `scan.py` is being removed, so the carve-out names one file, not two.
   **Shipped**: [ticket 87](87-remove-mixed-scan.md) deleted the file. It also
   found the same sentence echoed at `docs/DEFERRED.md:145` and left it here,
   so this ticket sweeps both copies together.

5. **§4 — a sector flag may be refused by a MODE without stopping being a
   default** ([ticket 92](92-cfl-gluon-term.md)). §4 now carries ticket 82's
   two-category rule: "a flag with two legal values is a DEFAULT and is False;
   a flag with only one legal value RAISES on the other and is a STATEMENT
   about the model. There is no third category." Ticket 92 refuses
   `alphabag.gluons` and `alphabag.thermal_neutrinos` in the `cfl` mode alone,
   which reads as that forbidden third category until one notices the rule is
   about the FLAG and this refusal is about the PHASE. State the distinction,
   because it is not a carve-out and it must not be read as one:

   > A flag's category is a property of the flag, judged over the modes the
   > model has. A mode may refuse a sector its physics does not contain
   > without changing the flag's category: `alphabag.gluons` keeps two legal
   > values in the unpaired modes and is a default there, and raises in `cfl`
   > because a colour-flavour-locked phase has no free gluon gas. That is the
   > same statement `abpr` makes by refusing the flag outright — `abpr` is
   > that phase and nothing else, so for it the phase's statement and the
   > flag's category coincide.

   Two things make this safe to state rather than a licence to drift. §3
   already holds the ground: `cfl` "is not a choice of equilibrium condition
   but a statement about which phase the model describes", so a per-mode
   sector refusal is the same kind of claim §3 already makes about the mode
   itself. And the drift check
   (`test_every_species_flag_defaults_off_or_raises`) is untouched — it
   iterates flag DEFAULTS, so a mode-conditional raise neither weakens it nor
   needs an exemption in it, which is exactly the property ticket 82's rule
   was chosen for.

   §4's "Setting a flag a model does not implement RAISES; a
   NotImplementedError is never turned into a silent no-op" needs no change:
   ticket 92 is that sentence being ENFORCED, not amended. The `cfl` arm of
   `alphabag.table.solve_at` had been dropping `thermal_neutrinos` silently.

**Do not batch these blind.** §2's warning applies to the specification too: a
sentence added to CLAUDE.md becomes an invariant the whole suite encodes, so
each one lands only when its ruling has actually shipped — which is why this
ticket is blocked by all three rather than running ahead of them.

Done when each sentence is in CLAUDE.md, each ruling's ticket points at the
section it changed, and no open ticket still cites ticket 22 as its vehicle.
