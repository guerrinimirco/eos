# The CLAUDE.md sentences the post-22 rulings owe

Type: task
Status: resolved
Blocked by: 70 (29, 84 and their execution ticket 86 have shipped;
item 5 arrives from 92, which is resolved)
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
   ([ticket 29](29-mixed-species-flags.md), shipped in `8bb546c`). The list
   (`adapters.py`, `api.py`, `responses.py`, `verify/`, its own `.tex`) is a
   list of what an engine HAS, and the ruling gives `mixed` its own flag set for
   the phase-common sectors.

3. **§5 — the DD2+vMIT front door sentence is RETIRED**
   ([ticket 84](84-vmit-params-in-the-plumbing.md)). §5 currently reads "with
   the plain `(par, flags, vmit_params)` signatures remaining the DD2+vMIT front
   door". Ticket 84 retires that signature entirely (the clause above said
   "ticket 81" — a renumber straggler; 81 is the solver-kwargs vocabulary
   ticket and has nothing to do with this).

   **SHIPPED** by [ticket 86](86-mixed-phase-pair-primary.md): `phases` is now
   the first positional argument of `eos_point`, `eos_table`, `hybrid_table`
   and `eos_response`, `(par, flags, vmit_params)` appears nowhere below them,
   and `grep -rn vmit_params eos/` outside `eos/zlvmit/` returns only the
   `default_pair(par, flags, vmit_params)` call form the ruling itself
   prescribes. So the sentence to land is a DELETION plus a replacement clause:

   > for the composite engine the Phase pair IS the parameter argument (which
   > is how §6's "parameters are arguments" reads there), in the first
   > position of every public entry point; the DD2+vMIT pairing is built by
   > `adapters.default_pair(par, flags, vmit_params)`, a call rather than a
   > privileged position.

   Two smaller consequences of the same ticket, to land in the same edit:

   - §5's phase-adapter paragraph lists the shipped adapters; nothing there
     changes, but the sentence that follows must not reintroduce a front door.
   - §7's single-home list needs no new entry: `hadronic_qn` /
     `hadronic_charges` moved from `eos/dd2/species.py` to
     `eos/general/basis.py` alongside `quark_charges`, which is what §2's
     "the density sums (n_B, n_C, n_S) from species densities live in
     `general/`" already required. Ticket 84 corrected itself on the reason
     (§7 single home, not a §2 duplicate); no sentence is owed, only the
     move, and the move has shipped.

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

---

## Resolution

Six sentences landed in `CLAUDE.md`; **two more that ticket 81 owed did NOT**,
because their rulings have not shipped. Nothing in `eos/` changed — this is a
prose commit, and the greps below are the whole of its gate.

### Landed, each re-read against the code first

- **§1 — the astro carve-out names `mixed/hybrid.py` ALONE.**
  `eos/mixed/scan.py` is absent, and the only real `import eos.astro.tov`
  under `eos/mixed/` is `hybrid.py:237-238`. Every other hit in the package is
  docstring prose — ten model `api.py` files say "the result feeds
  `eos.astro.tov`", which is not an import, and reading a grep count as the
  answer here would have manufactured a §1 violation in ten models.
- **§2 — the Naming block gains the `_nat` convention.** 147 `_nat` sites
  across ten packages; `docs/STRUCTURE.md` §5 already carried the rule for
  solver working variables, so this is CLAUDE.md catching up to a convention
  already in force.
- **§3 — `leptons=False` on a beta mode RAISES; `leptons=True` is accepted and
  ignored.** All ten `api.py` docstrings state it, and it is enforced once for
  all in the shared `ModeSpec`, covered by
  `test/general/test_modes.py::test_a_declaration_that_cannot_mean_anything_raises`
  (`ModeSpec(leptons=False)` -> `ValueError`, match `"fixed-fraction"`).
- **§4 — a flag's category is judged over the modes the model HAS, so a mode
  may refuse a sector.** `eos/alphabag/solver.py`'s `solve_cfl` refuses
  `include_gluons=True`; `eos/alphabag/table.py`'s `cfl` arm refuses
  `thermal_neutrinos`. Both `NotImplementedError`, neither a silent drop.
- **§5 — the DD2+vMIT front-door clause is REPLACED by the `default_pair` call
  form.** `phases` is the first positional argument of `eos_point`,
  `eos_table`, `hybrid_table` and `eos_response`
  (`eos/mixed/api.py:103,171,227,284`), and `grep -rn vmit_params eos/` outside
  `zlvmit` returns only the six `default_pair(par, flags, vmit_params)`
  call and docstring sites the ruling itself prescribes.
- **§5 — the composite-engine file list gains `species.py`.**
  `eos/mixed/species.py` exists (ticket 29, `8bb546c`), all six flags default
  False, and it carries the per-phase / phase-common split the new clause
  states.

The §4 sentence is an ADDITION beside ticket 82's two-category rule, which was
verified already present in §4 and **not** duplicated. Item 3's note that §7
owes nothing for the `hadronic_qn` move holds: `eos/general/basis.py` carries
it, which §2 already required.

### NOT landed, and why

Ticket 81 owed three sentences, not one. The other two fail this ticket's own
rule — a sentence becomes an invariant, so it lands only once its ruling has
shipped:

- **§3: `leptons=` defaults to False.**
  [Ticket 91](91-leptons-default-and-drift-checks.md) is **open**. Nine models
  still disagree (`False` in dd2/sfho/did/alphabag, `True` in
  enjl/njl/ccdm/zl/vmit and `eos/mixed/solver.py`), so the sentence would be
  false the day it was written.
- **§5: the units sentence names `s` (and `n_s`).**
  [Ticket 97](97-natural-record-leaves-the-result.md) is **open**. The
  natural-units record is still on the public result in three models —
  `n_B_fm`, `P_fm`, `eps_fm`, `s_fm`, `n_C_fm`, `n_S_fm` accessors survive in
  `eos/njl/thermodynamics.py` and `eos/enjl/thermodynamics.py`. The `_nat`
  half above is landed because it IS shipped; the boundary half waits for 97.

Both are recorded on [ticket 90](90-solver-signature-and-units-sweep.md) so
they are not lost when 91 and 97 close. **This ticket does not stay open for
them**: it succeeded ticket 22 in the vehicle role, and a collector that never
closes is the failure it was created to fix. 91 and 97 land their own
sentences, exactly as 82 landed its own.

### The fourth site

`docs/DEFERRED.md:145` carried the same `scan.py` claim, which ticket 87 found
and left here. Swept in the same edit: the paragraph now names one file and
says in a parenthesis that the second was removed, so the next reader does not
re-derive the history. A pass over `CLAUDE.md`, `docs/*.md` and `README.md`
finds **no** surviving "front door" and no other `eos/mixed/scan.py` — the one
remaining `scan.py` is `nucleation/analysis/scan.py`, a different file in a
different repository.

### Done-condition

`grep -rln "ticket 22|22-phase5-claudemd" issues/` returns ten files and **all
ten are `Status: resolved`**, so no open ticket still cites ticket 22 as its
vehicle. Each of the seven rulings — 70, 29, 84, 86, 87, 92, 90 — now ends with
a line naming the section its sentence landed in.

**Noticed** (map hard rule — to the Stage 7 report, not fixed here):
`docs/STRUCTURE.md`'s per-model walkthrough cites CLAUDE.md sections by stale
numbers — "§5's six names" for the species flags (that is §4) and "the §4 name
lowercased" for the mode names (that is §3). Its `par`-first citation to §5 is
correct. Prose outliving the thing it points at, one layer over from what this
ticket swept.
