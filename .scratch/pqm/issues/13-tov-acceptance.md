# TOV acceptance: M_max and R_1.4 against the NJL hybrid

Type: task
Status: open
Blocked by: 11
Parent: ../map.md

## Question

`docs/csc_bag_mapping.md:474-476`, acceptance criterion 5:

> *"**The M-R sequence.** `M_max` and `R_1.4` from the fitted EoS within a few
> percent of the NJL ones, through `eos.astro.tov`. This is the application
> test and it is the one that decides."*

It has never been implemented. It closes the map because it is the only test
that asks the question the model exists to answer: does replacing `eos.njl`
with `eos.pqm` change the star?

It is also the test that can fail after every other one passes, and the reason
is structural: a stellar sequence integrates the EoS, so a systematic few-tenths
of a percent that the pressure benchmark shrugs at can accumulate, while a
large local error somewhere the star does not reach costs nothing at all. The
error measure that matters here is not the one milestone 1 gated on.

### Two sequences, not one

**1. The pure quark sequence** -- `pqm` alone, beta equilibrium, T = 0 --
against `eos.njl` alone on the same grid. This isolates the quark EoS. It may
not produce a physical star at all (a bare quark branch starting at ~2 n_sat
has no crust and no low-density matter), so it is a comparison of two curves
rather than a claim about an object.

**2. The hybrid sequence**, which is the real test: DD2+pqm against DD2+njl
through `eos/mixed`'s `hybrid_table`, whose `.table.to_tov()` is the documented
contract into `eos.astro.tov` (`eos/mixed/api.py:227-265`). Ticket 11 already
built the pair and compared rows; this ticket integrates them.

Compare: `M_max`, `R_1.4`, `R_2.0` if the sequence reaches it, the central
density at `M_max`, and the tidal deformability `Lambda_1.4` -- which is a
higher derivative of the same EoS and is the sharpest of the set.

### Section 8's gate is a precondition, not a result

*"Any EoS table DELIVERED to a structure solver has P non-decreasing in n_B and
0 <= c_s^2 <= 1 ... the check runs before integration, returning a status
rather than a meaningless mass."*

So the gate runs FIRST, on the constructed table, and a violation returns a
status. This is where ticket 08's construction is actually exercised: a raw
branch may violate monotonicity inside the transition window, and if the
construction did not resolve it, TOV will produce a number that looks like a
mass and is not one. **A TOV run that did not check first is not evidence.**

### Two traps recorded in this repo

**Never time anything on battery** -- the duty cycle drops to ~5%, and
`caffeinate -s` needs AC. TOV sequences are the longest-running thing on this
map.

**A `hybrid_table` can return a PURE HADRONIC table** where the quark phase
never wins, and it does so without complaint. That is recorded for the DD2Y+NJL
and CCDM hybrids, where CCDM has no transition at all. If `M_max` comes back
suspiciously close to the pure DD2 value, check `chi` along the sequence before
concluding anything about `pqm`.

### What a failure would mean, and it is not one thing

- **Both sequences agree to a few percent** -- the map closes, and the
  surrogate is usable for the inference and simulation work it was built for.
- **The pure quark curves agree and the hybrids do not** -- the mixed solve is
  amplifying the fit error, and the place to look is the window edges, where
  ticket 11 already compared `n_onset`/`n_offset`.
- **`M_max` agrees and `R_1.4` does not** -- `R_1.4` is set by matter well
  below the quark onset, so a disagreement there is the HADRONIC side or the
  crust, not `pqm`, and the ticket must say so rather than blaming the quark
  EoS.
- **`Lambda_1.4` misses while `M_max` and `R_1.4` hold** -- expected if
  anything; it is the highest derivative in the set. Report it, and let the
  target be informed by it rather than pretending it was predicted.

## Gate

- Both sequences run through `eos.astro.tov`, with the section 8 check run and
  PASSING before integration, its status reported.
- **`M_max` and `R_1.4` within 2%** of the NJL equivalents, hybrid sequence.
- `Lambda_1.4` and the central density at `M_max` reported, with a target set
  by what is measured rather than in advance.
- `chi` along the sequence reported, so a pure-hadronic table cannot be
  mistaken for agreement.
- Timing stated, on AC, with the interpreter and library versions named.
- **A closing verdict for the map**: is `eos.pqm` usable as a drop-in for
  `eos.njl` in the applications it was built for -- Bayesian inference over
  `(eta_D, G_V0/G_S)`, and `(n_B, Y_C, T)` tables for mergers and supernovae --
  and if it is usable only within a stated domain, what that domain is.
- Anything unmet recorded in `docs/DEFERRED.md`.
