# Rename did's and dd2's phase-adapter surface to thermo_from_mu

Type: task
Status: resolved
Blocked by: 36
Parent: ../map.md

## Question

[Ticket 36](36-quark-engine-documents.md) settled the name ticket
[10](10-rename-approvals.md) deferred to `mixed.tex`: the §5 phase-adapter
surface — `(baryon potential, mu_C, mu_S, T) -> PhaseThermo`, solving the
phase's own internal self-consistency — is **`thermo_from_mu`** in every model,
and a lower evaluation layer that additionally takes the solved mean fields is
**`thermo_from_fields`**.

`sfho` carries that ruling under [ticket 45](45-rename-sfho.md). `did` is the
third model with the split, and [ticket 42](42-rename-internal.md) — which
covered `eos/mixed` and `eos/did` — closed before the name was settled, so it
has no ticket.

**dd2's half is also still open, and belongs here.** Ticket 44 carried the same
"Added by ticket 36" instruction and its list of 19 renames does not include
this one: `eos/dd2/thermodynamics.py:571` is still `thermo_at_potentials`, with
no `thermo_from_mu` in the package at all. Found while working ticket 45.
dd2 is the easy case the ticket-44 text already described — one name, no lower
layer to re-spell — but its call sites reach further than did's:

    eos/dd2/thermodynamics.py:17, 571      the def and its module docstring
    eos/mixed/adapters.py:52, 243, 280     a BARE module-level import, not an
                                           alias like the sfho and did ones
    eos/sfho/thermodynamics.py:566         a cross-reference in a docstring
    eos/enjl/thermodynamics.py:741         the same
    test/dd2/test_thermodynamics.py        4 sites
    docs/REFACTOR_PLAN.md:288              names it in prose
    docs/DEFERRED.md:769                   names it in prose

The bare import at `adapters.py:52` is the one to watch: once dd2's surface is
`thermo_from_mu`, that file will hold a module-level `thermo_from_mu` beside two
function-local aliased imports of the same name from sfho and did — shape 3 of
the three-shape collision check, which is exactly what broke
`test/mixed/test_hybrid_modes.py` under ticket 44. Alias it (`_dd2_at_mu`)
like its five neighbours.

`did/thermodynamics.py:542` is `thermo_at_potentials`, the surface;
`did/thermodynamics.py:358` is `thermo_from_mu(par, flags, fields,
mu_tilde_B, mu_C, mu_S, T, matter=None)`, the layer beneath it. **Rename the
lower one to `thermo_from_fields` first**, or the second rename lands on an
occupied name — the pattern that cost ticket 42 twelve silently-red tests
(`mixed/api.py`'s local `solve`) and ticket 43 five collisions
(`vmit/table.py`'s `warm_start`, plus four local `default_guess` bindings the
AST check cannot see).

Call sites: `eos/mixed/adapters.py:797, 813` aliases it as `_did_at_mu`, and
`did`'s own `solver.py` / `verify/`. Run the AST collision check tickets 43-45
carry before moving anything, and `test/baseline/` must not move at
rtol = 1e-10 — a rename that changes a number is not a rename.

---

## Resolution

Done. Three renames, in the order the ticket required — did's lower layer first,
so the surface rename does not land on an occupied name.

    did  thermo_from_mu        -> thermo_from_fields   (takes the solved fields)
    did  thermo_at_potentials  -> thermo_from_mu       (the section 5 surface)
    dd2  thermo_at_potentials  -> thermo_from_mu       (the section 5 surface)

Commit `2891715`. All ten models now spell the surface `thermo_from_mu`, verified
by introspecting every package rather than by reading; `sfho` and `did` are the
only two carrying a `thermo_from_fields` beneath it.

### The ticket's collision diagnosis was wrong, and the real one is three times bigger

The ticket predicted `eos/mixed/adapters.py` would hold "a module-level
`thermo_from_mu` beside two function-local **aliased** imports of the same name
from sfho and did". That cannot collide: sfho's is `_sfho_at_mu` (:705) and
did's is `_did_at_mu` (:815), and aliasing is precisely what makes them safe.

Sandboxed the naive rename in scratchpad before touching the repo — copied
`adapters.py`, applied only the bare dd2 rename, ran the three-shape checker:

    2 local binding adapters.py:426   'thermo_from_mu' rebound inside enjl_phase()
    2 local binding adapters.py:1081  'thermo_from_mu' rebound inside njl_phase()
    2 local binding adapters.py:1231  'thermo_from_mu' rebound inside ccdm_phase()

The real colliders are three **bare** function-local imports the ticket never
names. Shape 2, not shape 3 — and shape 3 could never have fired, because it
only inspects module level and every competing import here is function-local.
Aliasing dd2's to `_dd2_at_mu` returns the tree to CLEAN. All three were scoped,
so nothing was ever live-broken; this is the fragile shape, not a live bug.
Reported, NOT widened into a fix for enjl/njl/ccdm.

### `\b` is not supported by BSD sed

The first rename pass was `sed -i '' 's/\bthermo_from_mu\b/.../'` and was a
SILENT NO-OP — zero substitutions, exit 0. Caught only because the pass was
followed by a grep instead of being trusted. Every subsequent rename here used
`perl -pi -e`.

The working BSD spelling is the word-boundary CLASS, confirmed on this machine
against a string carrying both the bare name and a `_x` suffix:

    sed 's/\bNAME\b/NEW/g'              -> no-op, exit 0        (GNU only)
    sed 's/[[:<:]]NAME[[:>:]]/NEW/g'     -> correct, and leaves NAME_x alone
    perl -pe 's/\bNAME\b/NEW/g'         -> correct, same

**The exposure was checked and comes back GREEN.** Tickets 42-45 all describe
`sed`-based renames without flagging this, so every old name those tickets were
meant to remove was grepped across `eos/`: `thermo_at_potentials`,
`get_sfho_nucleonic`, `get_sfhoy_fortin`, `get_sfhoy_star_fortin`,
`get_sfho_2fam_phi`, `get_sfho_2fam`, `get_all_parametrizations`, `_GUESS_KIND`,
`from_dd2_defaults`, `solve_octet`, `Parametrization`. All clean. Two hits, both
deliberate and neither a miss: this ticket's own past-tense mention in
`mixed.md:165`, and `sfho/parameters.py:662`'s
`print(f"Parametrization: {params.name}")` — a printed label in a demo block,
prose in output rather than a symbol, cosmetic and ticket 45's territory.
So the trap is real and cost this ticket one pass, but **it bit no earlier
ticket** — verified by grep, independently, from both sessions.

### Five document passages corrected

The rename falsified prose in `dd2.md`, `sfho.md`, `sfho.tex`, `mixed.md` and
`mixed.tex` — each asserted either that dd2 was "the outstanding one" or that
three models "currently" spell the surface `thermo_at_potentials`. Left alone
they would state something untrue about the code, which is the same defect class
as the two docstring cross-references the ticket DOES list
(`sfho/thermodynamics.py:566`, `enjl/thermodynamics.py:741`). Corrected here.
`mixed.md` / `mixed.tex` keep a deliberate past-tense mention of the old name,
because that paragraph exists to record the ruling.

### Evidence

- **904 probe values bit-identical by exact `==`, not rtol.** A deep walk over
  the whole returned structure (dataclasses, dicts, arrays — not a hand-listed
  set of field names) for dd2 and did across `beta_eq_neutrinoless` and
  `fixed_YC`, T = 0 and 20 MeV, n_B = 0.16 and 0.4, plus 24 direct calls on the
  renamed surface. Zero keys added, zero dropped, zero moved. Re-run after the
  indentation fixes and identical again.
- **Full suite byte-identical to its before-image.** `12 failed, 1638 passed,
  15 skipped`, same node ids, and `diff` of the 121 `^E ` assertion lines is
  EMPTY. 0 added failures, 0 cleared.
  Before `output/_audit/pytest_after_ticket56_py314.txt`,
  after `output/_audit/pytest_after_ticket48_py314.txt`.
- **Baseline gate**, in ticket 56's corrected wording — "no SURVIVING baseline
  value moved": the six `test_baseline[*]` failure bodies are byte-identical, and
  `test/baseline/` was never written to (mtimes unchanged; the four at Aug 25
  15:17 are ticket 56's, predating this ticket's first edit).
- **verify PASS**: `did` all 13 checks (euler 4.53e-15, residual_gate 2.25e-14),
  `dd2` all 6 (golden SNM 1.40e-05, backend parity 4.40e-14), `mixed` all 9
  (euler/HVH 9.67e-15, TOV M_max 2.340 unchanged).
- **shadowcheck CLEAN** before and after, whole tree.

### Interpreter, and what that qualifies

Run on `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`
(3.14.2 / numpy 2.3.5 / scipy 1.17.0 / numba 0.63.1), chosen because
`output/_audit/` is a 3.14 audit trail and comparability is what a rename gate
needs. This is NOT a vote on [ticket 57](57-canonical-stack.md). If anaconda is
ruled canonical, the claim restates without weakening: no rename-attributable
number moved on the stack the audit trail was built on, and the 904-point
bit-identical probe is stack-independent evidence in any case.

### One measurement thrown away

The first post-rename suite run was killed and discarded. The shorter names left
continuation lines hanging, and I re-aligned them AFTER the run had started, so
it spanned a source edit — the exact contamination ticket 45 paid for. Restarted
on a frozen tree. The reported numbers come from that second run.

### Not done here

- `test/` is gitignored (`.gitignore:75`), so the edits to
  `test/did/test_did_thermodynamics.py` (6 sites) and
  `test/dd2/test_thermodynamics.py` (4 sites) live ONLY in the working copy —
  the same hazard the map records for tickets 39, 40, 45 and 56. Anyone
  reconstructing `test/` reintroduces `thermo_at_potentials` and the suite goes
  red at import.
- `test/sfho/test_thermo_from_mu.py` is named after the surface; did's and dd2's
  equivalents are not. Not in this ticket, not renamed.
- The tree was NOT clean: ticket 20's `fe68f20` + `7c5b7a9` were present in both
  the before-image and this run, and collection is 1665, not the map's 1663.
  Both measurements carry it equally, so the diff is still valid.
