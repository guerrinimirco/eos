# Which models and subsystems get a notebook at all

Type: grilling
Status: resolved
Blocked by: 02
Parent: ../map.md

## Question

The prompt's three notebooks cover nine models. The repo has thirteen model
subpackages plus `astro/`.

- **`abpr`** — Stage 2 asks explicitly whether it belongs in the quark notebook
  and why. Note it has no `table.py` (ticket 07 asks whether that is "the physics
  has no such part" or a gap), which bears directly on whether it can carry a
  notebook section built on `eos_table`.
- **`mixed`** — the composite engine, and the one named §1 exception that imports
  `eos.astro.tov`. `DD2vMIT_general1oPT.ipynb` currently covers this ground and
  ticket 03 removes it. Does a replacement exist, and in which notebook?
- **`astro/tov`** (including the RNS rotating backend) and **`astro/gmode`** —
  no usage notebook; `docs/REFACTOR_PLAN.md:110` once planned them.
- **`zlvmit`** — out of scope per the map; confirm.

Resolved when each of these has a ruling: a section in one of the three
notebooks, a fourth notebook (a new ticket), or explicitly nothing. Whatever
becomes "a fourth notebook" graduates out of the map's Not-yet-specified section
into its own tickets.

## Answer

**Four notebooks, not three. `mixed` gets the fourth; `abpr` gets a companion
panel; `astro/tov`, `astro/gmode` and `zlvmit` get nothing, each for a stated
reason.**

### Two of the ticket's own premises were wrong

**`abpr` already has `eos_table`.** `abpr/api.py:146` defines the full §5
surface — `eos_point`, `eos_table`, `eos_response`. What it lacks is
`table.py`, and its `__init__.py` says why: *"Nothing here iterates"* — P, n_B
and eps are polynomials in mu and the three inverse maps mu(n_B), mu(P),
mu(eps) are closed forms. There is no warm-started sweep to put in a grid
driver, so §5's "the names are mandatory; the existence is conditional" is
satisfied exactly as written. Nothing about the notebook question turns on it.
**This also answers [ticket 07](07-naming-sweep.md)'s open row: the absence is
"the physics has no such part", not a gap.**

**`astro/tov` and `astro/gmode` have no `eos_table` of any kind** — they consume
tables. A notebook section for either would have to attach to whichever
notebook produced the table it eats, which is what settles Q7/Q8 below.

### The rulings

| Unit | Ruling | Why |
|---|---|---|
| `abpr` | **companion panel inside `quark_eos`** | not a fifth peer in the knobs cell |
| `mixed` | **a fourth notebook, `notebooks/hybrid_eos`** | tickets 57a/57b below |
| `astro/tov` | **no notebook** — covered in situ | already in `hadronic_eos` figs 3–4, now also ends `hybrid_eos` |
| `astro/gmode` | **nothing, and it is a named gap** | no `eos_table`; no §11 rule forces one |
| `zlvmit` | **out of scope, confirmed** | §5 exempt; but see the scope line below |

**`abpr`.** One figure inside `quark_eos`, against `alphabag` at CFL and T = 0,
showing the O(m_s^4) difference `abpr/verify/run_full_check.py` already
measures — the two are driven as a matched pair through alpha_s = pi/2 (1 - a4).
It is CFL-only and T = 0-only, so as a fifth peer in the knobs cell it would trip
ticket 04's unsupported-combination pattern on nearly every cell; as a companion
panel its narrowness is the subject rather than the noise.

**`astro/tov`.** `mixed/hybrid.py` and `mixed/scan.py` already import
`eos.astro.tov` — the one §1 exception — and `HybridResult.table.to_tov()` is
the declared contract into the solver. So `hybrid_eos` ends on a mass–radius
curve without breaking layering, and TOV is exercised through its public
contract in two notebooks. A notebook whose subject is the TOV *solver* — crust
choices, the RNS rotating backend, tidal deformability — is a different
deliverable and is future work, not this map.

**`zlvmit`, with the scope line stated.** Out of scope means *no new notebook
and no conformance work*. It does **not** touch
[ticket 41](41-corrupt-notebooks.md): `ZLvMIT_hybrid.ipynb` is on Stage 0's KEEP
list and is currently unopenable JSON, and repairing it stays in scope. The user
confirmed the ZLvMIT notebook should work again.

### The fourth notebook: `notebooks/hybrid_eos`

Subject: hybrid constructions. How to choose the hadronic and quark phase, how
to set both parameter sets, how to generate tables, how to plot, and how to end
on M–R.

**Pairings.** `eos/mixed/adapters.py` ships four hadronic phases (`dd2_phase`,
`sfho_phase`, `did_phase`, `zl_phase`) and four quark (`vmit_phase`,
`alphabag_phase`, `njl_phase`, `ccdm_phase`) — sixteen pairings — plus
`enjl_branch_pair`. The notebook does not run sixteen. It runs:

- **DD2 + vMIT** as the headline, end to end (tables, figures, TOV pass);
- **DID + NJL** and **DID + CCDM** as the swap demonstration.

DD2 + vMIT is the headline because it is exactly what the held 46 MB of tables
were, which makes this notebook a *checkable* replacement for what it retires
rather than a promise. The swap cases change both sides of the pair at once,
which exercises the phase-adapter contract harder than swapping one. How deep
the two swap cases run — full grid or a short one — is a runtime call for the
build ticket, with a converged table the floor.

**Two tickets, no benchmark.** [Ticket 58](58-hybrid-skeleton.md) is skeleton, knobs,
the pairing choice and the tables; [ticket 59](59-hybrid-figures.md) is figures, the
TOV pass and the swap cell. No benchmark ticket: the mixed-phase solve is the
slowest thing in the repo, and a benchmark section would double an already
expensive notebook's runtime for a number nothing is being optimised against.
Both blocked by 04 and 05, like `quark_eos`.

**ENJL belongs to the `enjl` notebook, not this one.** `enjl_branch_pair` lives
in `eos/mixed/adapters.py` and §5 lists it among the shipped adapters, so the
two subjects overlap on exactly that object. The two branches are two branches
of *one functional*, not two models being coupled — the physics is ENJL's and
`eos/mixed` is the machinery it is expressed through. `hybrid_eos` states the
boundary in one line and points at `enjl`, so it is written down in both places.

### The Destination is amended: four notebooks

Three is the prompt's number and it is a **floor**, not a ceiling — an existence
claim, and [ticket 02](02-notebook-grouping.md) already replaced §11's "one per
model" with a rule rather than a count. The fourth is recorded as **this map's
own addition, not the prompt's**, because otherwise it is the thing that gets
dropped when a session runs long — and it is the only producer of the tables
that were just retired. Further notebooks (TOV/RNS, gmode, per-model deep dives)
are expected later and are outside this map.

### `notebooks/eos_tables_DD2vMIT/` — moved, not deleted

The user authorised deletion and reported `output/` renamed to `output_old/`.
Deletion was not the cheapest way to get what the ruling needs, so the folder was
**moved** instead:

    notebooks/eos_tables_DD2vMIT  ->  output_old/eos_tables_DD2vMIT_from_notebooks

`notebooks/` now holds only the two `zlvmit` files, which is the whole of what
Stage 0 wanted, and the 32 tables and 42 published figures survive at zero cost.
Ticket 03's held-until condition is therefore **discharged without an
irreversible act**: `hybrid_eos` can be compared against the held figures rather
than merely asserted to replace them. Nothing was in git — 0 tracked files —
so this was the only form of undo available.

**Two consequences of the `output/` rename, reported not fixed beyond one
restore.** `output_old/` is 24 GB and is **not** gitignored (`.gitignore:37`
ignores `output/`), so it shows as untracked in every concurrent session. And
the rename carried `output/_audit/` with it — the directory this map's
added-failure rule names by path. It was **copied back** to `output/_audit/`
(116 kB, nine files) so tickets comparing against
`output/_audit/pytest_after_ticket45.txt` still find it.

Status: resolved.
