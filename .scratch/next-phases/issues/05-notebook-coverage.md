# Which models and subsystems get a notebook at all

Type: grilling
Status: open
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
