# Does zl get an nmp.py, or is its absence recorded as deferred?

Type: grilling
Status: open
Blocked by: 07
Parent: ../map.md

## Question

[Ticket 07](07-naming-sweep.md) found that `eos/zl` having no `nmp.py` is a real
gap rather than an absence of that physics, which contradicts what
`docs/NEXT_PHASES_PROMPT.md` assumes ("`zl` has no `nmp.py`; state that in the
notebook rather than faking it").

The evidence: ZL is a nucleonic functional whose six parameters exist precisely to
set nuclear-matter parameters (`eos/zl/parameters.py:1-10`), and
`parameters.py:64-65` quotes `n_sat = 0.15951`, `E_sat = -16.00`,
`K_sat = 250.2`, `E_sym = 30.85`, `L_sym = 41.26` as "measured from the code at
T = 0". But grepping `eos/zl/` for `compute_nmp`, `invert_nmp`, `esym`,
`energy_per_baryon` or any saturation solve returns **nothing** — those numbers
were computed by something outside the package and pasted into a docstring, and
no `verify/` check reproduces them. `docs/DEFERRED.md` records nothing about it.

§5 names only `dd2` and `sfho` as the models with a nuclear sector, but
`eos/did/nmp.py` exists and `did` is not in that list either, so the list reads as
illustrative rather than exhaustive.

Three ways this can go:

- **Add `eos/zl/nmp.py`** with at least the forward map — zl can already solve
  symmetric matter through `thermo_from_n(n_B, 0.5, 0)` — and pin the six quoted
  numbers in `eos/zl/verify/`. The inverse map is a separate question.
- **Record the gap in `docs/DEFERRED.md`**, which is what §3/§5 require of any
  gap that stays open, and leave the docstring numbers as unverified provenance.
- **Rule the docstring numbers wrong to quote at all** and remove them, since a
  number no check reproduces is the thing the ledger exists to prevent.

Whichever way it goes, it settles what [ticket 12](12-hadronic-skeleton.md) says
about zl in the notebook's parametrisation section.
