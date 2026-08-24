# Does zl get an nmp.py, or is its absence recorded as deferred?

Type: grilling
Status: resolved
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

## Measurement — the quoted numbers are right, and the forward map is ~40 lines

The five values in `eos/zl/parameters.py:64-65` were checked by computing them
from the code, at T = 0, by finite differences on `thermo_from_n`:

| quantity | computed | quoted | delta |
|---|---|---|---|
| `n_sat` | 0.15951 | 0.15951 | 0.00000 |
| `E_sat` | −15.99648 | −16.00 | 0.00352 |
| `K_sat` | 250.16956 | 250.2 | −0.03044 |
| `E_sym` | 30.84803 | 30.85 | −0.00197 |
| `L_sym` | 41.26408 | 41.26 | 0.00408 |

Every one agrees to within the finite-difference accuracy of the stencil used
(h = 2e-4 fm^-3). So the docstring is **not** wrong — it is merely
**unverified by anything in the repository**, which is the actual gap.

**And the forward map is cheap.** `thermo_from_n(n_B, Y_C, T, par)` returns `e`,
so E/A follows directly, saturation is a 1-D root on its derivative, and
`E_sym`/`L_sym` come from the `Y_C` curvature at `delta = 1 - 2 Y_C`. The whole
`compute_nmp` is on the order of forty lines with no new physics — everything it
needs already exists in `thermodynamics.py`. That removes the main argument
against adding `eos/zl/nmp.py`: it is not a new capability, it is a name for a
calculation the model can already do and currently does not expose.

Three consequences for the ruling:

1. The "add `nmp.py`" option is much cheaper than it looked, and it comes with
   five verified numbers to pin in `eos/zl/verify/`.
2. The "record in DEFERRED" option is now weak — a gap whose closure is forty
   lines of existing physics is not really a deferral.
3. The "remove the numbers as unverifiable" option is dead: they verify.

The **inverse** map (`invert_nmp` / `from_nmp`) is a separate question and not
settled by this measurement. ZL has six parameters against five NMPs, so the
inversion is underdetermined without a further condition — the same structural
issue §5 describes for DD2, which closes its isoscalar sector with a
cross-constraint. Whether zl gets the inverse too should be decided separately
from whether it gets the forward map.

Note for [ticket 12](12-hadronic-skeleton.md): the notebook's parametrisation
section currently plans to say "zl has no `nmp.py`". If the forward map is added
that line changes; if only the inverse is refused, the notebook says *that*
instead, which is a more useful thing for a reader to know.

## Answer

**Forward map added; inverse raises.** `eos/zl/nmp.py` now carries
`compute_nmp`, and `invert_nmp` / `from_nmp` raise `NotImplementedError` naming
the reason.

The forward map returns, at T = 0:

    n_sat  0.15951      E_sat  -15.99648      K_sat  250.174
    E_sym 30.84803      L_sym   41.27034
    Q_sat -352.872      K_sym  -88.421

The five published values are pinned in `eos/zl/verify/run_full_check.py` at the
published precision (E_sat is tightest, at 70% of its tolerance), and
`test/zl/test_zl_nmp.py` carries ten tests. `parameters.py`'s docstring now
points at the map that reproduces it instead of asserting the numbers on trust.

**One thing the measurement turned up that the ticket did not anticipate:** the
symmetry energy had to be the `beta -> 0` curvature, **not** the full step to
pure neutron matter with a Richardson correction that `eos/did/nmp.py` uses.
That estimator returns 30.776 and 41.124 here — a real difference carrying
`beta^4` contamination, not numerical noise. DID needs the full step because its
E/A difference at small asymmetry sits in noise; ZL's does not, so ZL takes the
definition directly. Both `nmp.py` docstrings now say so, and a test pins ZL's
step-independence so that assumption fails loudly if it ever stops holding.

**The inverse is refused, not deferred.** Six couplings against the five NMPs of
§5's list leaves a one-parameter family, and unlike DD2 — which closes its
isoscalar sector with `f''_sigma(1) = f''_omega(1)` plus a pinned shape
coefficient — nothing in Constantinou et al. singles out a member. The docstring
names the two ways to close it (impose a sixth datum, or hold one coupling
fixed). §3's rule applied to a map rather than a mode: it raises, it says which,
and it is never a silent no-op.

Ticket 12's line about zl changes accordingly — see below.

Status: resolved.
