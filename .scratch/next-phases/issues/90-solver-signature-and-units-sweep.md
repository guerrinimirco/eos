# One solver signature, one unit system at the boundary

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

Ruled by [ticket 81](81-second-default-solver-kwargs.md), sections 2, 4 and 5.
Split out because **no value moves** in any of it: the whole gate is a green
suite plus unmoved baselines. The one regeneration it carries renames keys and
leaves every number identical.

§13 exists so "a physicist who has read one model can read the next without a
translation table". `solver.py` currently needs one. Three argument orders:
`(par, n_B, flags, ...)` in dd2/sfho/did, `(n_B, T, params=None, ...)` in
zl/vmit/alphabag, `(n_B_fm, Y_C, T=0.0, par=None, flags=None, ...)` in
njl/ccdm/enjl. `par` is required in two models, optional in seven, and spelled
`params` in three.

## Work

**Signatures**, every model's `solver.py`:

1. `par` first and required. `params=` -> `par=` in zl/vmit/alphabag.
2. `n_B_fm` -> `n_B` (87 sites). Natural-units working variables displaced by
   the rename take `_nat`, the convention `dd2/solver.py:159` already uses.
   Four functions hold both names at once and need care:
   `enjl/solver.py:186,520`, `enjl/verify/run_full_check.py:130`, and
   `dd2/solver.py:101` where the sense is REVERSED (`n_B` is the fm one).
3. The rest of the `_fm` family: `n_C_fm` (12), `n_S_fm` (8), `eps_fm` (7),
   `P_fm` (7), `s_fm` (6), and `n_b_fm` — which is a result field, and whose
   lowercase `b` also violates §2's B-for-baryon convention.
4. `zl`/`vmit`/`alphabag` solvers take `flags: SpeciesFlags` (required).
   `include_photons`, `include_gluons` and `include_thermal_neutrinos` are
   deleted into it — the flags dataclasses already carry all three fields.
   ~40 lines of pass-through in `alphabag/solver.py` go with them.
   `include_electrons` -> `leptons`, which stays a SEPARATE named argument
   (§5); it is not a species flag and never enters `SpeciesFlags`.
   72 call sites outside the three `solver.py` files: zl 19, vmit 21,
   alphabag 32.

**Units at the boundary** — §5 is already the rule; three models break it:

5. Drop the natural-units record from every public result: `njl` and `ccdm`'s
   `.state`, `enjl`'s `BetaPoint.point`. Lift what callers need onto the outer
   point in fm. 24 sites across njl/ccdm reach for `n_*` (10),
   `euler_residual` (5), `n_q` (3), `P` (1); `euler_residual` is dimensionless
   and stays reachable through an explicitly internal path.

   Why not a rename: the inner record differs in CONTENT as well as units —
   it is matter-only, so `njl .state.P / hc3 = 146.854334` against an outer
   `146.939710`. A caller who spots the unit problem and corrects by `hc3`
   still gets a wrong answer, silently, by 0.085376 MeV/fm^3.

## Gate

- **No value moves anywhere.** Every `test/baseline/*.npz` unmoved at
  rtol = 1e-10, `enjl.npz` excepted below.
- `test/baseline/enjl.npz`: the `n_b_fm` -> `n_B` rename changes **234 keys**.
  Verify by comparing old and new arrays key-for-key after mapping the name —
  every value identical, or it is not a rename.
- Full suite green (§12). `verify/` green for every model touched.
- `zl`, `vmit` and `alphabag` baselines must stay green WITHOUT regeneration:
  because `flags` is required, the generator's 6 call sites name
  `photons=True` and reproduce the frozen rows exactly. A moved row there
  means the flags were mis-translated, not that the numbers were due to move.

---

## Note from [ticket 82](82-alphabag-gluons-default.md) (2026-08-26)

**"No value moves" does not hold for alphaBag.** Section 2's deletion of
`include_photons` / `include_gluons` / `include_thermal_neutrinos` into the
flags moves `test/baseline/case_alphabag`, because that generator calls the
raw solvers and names none of the three; once they read the flags it picks up
the defaults, and all three alphaBag defaults are now `False` — `photons` and
`thermal_neutrinos` since [ticket 65](65-species-flag-defaults.md), `gluons`
since [ticket 82](82-alphabag-gluons-default.md).

Measured through `eos_point` at n_B = 0.8, for the gluon sector alone:

    beta.T0     unchanged — every thermal sector vanishes at T = 0
    beta.T10    P  -1.465838e-03 MeV/fm^3
    beta.T30    P  -1.187329e-01 MeV/fm^3

`zl` and `vmit` carry the same shape for `include_photons`. So this ticket
needs a baseline regeneration and its own key-by-key diff after all, or the
three affected models split out the way [ticket 89](89-dd2-honours-species-flags.md)
was — which was the right instinct applied to the wrong model.
