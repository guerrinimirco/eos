# `zl.invert_nmp`: the closed form exists and was verified

Type: task
Status: resolved (2026-08-28)
Blocked by: -
Parent: ../map.md

## Question

`zl/nmp.py` says the inverse map is "deliberately absent" and `invert_nmp`
raises. **The closed form exists**, was supplied by the user from an earlier
derivation, and was re-derived and checked against this repository's own
forward map this session.

With kinetic parts subtracted (`Kb = K_sat - K_sat,K` etc.),
`X = Eb n0 + P_K,0` and `D = (9 Eb + Kb) n0^2 + 9 P_K,0 n0`:

    a0     = [Kb Eb n0^2 - 9 P_K,0 (Eb n0 + P_K,0)] / D
    b0     = 9 X^2 / D
    gamma  = -Kb n0 / (9 X)
    a1     = [3 g1 Sb - Lb + 3 a0 (g1 - 1) + 3 b0 (g1 - gamma)] / [3 (g1 - 1)]
    b1     = [3 Sb - Lb + 3 b0 (1 - gamma)] / [3 (1 - g1)]

with `gamma1 = g1` a FREE input (it tunes the high-density isovector
behaviour and no saturation-density observable constrains it).

Verified, target {n0 = 0.16, E_sat = -16, K_sat = 250, E_sym = 31.6,
L_sym = 43, gamma1 = 2.45}:

    recovered  a0=-96.6555  b0=58.8619  gamma=1.39854  a1=-25.1985  b1=7.1850
    published  a0=-96.64    b0=58.85    gamma=1.40     a1=-25.19    b1=7.18

and round-tripped through `zl.nmp.compute_nmp`: n_sat to 1.2e-14, E_sat to
5.6e-13, E_sym to 5.4e-07, K_sat and L_sym to 9.5e-03 and 6.4e-03 — the last
two being the forward map's own stencil, not the algebra.

**Why this matters beyond one model.** It makes `zl` the only model in the
repository whose inversion has no seed, no basin, no restart count and no
lottery — see [ticket 105](105-dd2-isoscalar-conditioning.md) and
[ticket 93](93-invert-nmp-basin-lottery.md) for what the alternative costs.
For an inference run that is a different class of object.

## Work

1. `zl/nmp.py::invert_nmp` and `from_nmp`, closed form, replacing the raise.
   `gamma1` an explicit argument, no default that hides the choice.
2. Kinetic pieces from the model's own code with the interaction switched off
   (`replace(par, a0=0, b0=0, a1=0, b1=0)`), not a second transcription of the
   Fermi integrals (§7).
3. **State the two conventions in the docstring, both of which silently give
   wrong answers if mixed:**
   - the rest-mass bookkeeping — `Eb = E_sat - E_sat,K` holds only when both
     are binding energies or both include the mass; the `+ m_H` in the
     original derivation belongs to the first reading;
   - which `E_sym`. This repository's `symmetry_energy` is the quadratic
     coefficient; Constantinou et al. use `E(n0, 0.5) - E(n0, 0)`. That is
     exactly why the shipped set carries `a1 = -26.06, b1 = 7.34` where the
     same NMPs through this inversion give `-25.19, 7.18`. A round-trip test
     that does not fix the convention will fail for the wrong reason.
4. **Two ZL-specific facts worth stating in `zl.md`/`.tex`**, both verified:
   - `Q_sat,V = 3 (gamma - 2) K_sat,V` exactly (checked to 6 digits,
     -1.804378 both ways). So Q_sat is NOT free in ZL: once
     {n_sat, E_sat, K_sat} fix gamma and b0, Q_sat is determined. A prior over
     (K_sat, Q_sat) in ZL lives on a curve, not in a plane.
   - the isovector sector has THREE knobs (a1, b1, gamma1), so ZL is the one
     model here that can impose K_sym, through
     `K_sym,V = 9[gamma1(gamma1-1) b1 - gamma(gamma-1) b0]`. Offer it as an
     option; leave gamma1 free by default.

## Gate

Round trip on the published set to the forward map's own stencil; the XOA
numbers above reproduced; `test/baseline/zl.npz` unmoved (this adds a
constructor, it changes no evaluation); one test asserting the Q_sat
rigidity, since it is the identity most likely to be broken by a later edit.

## Resolution

Built 2026-08-28. Execution as scoped: the closed form was already verified,
and the two convention traps were the whole risk. Both are now stated in the
code before any test asserts anything, so nothing here failed for the wrong
reason.

### The algebra, re-derived rather than transcribed

The ticket's formulae were re-derived from the functional before being typed,
and they check out exactly. In symmetric matter `V/n = a0 u + b0 u^gamma`, so
the three isoscalar conditions at `u = 1` are `a0 + b0 = Eb`,
`n0 (a0 + gamma b0) = -P_K,0` (total pressure vanishes) and
`9 gamma(gamma-1) b0 = Kb`; eliminating gives `(gamma-1) b0 = -X/n0` and
thence every line of the ticket, including the `a0` numerator
`Kb Eb n0^2 - 9 P_K,0 X`. The isovector pair falls out of
`S_V(u) = (a1-a0) u + b1 u^gamma1 - b0 u^gamma` the same way. The
implementation reproduces the ticket's numbers digit for digit:

    a0=-96.6555  b0=58.8619  gamma=1.39854  a1=-25.1985  b1=7.1850

and the round trip through `compute_nmp` returns n_sat to 1.2e-14, E_sat to
5.6e-13, E_sym to 5.4e-07, K_sat to 9.5e-03 and L_sym to 6.4e-03.

**The unstated premise, now stated: `n0 := n_sat`.** The closed form is exact
only because the functional's reference density is SET EQUAL to the requested
saturation density, so saturation is imposed at `u = 1` rather than found.
That is why the round trip closes to 1e-14 on n_sat, and it is also why
inverting the published NMPs does not return the published couplings exactly
(the shipped set saturates 0.3 % below its own n0): `gamma` comes back to
3e-5, `a0` and `b0` to 0.3 %.

### The E_sym convention, measured rather than asserted

The ticket said the shipped `a1 = -26.06` against this inversion's `-25.19`
is the E_sym convention. **Confirmed, with the number that proves it.** On the
shipped set:

    quadratic coefficient (this repo)   E_sym = 30.848   L_sym = 41.270
    full PNM - SNM step (Constantinou)  E_sym = 31.561   L_sym = 42.718

So the familiar target `{31.6, 43}` IS the shipped set stated in the other
convention — it was never an independent target. Feeding a
full-step-convention E_sym into a quadratic-convention inversion is exactly
the 0.87 MeV that moves a1 from -26.06 to -25.19. Both numbers pinned in
`test_the_two_E_sym_conventions_are_pinned`; both stated in the module
comment, `zl.md` and `zl.tex` before the round-trip test was written.

The rest-mass trap is pinned differently, and the first attempt at it was
WRONG: asserting the couplings are invariant under a nucleon-mass shift
fails, because the free Fermi gas genuinely depends on m, so `E_sat,K` moves
and the interaction remainder moves with it. The convention is about the
SUBTRACTION, not about mass-independence. What actually pins it is one line —
feed `E_sat = -16` and the recovered functional must bind at -16, not at
`m - 16`. A derivation carrying `+ m_H` lands 939 MeV away and that catches
it outright, with no tolerance.

### Both ZL facts confirmed, and one of them changed how it is tested

`Q_sat,V = 3(gamma-2) K_sat,V` is exact in the coefficients (both are
`b0 u^gamma` derivatives). It is NOT exact through the forward map: the
third-derivative stencil at h = 0.02 puts the ratio at -1.8006917 against
-1.8000000, converging as h^2 (-1.8001728, -1.8000433, -1.8000061 at
h = 0.01, 0.005, 0.002). So the test asserts the identity where it IS exact —
nothing isovector reaches gamma or b0, so moving E_sym, L_sym and gamma1
leaves the predicted Q_sat **bit-identical**. No tolerance, and it catches
any later edit that lets an isovector input leak into the isoscalar solve.

`K_sym` imposition needs no root find either. With `(gamma1-1) b1` already
fixed by E_sym and L_sym alone, `K_sym,V = 9[gamma1 (gamma1-1) b1 -
gamma(gamma-1) b0]` is LINEAR in gamma1:

    gamma1 = [K_sym,V/9 + gamma(gamma-1) b0] / [(gamma1-1) b1]

Verified at three targets: K_sym = -90, -50, 0 return gamma1 = 2.46725,
2.89385, 3.42710 and come back through the forward map at -90.008, -50.006,
+0.002, with E_sym held to 1e-6 throughout.

### What was built beyond the ticket's four items

**A status object.** §6 makes non-convergence a return value, and the closed
form does have unrealisable inputs: a target whose recovered functional
saturates outside `N_SAT_BRACKET` (n_sat = 0.30 is the test case), a
degenerate isoscalar target (X or D zero), gamma1 = 1. `invert_nmp` returns
`(Parameters, InversionStatus)` like `dd2` and `sfho`, with `ok` judged by
putting the couplings back through `compute_nmp` — the round trip IS the
score — plus `residual` and the `predictions` the closure left free.
`from_nmp` raises on failure, matching sfho's split exactly.

**One free choice, named once.** `gamma1` has no default (a default picks the
functional's high-density isovector behaviour on the caller's behalf), and
passing both `gamma1` and a dict `K_sym`, or neither, is a `ValueError`: they
are two names for the same freedom. Q_sat is refused as the sixth datum for
the reason above, rather than silently ignored.

### The downstream site the ticket did not list

`notebooks/hadronic_eos.py` called `zl_nmp.invert_nmp(**target_nmp)` and
relied on the `NotImplementedError`; under the new signature that is a
`TypeError`, which `run()` does not catch — a hard break, not a stale
comment. Fixed along with the markdown bullet above it that told the reader
zl "cannot be built *from* a set of them". The `m_eff_ratio` key in the
shared target is dropped for zl, which has no scalar field. The .py and
.ipynb were checked to be substantively in sync first (they differed by a
jupytext version stamp and nothing else) and re-paired after.

## Gate

`test/zl` (49) + all 12 baselines + `test_imports` + `test_parameter_routes`
+ `test_nonconvergence_return`: **329 passed, 5 skipped, 0 failed**. The full
suite was NOT the gate: a concurrent session holds `dd2`, `general`, `mixed`,
`CLAUDE.md` and both notebooks, so a whole-tree run is unattributable — the
blast radius here is `eos/zl` plus three cross-model tests, and that is what
was run.

`test/baseline/zl.npz` unmoved, as predicted: this adds a constructor and
changes no evaluation path. `zl.tex` compiles clean (8 pages).

One document needed no edit and is worth recording: `docs/STRUCTURE.md:292`
already listed zl in the "nuclear-matter parameters, inverse" row at HEAD,
which was FALSE while `invert_nmp` raised. This ticket made a standing claim
true rather than changing it -- so nobody should read that row as evidence the
map was checked.

`test/test_parameter_routes.py` flipped `HADRONIC["zl"]` False -> True, so the
"a refused route raises with a reason" arm no longer covers zl and the
present-and-working arm does. `docs/DEFERRED.md` had no zl entry for the
inversion to retire — the refusal was recorded in `did`'s entry as "the way
`eos.zl` does", and that cross-reference is corrected in place: DID cannot
follow zl here, because for DID it is the LIST to impose that is undetermined,
not one member of a family.
