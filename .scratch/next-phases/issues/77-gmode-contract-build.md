# Build the T = 0 g-mode composition contract and drop the DD2 import

Type: task
Status: resolved
Assignee: session 77-gmode
Blocked by: 53
Parent: ../map.md

## Question

Execution of [ticket 53](53-gmode-contract.md)'s ruling. Read it first — the
design is settled and grounded in Zhao & Lattimer, arXiv:2204.03037 Eq. (1).

A `general/` table carrying **the two sound speeds along the sequence** —
`c_e = sqrt(dp/deps)` and the frozen-composition `c_s` — beside
`EOSTable_for_TOV`, the layer both `astro/` and the models may import. Models
produce it; `gmode` consumes it; **no model internal is imported**.

**T = 0 only.** At T = 0 the thermal axis collapses and the composition axis
does not, so the payload is exactly two numbers per point with no thermal-axis
naming problem. Finite T when it is useful.

Removes the last live §1 breach: `gmode/rates.py:85` and
`gmode/sound_speeds.py:94,149` import `eos.dd2.solver` and `eos.mixed.responses`
at module level, so `import eos.astro.gmode` pulls DD2 in today.

`gmode/verify/run_full_check.py:39-41` reaches into dd2 as well. §1's `verify/`
carve-out is written for the model-to-model half of the rule; **whether it
extends to an astro suite reaching down into a model is a separate call** and
this ticket must make it explicitly rather than assume it.

Only `dd2` can fill the contract today ([ticket 78](78-composition-freeze-nine-models.md)).
A model that cannot raises saying so — §3's own answer to a partly-filled
surface. So this ticket does not make `gmode` general; it makes gmode's
DD2-only-ness **visible and per-model** instead of hidden in an import.

**Rides along, same files** (§6, §7): `gmode/rates.py:90-97`'s `G2_FERMI`,
`G_A`, `F_PI_NN`, `M_PI` are module constants with no override path, and `M_PI`
duplicates a mass `general/particles.py` already declares. Mass from there;
weak couplings become arguments.

Done when `import eos.astro.gmode` pulls in no model package, `test_imports.py`
gates it, and the dd2 g-mode numbers are unchanged.

## Resolution

Built. `import eos.astro.gmode` pulls in **no model package**, every dd2 g-mode
number is **bit-identical**, and `test_imports.py` gates the rule two ways.

**The contract — `eos/general/sound_speeds.py` (new).**
`EOSTable_for_gmode` SUBCLASSES `EOSTable_for_TOV`, adding `cs2_equilibrium`
and `cs2_frozen`. Subclassing rather than wrapping because a g-mode table IS a
structure table: the same object that carries the buoyancy is the one the
background integrates, and anything already taking a TOV table takes this one
unchanged. `__post_init__` asserts the five columns are parallel — a length
mismatch is a silent misalignment, not a physics statement, so it is the one
thing validated (monotonicity, causality and `N^2 >= 0` are deliberately not,
matching `EOSTable_for_TOV`'s own docstring and section 8). The module also
holds `sound_speed_eq(P, eps)`, `sound_speed_frozen(compressed_state)` and
`cs2_frozen_isobaric(cs2_H, cs2_Q, chi)`.

**Field names: `cs2_equilibrium` / `cs2_frozen`,** settling the third spelling
ticket 53's Note parked here. Both speeds are at T = 0, so the thermal axis
distinguishes nothing and naming it would advertise a freedom the table does
not carry; what separates them is composition. **gmode's internal `cs2_eq` /
`cs2_ad` are deliberately NOT renamed**, and not for cost: `StellarBackground`'s
second slot holds the DYNAMICAL sound speed once `at_frequency` folds in a
finite rate, at which point "frozen" would be false. The table's column is the
strict limit; the star's field is not. Both docstrings now say so.
53's "mixed and gmode are ONE vocabulary" premise is severed by this ticket —
gmode no longer imports mixed — so `mixed.eos_response`'s keys are now an
independent, section-5-governed surface and were left alone.

**The four sites.**
- `rates.py:85` (top-level dd2) — gone. The rates needed exactly THREE numbers
  from the model: `m*_n`, `m*_p` and the susceptibility `A`. Fermi momenta are
  kinematics from (n_B, Y_p) and `mu_e` comes from `general/`, so
  `equilibration_rate` now takes those three as arguments. `susceptibility_A`
  takes the chemical imbalance as a CALLABLE, because it is a derivative of it
  and only a model can evaluate one at a perturbed composition.
- `sound_speeds.py:94` (top-level mixed) — gone. `sound_speed_eq` moved to
  `general/`; `eos/mixed/responses.py` now imports it from there rather than
  keeping a second `np.gradient` (section 7), and it stays on mixed's surface.
- `sound_speeds.py:149` (function-local dd2, inside `cs2_frozen_nucleonic`) —
  the function left the package; see the producer note below.
- `cs2_frozen_point` / `cs2_frozen_along` were **deleted outright**: grep found
  no caller anywhere in `eos/`, `test/`, `notebooks/` or `docs/`. The physics
  they carried survives as `cs2_frozen_isobaric` in `general/`, and the wings
  it combines are already `mixed.responses.sound_speed_frozen` at chi = 0 and 1.

**The verify/ call, made explicitly.** Section 1's `verify/` carve-out DOES
extend to an astro suite importing a model, and does NOT extend to the reverse.
The directions are not symmetric. A model importing `astro/` is the cycle the
rule exists to prevent, and section 1 gives that half no carve-out anywhere —
"the astro half of the rule has no such carve-out" is that sentence, sitting
directly after the list of what a model's suite may import, where `astro/` is
conspicuously absent. An astro suite importing a model creates no cycle at all,
because `astro/` already sits above the models; and the carve-out's own stated
justification — "a suite is not on the path an inference sampler imports, which
is what the layering rule protects" — is a claim about SUITES, not about which
layer a suite belongs to, and holds here word for word. It is used narrowly:
five of the eight checks are model-free and `include_dd2=False` turns the rest
off. Written into the suite's docstring, `test_imports.py`'s new test, and
`docs/DEFERRED.md`.

**The DD2 producer, and an unplanned finding.**
`dd2.eos_response(frozen='composition')` returns the **leptonless** `cs2_ad`.
For a g-mode that is the wrong half: the equilibrium speed follows the
beta-equilibrium sequence WITH leptons, so differencing the two compares two
different fluids, and the lepton term is a sizeable fraction of the entire
`c_s^2 - c_e^2` signal — a leading-order error in `N^2`, not a refinement. So
dd2 cannot fill the contract through its current `eos_response` either. The
right fix is section 5's THIRD response axis (`leptons=`) on
`eos_response(frozen='composition')`, in `eos/dd2/`.

**That fix was not made, and deliberately.** A concurrent session is editing
`eos/dd2/responses.py` and `eos/dd2/api.py` right now — `eos.dd2` was
transiently unimportable mid-session (`sound_speed_adiabatic` renamed to
`sound_speed_isothermal_frozen` before `dd2/__init__` caught up). The ticket
scoped ownership to `eos/astro/gmode/` and the new `general/` module for
exactly this reason. So the producer sits in
`eos/astro/gmode/verify/run_full_check.py` as the public `dd2_table` /
`dd2_frozen_cs2` / `dd2_equilibration_rate`, which the tests import so there is
ONE copy rather than two, with a docstring saying it belongs in `eos.dd2`.
Recorded in `docs/DEFERRED.md`. This is a ticket-78 follow-on, not a gap this
ticket left open by accident.

**Rides along (sections 6, 7).** `G2_FERMI`, `G_A`, `F_PI_NN` became a frozen
`WeakCouplings` dataclass with the published set as its default, threaded
through `lambda_direct_urca`, `lambda_modified_urca` and `equilibration_rate`.
`M_PI` now comes from `eos.general.particles.PiP.mass`.

**Numbers.** Baselined against a clean `git archive HEAD` copy (the working
tree was mid-edit), then re-measured through the contract:

    cs2_equilibrium / cs2_frozen        0.000e+00   IDENTICAL
    the same after crust attachment     0.000e+00   IDENTICAL
    N^2 on the 1.4 M_sun background     0.000e+00   IDENTICAL
    M, R                                0.000e+00   IDENTICAL
    mode frequencies g3 g2 g1 f         0.000e+00   IDENTICAL
      63.568 / 88.221 / 149.565 / 2064.516 Hz
    equilibration rate at T = 3 MeV     1.292e-05   changed, as ordered

The single change is the rates, and it is the pion mass: `general/particles`
carries 139.570 against the deleted module constant's 139.57039, and `M_PI`
enters the modified-Urca prefactor to the fourth power. 1.3e-5 relative, far
inside the Fermi-surface approximation the module already documents.

**Tests.** `test/gmode` 37 passed (831 s — it is slow, not hung, as
`docs/DEFERRED.md` records) plus a new `test/gmode/test_contract.py`, 9 passed.
`test/test_imports.py` 199 passed. `gmode/verify/run_full_check` PASS on all
eight checks, g1 = 149.6 Hz. The new gate was confirmed to BITE: a deliberate
`from eos.dd2.solver import ...` planted in `background.py` failed both the AST
test and the fresh-interpreter runtime test.

Interpreter: python.org 3.14.2, numpy 2.3.5, scipy 1.17.0. Whole suite collects
1711 tests.

Status: resolved.
