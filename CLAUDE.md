# CLAUDE.md — conventions for the `eos` repository

Durable project conventions. Read this before editing anything. These are not
suggestions: they are the invariants the test suite encodes. If a change
appears to require breaking one, stop and ask rather than working around it.

---

## 1. What is frozen

`eos/dd2/` and `eos/tov/` are the **completed, validated Phase-1 baseline**.
`eos/vmit/` is the quark-sector baseline. Do not refactor, rename, or
"improve" these without an explicit instruction naming the file. Phase-2 work
is *additive*: new modules that consume these as libraries.

`eos/zlvmit/` is the **first-generation** nucleons+quarks mixed-phase code. It
is a reference for behaviour, not a template for structure — its per-mode
branch duplication is the thing Phase 2 exists to replace.

The golden reference values embedded in `test/dd2/` pin the T=0 nucleonic
sector. They are ground truth: if a new implementation disagrees with them,
the new implementation is wrong.

## 1b. Dependency direction (non-negotiable)

**`eos` must never import `nucleation`** (or any other downstream project).
`eos` is the library; `nucleation` and friends are its consumers and declare it
as a dependency. An import in this direction is a cycle: it makes
`pip install eos` alone insufficient to use `eos`.

This was violated once — `eos/dd2/notebook_api.py` imported
`nucleation.analysis.figure` for the M-R constraint overlay. The overlay now
lives where it belongs, in `eos/general/observational_constraints.py`.
`test/test_imports.py::test_eos_never_imports_nucleation` enforces the rule.

Corollary: shared *figure* code belongs in `eos/general/figure_style.py`, the
one home for publication styling. Do not re-declare `STANDARD_COLORS` or write
a second rcParams setter in a submodule — import them.

## 2. Sign and naming conventions (non-negotiable)

- **Strangeness**: `S = +1` per *s*-quark. So `Λ` has `S = +1`, `Ξ` has
  `S = +2`, and the *s* quark itself has `S = +1`. This is the opposite of
  the PDG convention; it is used consistently throughout this repo. Never
  silently flip it.
- **Charge `C` excludes leptons.** `n_C` is the *non-leptonic* electric charge
  density (`n_C = n_p` for nucleonic matter). `Y_C = n_C / n_B`. Total
  electric charge neutrality is a separate, additional condition. Conflating
  these two is the single most common error in this domain.
- **`Y_S`, `Y_L`** are likewise ratios to `n_B`, not to total particle number.
- Chemical potentials follow the conserved-charge decomposition
  `mu_i = B_i mu_B + Q_i mu_Q + S_i mu_S + L_i mu_L`. The species potentials
  are *derived*, never independent unknowns.
- **Kinetic vs. full potentials**: `nu_i = mu_i - Sigma0_i`. Solver unknown
  vectors use *kinetic* potentials (`nu_n`, not `mu_B`) — see the docstring of
  `eos/dd2/physics/residual.py` for why (the rearrangement term and the
  `Gamma_omega*omega0` shift cancel out of the iteration, and `nu_n` warm-starts
  well across a density sweep where `mu`-based unknowns do not).

## 3. Units

- `eos/dd2/physics/` works in **natural units**: MeV, MeV^3 (densities),
  MeV^4 (eps, P). The constant `hc3` converts to/from fm-based units.
- `eos/general/fermi_integrals.py` and `bose_integrals.py` (JEL) return
  **fm-based** quantities. Every call site must convert. `kinetic_thermo`
  already does this — use it rather than calling JEL directly.
- Public APIs and tables are fm-based (`fm^-3`, `MeV/fm^3`). Do not leak
  natural units across a module boundary.

## 4. The reference/fast split

Every solver exists in two flavors and this pattern must be preserved:

- `*_ref` — readable, straightforwardly correct, no compiled kernels, no
  hand-rolled algebra. This is what correctness is judged against.
- `*_fast` — the optimized path (Numba kernels, analytic Jacobians).

**Rule**: `_fast` must agree with `_ref` to the tolerances the tests state.
When they disagree, `_ref` is right. Never delete or bypass a `_ref` path to make a
`_fast` path pass, and never make `_fast` the only implementation.

Analytic Jacobians are hand-coded (see `eos/dd2/physics/jacobian.py`,
`coefficients_jac.py`). Every analytic Jacobian **must** have a
finite-difference agreement test.

**On autodiff — read this before "finishing" the JAX backend.** Autodiff is
*deferred, not adopted*, and this was a deliberate Phase-1 decision with a
physics reason, not an unfinished task:

- The JAX autodiff path was abandoned because the JEL Fermi/Bose core and the
  T=0 threshold kink **do not trace cleanly**. See the docstrings of
  `eos/dd2/physics/jacobian.py` and `test/dd2/test_dd2_m9.py`.
- What actually shipped is the **hand-coded exact analytic Jacobian**
  supplied to the same MINPACK solver
  (`eos/dd2/physics/jacobian.py`). `eos_fast` means analytic-Jacobian, not JAX.
- `eos/dd2/xp.py` is a *prepared seam only* — it currently imports numpy, and
  `jax` is not a dependency of this project. Physics modules import `xp` from
  it so a Numba/JAX wrapper could drop in later; that is an option kept open,
  not a shipped backend.
- Do not "restore" a JAX path on the strength of a stray comment; the
  analytic Jacobian is the shipped design.

The mixed-phase solver in `eos/mixed/` follows the same pattern: hand-coded analytic
Jacobians, finite-difference-verified.

## 5. Species flags

`SpeciesFlags` (`eos/dd2/species.py`) makes every degree of freedom an
explicit named boolean. Two rules:

- No sector is enabled/disabled implicitly by "its coupling happens to be
  zero". If a sector is off, its flag is `False`.
- Setting a not-yet-wired flag **raises**. Never turn a `NotImplementedError`
  into a silent no-op to make a call succeed.

## 6. JEL integrals

The JEL Fermi/Bose integral implementation must remain a **selectable
option**. It may be supplemented by alternatives, never replaced by them.
It is the validated path against which any alternative is checked.

## 7. Thermodynamic consistency — the invariants that must hold

These are checked by the verification suite and any new physics must satisfy
them. They are the fastest way to catch a wrong implementation:

- Euler relation, per phase: `eps + P = T*s + sum_i mu_i n_i`
- Free energy: `f = eps - T*s`, and `f = -P + sum_i mu_i n_i`
- The rearrangement self-energy `Sigma^R` enters `mu` and `P`, **never**
  `eps`. This is what makes DD-RMF thermodynamically consistent; getting it
  wrong produces an EoS that looks plausible and violates Hugenholtz–Van Hove.
- HVH / `dP/dn_B` consistency along solved sequences.
- `P` monotonic in `n_B` along an equilibrium sequence, `0 <= c_s^2 <= 1`.

## 8. Testing

- Tests live in `test/dd2/`, `test/mixed/` etc. Name a test file after the
  physics it checks (`test_fixed_yc.py`), not after a development milestone.
- New physics gets a test in the same style *and* an entry in the
  verification suite (`eos/dd2/verify/`) where it is a physics invariant
  rather than a unit behaviour.
- The full suite must pass before any commit that touches solver internals.
- Do not loosen a numerical tolerance to make a test pass. If a tolerance
  genuinely needs to change, say why in the test.

## 9. Style

- Python only. No compiled-language ports.
- NumPy/SciPy + Numba for the fast paths. No new heavy dependencies without
  asking.
- Every module gets a docstring that states the physics it implements and is
  self-contained (see §10). Follow the existing docstrings — they are
  unusually informative by design and that is deliberate.
- Prefer adding a module over growing an existing one past ~600 lines.

## 10. When in doubt

**Docstrings must stand on their own.** This is a public repository, so a
comment may not depend on a document that is not in it. Working notes under
`docs/` and the drafting material that produced a module are not part of the
published code: state the physics, name the equation, give the literature
citation — never "see spec §3.27" or a milestone number. `eos/` and `test/`
are currently free of such references and should stay that way.

`DD2_OPEN_QUESTIONS.md` records choices that were pinned to make progress —
check it before re-deciding something. If the physics is genuinely ambiguous,
ask rather than picking a convention silently; an undocumented convention
choice is a bug that surfaces months later.
