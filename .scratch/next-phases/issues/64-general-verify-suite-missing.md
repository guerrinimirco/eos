# eos/general/ has no verify/, which CLAUDE.md section 5 states it has

Type: task
Status: resolved
Blocked by: 11
Parent: ../map.md

## Question

Found while running [ticket 49](49-nonconvergence-return.md)'s verify sweep,
and pre-existing — this is not a regression from that ticket.

CLAUDE.md section 5 says it in the present tense, as a paragraph of its own:

> **`general/` carries a `verify/` too.** It is not a model either, but it is
> the single home of the Fermi and Bose integrals (section 7), the
> conserved-charge basis maps (section 2) and the thermal meson gas — the
> pieces every model's correctness rests on, and the ones a wrong result is
> hardest to trace back to. Its suite checks those shared pieces against each
> other: JEL against the alternatives section 7 requires be validated against
> it, the basis maps against the species tables, the T = 0 limits against the
> finite-T forms as T -> 0.

It does not exist. `ls eos/general/` shows no `verify`, and

    python3 -m eos.general.verify.run_full_check
    ModuleNotFoundError: No module named 'eos.general.verify'

Nine of the ten `verify/` entry points a full sweep should hit are real and
pass; this is the missing tenth, and it guards the code with the widest blast
radius in the repository.

**Not covered by [ticket 51](51-verify-invariants.md)**, whose four missing
invariants are all inside models — checked, `general` does not appear in it.

The section text already names the three checks the suite owes, so the scope is
written: JEL against the alternatives validated against it, the basis maps
against the species quantum-number tables, and the T = 0 limits of the
finite-T forms as T -> 0. `test/general` (128 passing cases) may already cover
some of this as unit behaviour — the first job is to sort what is there into
"physics invariant, belongs in `verify/`" and "unit behaviour, stays a test",
per section 12, rather than writing three checks that duplicate tests.

Note the second half of section 8's gate does NOT apply: `general/` has no
`table.py` and hands no table to a structure solver, so it does not owe the
monotone-P / causal-c_s^2 delivery check.

## Answer

**Shipped: `eos/general/verify/` — `__init__.py` (docstring only, the shape 8 of
the 11 model suites use) and `run_full_check.py`, 5 checks, one entry point.**
`python3 -m eos.general.verify.run_full_check` now prints PASS instead of
raising `ModuleNotFoundError`.

    general run_full_check: PASS
      [ok ] Fermi: JEL vs alts       max_err=1.16e-03  scipy 1.2e-03 over 11 points, Gauss-Laguerre 3.1e-04 over 4
      [ok ] Bose: JEL vs alts        max_err=1.05e-03  scipy 6.5e-04, Gauss-Laguerre 1.1e-03, over 8 points
      [ok ] basis vs species table   max_err=1.14e-19  24 hadrons and quarks, 3 leptons
      [ok ] thermal meson gas        max_err=6.42e-17  nonet quantum numbers, Euler at T = 40 MeV
      [ok ] T -> 0 limit             max_err=1.73e-05  4 points, T = 0.1 MeV; the T^2 approach holds

Nothing outside `eos/general/verify/` was touched. The suite imports only
`eos.general.*`, so section 1's bottom layer stays importing nothing else in
the repo.

### Sorting what was already there, first — as the ticket asked

`test/general` (now 329 cases with `test_imports.py`) was read before anything
was written, and one of section 5's three checks turned out to be ALREADY DONE
as a test: `test/general/test_fermi_gauss.py::test_agrees_with_jel` is the
split-panel Gauss-Legendre gas validated against JEL, which is section 7's rule
for the third alternative. Repeating it in `verify/` would have been the
duplication section 12 warns against, so the suite does not, and its module
docstring says where that validation lives instead. What had NO cross-check
anywhere is what the suite took: the Gauss-Laguerre rule, the whole Bose family,
and the T = 0 closed forms.

`test/general/test_basis.py` overlaps the basis check by design and not by
accident: those are ~12 NAMED species and hardcoded-copy comparisons, while the
verify entry sweeps the WHOLE table — 24 hadrons and quarks, every row through
Gell-Mann-Nishijima, C = Q, `charges_of`, `species_potential` and the name
lookup. A single wrong row cannot hide behind the species somebody thought to
name.

### The five checks

1. **Fermi: JEL vs alternatives.** JEL against `Fermi_Numerical` (scipy) over
   11 points and against `solve_fermi_gl` over the 4 that are not strongly
   degenerate. Tolerance **2e-3**, JEL's own accuracy class (quoted ~1e-4,
   measured worst on the grid 1.2e-3, reported in the detail string).
2. **Bose: JEL vs alternatives.** The same for the boson family, which nothing
   in the repository cross-validated before — and which the thermal meson gas
   is built on.
3. **Basis vs species table.** The whole-table sweep above, plus the density
   sums (with a lepton in the dict that must not enter any of the three), the
   quark flavour sums, the `quark_potentials` round trip and the octet map.
4. **Thermal meson gas.** The nonet's inline (Q, S) table against
   `eos.general.particles` — the kaon rows are where section 2's S = +1-per-s
   convention bites — plus the gas's per-species Euler relation.
5. **T -> 0 limit.** `solve_fermi_t0` against the exact finite-T quadrature
   down a halving ladder T = 0.4, 0.2, 0.1 MeV. **Two statements, and the
   second is the one that bites**: the error must be small at the lowest T,
   AND it must fall like T^2 (Sommerfeld), so halving T quarters it. A closed
   form carrying a wrong CONSTANT passes the first at a loose enough tolerance
   and fails the second outright. JEL is deliberately NOT the finite-T side
   here: its own ~1e-4 approximation error floors the ladder at the third rung
   and would hide exactly that signature (measured — JEL's error against the
   T = 0 forms stops falling at 1.1e-4 while the quadrature's keeps quartering).

### Every check was proved able to fail

A check that cannot fail is not a check. Each was re-run against a deliberately
broken `general/`, monkeypatched at the suite's own import site:

| break | result |
|---|---|
| JEL Fermi `P` x 1.005 | FAIL 5.0e-03 |
| JEL Fermi `n` x 1.003 | FAIL 3.0e-03 |
| JEL Bose `eps` x 1.005 | FAIL 5.1e-03 |
| `charges_of` returns the PDG strangeness sign | FAIL, names `Lambda`, `Sigma+`, ... |
| `mu_S` dropped from `species_potential` | FAIL 2.6e-01 |
| leptons let into `charges_from_densities` | FAIL 3.0e-05 |
| `quark_charges` n_S sign flipped | FAIL 6.0e-04 |
| meson gas kaon `S` flipped to PDG | FAIL, names K+, K-, K0, K0_bar, K*... |
| meson gas `eps` x 1.0001 | FAIL 1.0e-04 |
| T = 0 closed-form `P` x 1.0005 | FAIL, "the T^2 approach FAILS" |
| T = 0 closed-form `P` x 1.00002 | FAIL — **below the magnitude tolerance; caught by the ratio alone** |
| (control) unbroken | PASS 1.73e-05 |

The last two rows are why the ladder is in the check: a 2e-5 constant offset is
invisible to any magnitude test at this tolerance and dies on the T^2 ratio.

**The pressure scale was tightened after the falsification run, not before it.**
The first draft normalised P by eps; a degenerate gas has P/eps ~ 0.1, so a
half-percent error in P arrived as 5e-4 and the check PASSED the first broken
input. P is now measured against P, which costs nothing (the measured worst is
unchanged at 1.16e-03) and is what makes row 1 of that table a FAIL.

### Findings — reported, not fixed (both are in `fermi_integrals.py`, which
this ticket does not touch)

1. **`solve_fermi_gl` is unusable in a window its own fallback does not cover,
   while its docstring says "Higher accuracy than JEL".** A 30-node
   Gauss-Laguerre rule cannot resolve a Fermi step. It falls back to the
   analytic forms only below `T < 1e-4` MeV, but the breakdown starts around
   `T / (mu - m) ~ 0.08`, four orders of magnitude higher. Measured against the
   scipy quadrature at m = 100, mu = 500 MeV:

   | T (MeV) | T/(mu-m) | GL rel. err | JEL rel. err |
   |---:|---:|---:|---:|
   | 0.5 | 0.001 | **9.8e+02** | 1.2e-03 |
   | 2 | 0.005 | **5.4e+02** | 1.2e-03 |
   | 5 | 0.013 | **1.2e+01** | 1.2e-03 |
   | 10 | 0.025 | 6.9e-01 | 1.1e-03 |
   | 20 | 0.050 | 2.8e-02 | 1.0e-03 |
   | 30 | 0.075 | 2.2e-03 | 9.1e-04 |
   | 50 | 0.125 | 7.9e-05 | 5.3e-04 |

   At T = 0.5 MeV it returns a density three orders of magnitude wrong — and
   returns it silently. The suite states the boundary as a named constant,
   `GL_MIN_DEGENERACY = 0.1`, and compares only above it, rather than
   hand-picking a grid that happens to avoid the hole. Nothing in `eos/` calls
   `solve_fermi_gl` on a solve path (checked), so this is a validation-tool
   hazard, not a wrong number in any model — but raising the fallback threshold
   to the measured breakdown is a one-line change somebody should own.

2. **The split-panel entropy is 1.1% off JEL at (m = 939, mu = 960, T = 1) MeV**
   — s = 0.008221 against JEL's 0.008309 and scipy's 0.008307, so it is the
   panel rule that is out, not JEL. Not a k_max artifact (unchanged at pad =
   200, 400, 800). The entropy integrand's width is ~T inside a 25 MeV
   `THERMAL_COLLAR` panel carrying 24 nodes, which is marginal at T = 1 MeV.
   `test/general/test_fermi_gauss.py` allows s at rel = 5e-3 and its grid does
   not reach this corner. Recorded because a narrow-degeneracy, low-T nucleon
   gas is not an exotic point.

3. **Cosmetic, not acted on**: `basis.charges_of` returns `particle.charge`, not
   `particle.strong_charge`, so `charges_of(Electron)[1]` is -1 where section 2
   says leptons carry no C. Harmless — the two agree for every hadron and quark,
   and `charges_from_densities` excludes leptons from the sums, which is the
   place it would matter. The suite therefore checks the exclusion where it is
   enforced rather than asserting an identity the code does not claim.

### Gates

- `python3 -m pytest test/test_imports.py test/general` — **329 collected, 329
  passed** in 3.58 s. The layering test sees the new subpackage and is content:
  `general/` still imports nothing else in the repo.
- `python3 -m pytest test/baseline` (rtol = 1e-10) — **6 failed, 10 passed**:
  `ccdm, dd2, enjl, njl, tov, zlvmit`. **Byte-identical to the failing set in
  `output/_audit/pytest_after_ticket61_baseline_py314.txt`, so zero added.**
  These are the interpreter artifact ticket 57 ruled on and
  [ticket 62](62-regenerate-baselines-py314.md) owns.
- Interpreter **python.org 3.14.2** (numpy 2.3.5, scipy 1.17.0), never prefixed
  with `timeout`.
- A full-suite number is not measurable while three sessions hold
  `notebooks/*.py`; the targeted suites above are the whole of what this ticket
  can affect, since nothing in the repository imports `eos.general.verify`.
