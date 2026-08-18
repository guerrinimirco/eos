# The ENJL first-order transition: audit, and the design for the construction

Phase A of the transition work. This document changes no code. It records what
the author's own Maple worksheet does (§1), what the reference tables therefore
are (§2), and the design the construction has to have to be built on top of
`eos/mixed` (§3-§6). Every number quoted here was measured, and the measurement
is named beside it.

---

## 1. What the author's worksheet actually does

The worksheet is an oracle, not a source; the handling rules and the decoder are
in `test/enjl/reference/maple/README.md`, which is gitignored and stays that
way. Three procedures matter: `HMasymcal` (fixed composition), `QMasymcal`
(quark matter), and `MPcal`, which is the one that produced the five
`Beta_*.dat` files this repository validates against.

### 1a. There is no construction anywhere in it

`MPcal(nH, wrt)` declares

```
local  i, esigmatt, ms, m0, esigma0, Omegavac, epsk, epst, epa, P, Sigmab,
       Deltam, av, atv, dav, datv, nchb, n00, ti, as, das, nV, Sigmaq, nbq;
global nuo, esigmat, kk, E0, mix, f, ns, nn, mumu, fL, fv;
```

and its body is one under-relaxed Picard iteration on the three quark
condensates, `mix = 0.5`, converged when `sum_q |esigma0[q] - esigmat[q]|`
falls below `10^(8 - Digits)` = 1e-7 in MeV^3 units (relative ~1e-14 on
`esigma[q] ~ 241.9^3`), abandoned with `print("can't find convergency")` after
24999 sweeps. Inside it there is a `fsolve` for the muon threshold and a
`Heaviside` gating the muon onset, and that is the whole of its root-finding.

**There is no Maxwell construction, no common tangent, no pressure
maximization, no second branch and no pairing of any kind.** `MPcal` solves one
state at one density.

### 1b. The branch is chosen by the seed, and the seed is the previous density

`esigmat` is **global**. Each call seeds the gap iteration from the condensates
the previous call converged to (`esigma0[i] := 2*esigmat[i]` is only the device
that forces entry to the `while` loop; the iterate itself starts at
`esigmat[i]`). The hard-coded vacuum masses in the procedure,

```
m0[u] = 367.648260165719      m0[s] = 549.479210995025
```

are used for the vacuum subtraction, not as a per-call seed — they match
`docs/enjl/REFERENCE_TABLES.md` §4d and this repository's `vacuum_solution()`
to 8e-9 relative.

So the worksheet is a **continuation**, and the branch it lands on is whatever
is continuously connected to where the sweep started. Nothing in it can select
between branches, because nothing in it ever holds two at once.

The driver cells confirm that the author steered the branch by hand, by
choosing the sweep direction and the starting density per run. The `SYM`, `PNM`
and `QM` cells all sweep `for nB from 0.01 to 1 by 0.01` (upward); the `MP`
cell that writes `Beta_fV07_B1000_L.dat` — the f_q = 0.7, B = 1 GeV/fm^3 set,
`_L` for the low-density piece — sweeps **`for nB from 0.6 to 0 by -0.01`**,
downward. Sweeping down from 0.6 through the 0.4486-0.5342 window is exactly how
you map the chirally-restored branch *below* its own transition; you need that
branch and the upward one at once before a Maxwell point can be located at all.

**This answers DD2_OPEN_QUESTIONS G3's third candidate, and it is the right
one:** branch (iii) is not excluded by any rule in the author's treatment. It is
simply never reached, because a continuation started on the hadronic side at
n_b = 0.01 never visits it. The published curves are not the output of a
selection rule, so no selection rule will reproduce them.

### 1c. The coexistence rows were inserted afterwards, by hand, and not everywhere

This is visible in the tables without reference to the worksheet at all. Per
file, on solved rows, counting places where mu_b decreases with n_b:

| file | equal-(mu_b, P) pair | raw dmu_b/dn_b < 0 |
|---|---|---|
| `Beta_fq1.0_B0.dat` | 0.6438 / 0.6659 | at 0.666 (the pair itself) |
| `Beta_fq1.0_B1.dat` | 0.6377 / 0.6767, `munr` blank | at 0.670 |
| `Beta_fq0.7_B1.dat` | 0.4486 / 0.5342 | at 0.534 (the pair itself) |
| `Beta_fq0.7_B0.dat` | 5.5757 / 5.6010 | **at 0.600** |
| `Beta_fq0.5_B1.dat` | two plateaus, interpolated | none |

Read the rows and the three regimes separate cleanly.

*Constructed, interior omitted* — `fq0.7_B1` runs 0.44 (M_u = 246.5, broken),
then the endpoint 0.448564, then straight to the endpoint 0.534224
(M_u = 5.5, restored) at the same mu_b = 1168.4748 and the same P = 69.6419,
then 0.54. The window interior is simply not in the file.

*Constructed, raw branch retained alongside* — `fq1.0_B1` runs 0.63 (broken,
mu_b = 1400.23), then **two inserted rows with `munr` blank** at 0.6377 and
0.6767 both at P = 202.1530, then 0.67 (restored, mu_b = 1399.87 — *below* the
0.63 value, i.e. a metastable continuation row sitting inside the window).

*Not constructed at all* — `fq0.7_B0` at its **chiral** transition runs
0.59 (M_u = 93.3, mu_b = 1217.64, P = 109.24) straight into
0.60 (M_u = 15.7, mu_b = 1214.41, P = 107.30). Both mu_b and P step
*backwards*. That is a continuation that jumped branch mid-sweep and was left
in the file. Only that file's *deconfinement* transition at 5.5757/5.6010 got
a constructed pair.

**Consequence for the gates.** The four windows of REFERENCE_TABLES §5 are the
only constructed objects in the five files, and they are the only thing a
construction may be gated against. The 0.59 -> 0.60 step in `fq0.7_B0` is not a
transition the author located; reproducing it is not a target, and a gate that
demands it would be gating against a solver artefact.

### 1d. E0 — confirmed, and it is the same number

The worksheet hard-codes

```
E0 := -4263.84421113853 * tran^3        (tran = hbar c = 197.327 MeV fm)
```

against this repository's `thermodynamics.vacuum_energy_density()` =
-4263.8455 MeV/fm^3 — agreeing to **3.0e-7 relative** — and against the
-4263.849 offset REFERENCE_TABLES §4d extracts from the tables themselves.
Three independent routes, three matching numbers, all negative. Nothing to
change.

### 1e. Quark occupation — there is no restriction in the code (for SESSION 4)

Paper 1 §II says that in the quarkyonic phase "only baryons can be excited to
higher energy states while quarks are restricted to the lowest energy states as
they are still confined". **The worksheet implements no such constraint.** Its
quark sector is a plain filled Fermi sphere: `nuo[i] = (6 pi^2 nn[i]/g[i])^(1/3)`,
with mu_q from the ordinary chemical equilibrium mu_i = B_i mu_b - q_i mu_e and
n_q from nu_q with no cap, no occupation weight and no Pauli blocking against
the baryon sea. The tables agree — they satisfy Eq. (23) over all six species
to 0.17-1.19 MeV (REFERENCE_TABLES §4) with independent mu_u, mu_d, mu_s.

So at T = 0 the sentence is a *gloss on what filling a Fermi sea from the bottom
already means*, and carries no implementation content. Its finite-temperature
content is therefore an open modelling decision, **not** a constraint inherited
from the author's code: at T > 0 a smeared quark distribution is what the
equations as written give, and anything else has to be argued for from the
paper's physics rather than from precedent. Recorded for session 4 and for
DD2_OPEN_QUESTIONS G4.

---

## 2. What the reference tables are, restated

A `Beta_*.dat` file is: continuation output along one or more branches, with the
author's hand-located coexistence endpoints spliced in, and — in `fq0.5_B1`
only — the plateau interiors filled by linear interpolation. It is not a
constructed EoS. `docs/enjl/REFERENCE_TABLES.md` §5 and §6 already say most of
this; §1c above adds that `fq0.7_B0`'s chiral transition was never constructed.

**One correction to REFERENCE_TABLES §5.** It states the four windows' mu_b
should be read from `munr`. On `Beta_fq1.0_B1.dat` that is wrong: `munr` is
**blank** on both endpoints (they are inserted rows, and blank `munr` is
precisely the file's marker for "not solved" — §6). `mun` carries the correct
1411.0842 there. Measured on all four pairs:

| file | n_B | `mun` | `munr` | mu_u + 2 mu_d | P |
|---|---|---|---|---|---|
| fq1.0_B0 | 0.64375 | 1381.2899 | 1381.2899 | 1475.9902 | 186.5964 |
| fq1.0_B0 | 0.66592 | 1381.2898 | 1381.2898 | 1381.2898 | 186.5964 |
| fq1.0_B1 | 0.63771 | 1411.0842 | — | 1760.0857 | 202.1530 |
| fq1.0_B1 | 0.67673 | 1411.0842 | — | 1440.0995 | 202.1530 |
| fq0.7_B1 | 0.44856 | 1168.4748 | 1168.4748 | 1352.3300 | 69.6419 |
| fq0.7_B1 | 0.53422 | 1168.4747 | 1168.4747 | 1168.4747 | 69.6419 |
| fq0.7_B0 | 5.57571 | 6348.7561 | 6348.7561 | 6333.0111 | 14262.3172 |
| fq0.7_B0 | 5.60098 | **5852.3091** | 6348.7562 | 6348.7562 | 14262.3172 |

The rule the gate must use is therefore **`munr` where finite, else `mun`** —
which is also the rule that survives the `fq0.7_B0` deconfinement trap in the
last row, where `mun` is the dissolved neutron's own potential and off by
496 MeV.

---

## 3. The adapter pair, and how the seed trap is closed

ENJL is one functional with three self-consistent branches, so the pairing is
two adapters over **one** model, not two models. Both go through
`eos.enjl.thermodynamics.thermo_from_mu(par, mu_B, mu_C, mu_S, T, x0)`, which is
already on the branch at 020df3b and whose docstring already states the trap:
"WHICH BRANCH is returned is decided by `x0`".

**The trap, measured.** At the f_q = 0.7, B = 1 coexistence potential
mu_B = 1168.4748 MeV and its mu_C, `thermo_from_mu` returns

| seed | n_B [fm^-3] | P [MeV/fm^3] |
|---|---|---|
| M_q = (244, 244, 470) — broken | 0.51712 | 94.6930 |
| M_q = (m_u0, m_d0, m_s0) — restored | 0.55522 | 90.3011 |

Same arguments, two states, 7% apart in n_B. A finite-difference Jacobian taken
across a seed change of that kind is not the derivative of anything.

**How it is closed.** The branch is a *declared property of the adapter*, and
the adapter's seed is a pure function of its arguments:

```python
def enjl_phase(par, branch, mu_B, mu_C, mu_S=0.0, T=0.0):
    """branch in {'broken', 'restored', 'deconfined'} -> PhaseThermo."""
    x0 = _branch_seed(branch, par, mu_B, mu_C, mu_S)   # pure; no state
    ...
```

with `_branch_seed` depending on nothing but `(branch, par, mu_B, mu_C, mu_S, T)`
— it may scale with mu_B, it may not remember. Concretely: `broken` seeds the
light masses at the vacuum solution, `restored` at `(m_u0, m_d0, m_s0)` (this is
`solver._restored_branch`'s rule, already in the repo), `deconfined` likewise
but with the baryon densities seeded at zero. `MixedCtx`'s existing warm-start
cache must **not** be extended to this adapter; the whole point of caching in
`eos/mixed` is speed, and here it would change physics. A post-solve assertion
that the returned state is still on the declared branch (light M_q above/below a
threshold, n_p + n_n above/below one) makes a silent hop loud.

That single sentence — *branch declared, seed pure, cache off* — is the answer
to §4 of the brief.

---

## 4. More than one window: the size of generalizing the locator

`eos/mixed/solvers/sweep.py::locate_window` returns one `MixedWindow`. It probes
the grid, brackets the chi = 0 and chi = 1 crossings, and bisects each. Its
inner helper is already `crossing(target, above=None)`, where `above` "restricts
the search to densities beyond a boundary already found".

**Size: small — roughly 30 lines and no change to `locate_window` itself.** A
`locate_windows(...) -> list[MixedWindow]` that calls `locate_window` in a loop,
each time with `hint=(previous n_offset, grid[-1])`, and stops when the returned
window does not `exist`, covers `fq0.5_B1`'s two windows. `MixedWindow.exists`
and `.reason` already give the loop its termination condition, and `.probes`
already lets each pass reuse the last one's solves. `eos/mixed/api.py::eos_table`
already returns `(rows, windows)` as a list per line, so the return type at the
public boundary does not change shape.

**The part that is not small, and is not this.** A window found by that loop is a
window *of one adapter pair*. ENJL's three branches admit three pairings —
broken/quarkyonic, quarkyonic/deconfined, broken/deconfined — and
`Beta_fq0.5_B1.dat`'s two windows need not be the same pairing. Finding all of a
parameter set's transitions therefore means running the engine once per pairing
and merging, which is a **caller-level loop over `enjl_phase` branch pairs**, not
a change to `eos/mixed`. That is the right split: the engine stays a two-phase
engine and knows nothing about how many branches a model has.

Recommendation: ship `locate_windows` as the thin loop; do the branch-pair sweep
in `eos/enjl`'s own driver, not in `eos/mixed`.

---

## 5. The eta family, and what minimizing over eta can and cannot mean

`eos/mixed`'s eta is a lepton-neutrality interpolation and nothing else.
`mixed.tex` §"The eta family of constructions" says so in its own words:
"Physically the intermediate case stands in for the finite surface tension and
Coulomb energy of the mixed-phase structures ... here it is a controlled
interpolation, **not a derivation from a surface tension**."

**The structural argument, and the loophole checked.** At eta = 1 each phase is
separately neutral; at eta = 0 only the average is. The eta = 1 constraint set is
strictly stronger, so every eta = 1 state's *matter* configuration is feasible at
eta = 0. The brief flags the lepton bookkeeping as the place a loophole could
live, and it does not survive: an eta = 1 state carries two local lepton gases at
different mu_e^L, and merging them into one global gas at a common mu_e at fixed
total lepton number *lowers* the lepton free energy by convexity. So the eta = 0
feasible set contains a state at or below every eta = 1 state's f. The
expectation that a minimizer returns eta = 0 is structural, and one functional
versus two does not enter it.

**Where the argument genuinely stops.** At *intermediate* eta both populations
exist simultaneously with weights eta and 1 - eta and enter eps, P and s
additively (`mixed.tex` Eqs. for P, eps, s). That is not a constrained version of
one physical system, so f(eta) on 0 < eta < 1 is not a variational quantity and
"the minimum of f over eta" is not defined by the argument above — only its
endpoints are comparable. This is worth saying out loud because it means an
interior minimum, if one showed up numerically, would be an artefact of the
weighting rather than physics.

**What Phase B must therefore do, and must not do.** Compute f(eta) on a grid —
eta in {0, 0.25, 0.5, 0.75, 1} at three or four densities inside each window —
tabulate it, and report the curve with numbers. Do **not** ship a minimizer whose
zero is the argument above restated as code. If the curve is monotone, say so
with the values. What would make an interior minimum meaningful is a surface +
Coulomb contribution to f — the pasta calculation, `eos/enjl`'s Paper 2, and a
separate project.

(The framework is Constantinou, Han, Jaikumar & Prakash, PRD **107**, 074013
(2023) and arXiv:2506.20418; `mixed.tex` cites both. Its own statement of what
eta means should be checked against §5's reading before Phase B reports.)

---

## 6. Phase B, in order

1. `eos/enjl/adapters.py` — `enjl_phase(par, branch, mu_B, mu_C, mu_S, T=0.0)`
   over `thermo_from_mu`, returning `eos.mixed.adapters.PhaseThermo`. Branch
   declared, seed pure, no cache. `T` present and raising for T != 0.
2. `eos/mixed/solvers/sweep.py` — `locate_windows`, the thin loop of §4.
3. The construction driver in `eos/enjl`: for each branch pair, locate windows,
   at eta = 1 report (n_onset, n_offset, mu_B, P).
4. Gates: the four windows of §2's table — P to 0.1%, both edge densities to 1%,
   mu_b read as `munr` where finite else `mun`. On `fq0.5_B1` restrict to
   `~numpy.isnan(col["munr"])`. Nothing gated on `fq0.7_B0` below n_b = 1.
5. `eos/enjl/verify/run_full_check.py` — add P and mu_B equal across each pair,
   and the delivered table monotone in P with 0 <= c_s^2 <= 1 (CLAUDE.md §8).
6. f(eta) tabulated and reported per §5.

Everything in that list takes a `T` argument defaulting to 0 and raising
otherwise, so session 4 extends the construction rather than reopening it.
