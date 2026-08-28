# `Y_L` on a point is two different quantities wearing one name

Type: grilling
Status: open
Blocked by: -
Parent: ../map.md

## Question

Surfaced by [ticket 100](100-vmit-point-Y_S-never-assigned.md)'s first
instruction: **establish whether any other model has the same unassigned-field
shape before touching vmit.** It does, on a different field, and the answer is
a ruling rather than a fix — which is why it is here and not in that diff.

Ticket 100 swept every model's point against `eos.general.basis` on that
point's own solved densities, at n_B = 0.5 fm^-3, T = 0, every mode, default
flags. On **`Y_S` the sweep found `vmit` alone**, exactly as ticket 100
claimed. On `Y_L` it found three:

| model | mode | cached `Y_L` | (n_e + n_mu + n_nu)/n_B |
|---|---|---|---|
| `zl` | `beta_eq_neutrinoless` | 0.00000000 | 0.11398814 |
| `zl` | `fixed_YC` | 0.00000000 | 0.30000000 |
| `vmit` | `beta_eq_neutrinoless` | 0.00000000 | 0.00003555 |
| `vmit` | `fixed_YC` | 0.00000000 | 0.30000000 |
| `vmit` | `fixed_YC_YS` | 0.00000000 | 0.30000000 |
| `alphabag` | `beta_eq_neutrinoless` | 0.00000000 | 0.00005545 |

In all six the leptons are present and the field reads the dataclass default.

## Why it is NOT the same ticket as 100

Because `Y_S` had one meaning and `Y_L` has two, and nobody has ruled which.

- `eos/zl/solver.py:79` and `eos/alphabag/solver.py:97` both annotate the
  field `# electron-family lepton fraction (trapped mode)`. Read that way the
  zeros above are not a bug: outside `beta_eq_neutrino_trapped` there is no
  Y_Le condition, the field is inapplicable, and both models expose the
  measured lepton content as `Y_e` beside it.
- `eos/njl/solver.py:473` and `eos/ccdm/solver.py:520` compute
  `Y_L = (n_e + n_nu)/n_B` in **every** mode, so `njl` reports Y_L = 0.3 in
  `fixed_YC`. Read that way the zeros are the same defect ticket 100 fixed.
- `eos/did/solver.py:135` calls the field `Y_Le` and measures it in every
  mode — a third spelling of the same slot.
- `eos/vmit/solver.py:72` says only `# Lepton fraction`, committing to
  neither.

CLAUDE.md §13 rule 2 — the same job carries the same name in every model — is
violated whichever way it goes, so the decision is what the job IS:

- **(a) `Y_L` is the trapped mode's condition, echoed back.** Then `njl`,
  `ccdm` and `did` are the ones that drift, and the field should be absent or
  documented as inapplicable outside the mode that fixes it. Cheapest, and it
  makes the zeros correct as they stand.
- **(b) `Y_L` is the measured lepton fraction, in every mode.** Then it is
  ticket 100's defect three more times, `zl`/`vmit`/`alphabag` assign it from
  their own lepton densities, and one spelling wins over `Y_Le`.
- **(c) the field is derived and should not be cached at all.** CLAUDE.md §7's
  single-home argument, the position ticket 99 took for its own reader. Every
  point already carries `n_e`, `n_mu` and `n_nu`, so `Y_L` is one division the
  caller can do — and a second home for a derived quantity is precisely what
  went stale in ticket 100.

Note that `Y_C` has a milder version of the same question: `vmit`'s and
`zl`'s fixed-fraction solvers echo the requested `Y_C` while `alphabag`'s
report the solved one (0.29999999998989824 against a requested 0.3). Ticket
100 left that alone deliberately — the values agree to 1e-8, so nothing is
wrong today — but whichever way `Y_L` is ruled should say what `Y_C` is too,
since they sit two lines apart in the same dataclass.

## Blast radius

**Arm (b) moves `test/baseline/vmit.npz` and `zl.npz` a second time.** Ticket
100 has already regenerated `vmit.npz` once (39 `.Y_S` keys). So this ticket
**must not run concurrently with [ticket 95](95-vmit-solver-flags.md)**, which
moves vMIT rows for its own reason, and the two regenerations should be
sequenced rather than interleaved.

`eos/mixed/backends/jacobian.py:148` reads `st.Y_S` of a MIXED state, not of
any point above, so it is unaffected. Nothing in the package reads a `Y_L`
off one of these points.

## Gate

- One meaning for `Y_L` across the ten models, stated in CLAUDE.md §13's
  vocabulary list or in each dataclass, and `Y_Le` reconciled with it.
- `test/test_cached_fractions.py` extended to cover the field. It skips
  `Y_L` today, and says in its module docstring exactly why: there is no
  convention to assert yet. That paragraph comes out when this ticket lands.
- Any baseline key that moves is named, and the regeneration sequenced
  against ticket 95.
