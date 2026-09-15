# Per-pattern verdict: does the ansatz reach the targets, and if not, why not

Type: grilling
Status: open
Blocked by: 03, 05
Parent: ../map.md

## Question

This ticket closes milestone 1. It asks one thing: **with closed-form
derivatives (01), a genuine conformal limit (02) and finite T (05), does each
pattern separately reproduce `eos.njl` at the stated targets over 3-10 n_sat?**

It is a grilling ticket rather than a task because the interesting outcome is
not the table -- ticket 03 and ticket 05 already produce most of it -- but the
JUDGEMENT about rows that miss, and whether the miss is a fit that needs more
work or a physics the ansatz structurally cannot carry. Those two have opposite
responses and the difference is not visible in the residual.

### The table to fill in

Per pattern (unpaired, 2SC, CFL), per mode, `n_B` = 1-15 n_sat with 3-10 n_sat
weighted heaviest, error normalized by `max|P|` over the window:

| | target |
|---|---|
| T = 0, beta equilibrium | median <= **1%**, max <= 3%, `c_s^2` within **10%** |
| T = 0, `Y_C` = 0.5, `Y_S` = 0, no leptons | median <= **2%**, max <= 5% |
| T = 0, `Y_C` = 0, `Y_S` = 0, no leptons | median <= **2%**, max <= 5% |
| T = 0-100 MeV, `Y_C` = 0.01-0.5, `mu_S` = 0 | median <= **3%** in P, `s` within **10%** |

with the per-pattern fit rms in `P`, `n_B`, `n_C`, `n_S`, `s` beside it.

Phase agreement is NOT gated here -- it is milestone 2's subject and tickets 07
and 08 own it. But record it, because ticket 07 starts from these numbers.

### The four things worth grilling, each with a measurement behind it

**1. The low-density end, and whether the domain should be stated rather than
fixed.** Measured: *"the ansatz is at its best deep inside the quark phase and
at its worst near the onset, which is precisely where a hybrid star puts the
quark core"* (`docs/csc_bag_mapping.md:570-576`), and the cause is named --
*"at the low-density end the NJL constituent masses are still running hard, and
a bag model with a fixed `m_s` and massless `u`, `d` has nothing to run."* Also
measured: accuracy over 2-15 n_sat is 2-3x worse than at high density *"and
that is not fixable by fitting harder."* So the honest options are a density
dependence for `m_s`, more fixed-mass gases, or a stated lower domain with the
model matched to something else below it. **Which, and on what evidence?**

**2. `n_S` is the worst row everywhere, and its worst point is always at the
extreme of the `mu_S` grid**, where the strange sea is nearly empty
(`docs/csc_bag_mapping.md:544-547`). The three fixed-mass gases took it 9.5% ->
7.6%. Is that good enough for the application? A merger table's strangeness
content matters; an inference run's may not. **The target should be set by what
uses it, and nobody has said.**

**3. `pair` goes to zero and the LO condensation term does not survive.**
Measured twice (19.2% in the docs, 20.7% in the code -- two runs, treat it as
~20%): pinning `pair = 1` costs a factor of a hundred, and left free the
coefficient pegs at its lower bound. The pairing physics reaches the model only
through the per-pattern coefficient set and through `(Delta_star, sigma)`. That
is a RESULT, stated as such -- but it means the model's pairing content is
almost entirely empirical, and `pqm.tex` has to say so plainly rather than
presenting a condensation term that is multiplied by zero. **Does `pair` stay
in the basis at all?**

**4. Colour potentials are absorbed, not represented.**
`docs/csc_bag_mapping.md:58-62`: *"`mu_3` and `mu_8` are NOT in this list ...
the bag model has no colour potentials at all, so whatever their cost is, the
fit absorbs it into the other parameters. This is stated here because it is a
real approximation, not an omission."* Off the beta-equilibrium line the
colour-neutrality cost varies differently, and nothing in the ansatz tracks it.
`notebooks/csc_dataset.py:186-187` stores `mu_3` and `mu_8` per row -- the only
place they survive -- so the size of the absorbed cost is MEASURABLE from the
existing data. **Measure it.** If it correlates with the residual, that is the
next basis term; if it does not, the approximation is safe and the `.tex` can
say so with a number.

## Gate

- The table above, filled in, per pattern and per mode, with the interpreter
  and library versions named.
- **An explicit verdict sentence per benchmark row**: at target, or missed by
  how much and for which of the two reasons -- a fit not yet good enough, or a
  physics the ansatz cannot carry. No row left ambiguous.
- A decision on the low-density domain: extended, or stated and bounded, with
  the evidence.
- A measured statement of the `mu_3`/`mu_8` absorption cost, and whether it
  correlates with the residual.
- A decision on whether `pair` stays in the basis.
- Anything that cannot be met recorded for `docs/DEFERRED.md` per CLAUDE.md
  section 3 -- a gap recorded is a decision, a gap not recorded is a bug.
- Milestone 1 closes, or this ticket says what it is waiting for.
