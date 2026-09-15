# Does `free` belong in the default enumeration?

Type: grilling
Status: closed
Assignee: guerrinimirco
Blocked by: 05
Parent: ../map.md

## Question

[The cheap pre-screen](05-cheap-pre-screen.md) measured that the `free`
candidate is **91.9% of the default 200-point CSC build** and changes **no
delivered row** of it. Dropping it is 12.5x, with zero `pattern_realised`
mismatches and worst |dP|/P = 6.654e-10. That is the largest single number in
this map, and it cannot be taken as a solver optimisation, because what it
removes is a physics probe.

**Is `free` dropped from `DEFAULT_PATTERNS`, kept and made cheap, or kept as
it is and the benchmark restricted?**

### What is settled, and must not be re-derived

- `free` shares CFL's layout (`PATTERNS` in `eos/general/pairing.py`) and
  differs only in its seed, (0.3, 0.6, 1.0) x gap scale against CFL's
  (1, 1, 1). Its purpose is the states no named seed aims at: uSC, dSC, sSC.
- It can **never** carry a warm start, because `realised_pattern` never
  returns `'free'` and `solve`'s seed filter tests `pattern_realised ==
  pattern`. This is structural, not a bug in the sweep.
- On the pinned benchmark it realises CFL 141 / 2SC 26 (duplicates) and uSC 33,
  of which **27 do not converge**. The 8 uSC states that converge lose by
  **+29.6 to +41.5 MeV/fm^3** in f. Never close, never winning.
- Warm-starting it (V1) **captures** it onto one root for the whole sweep —
  200/200 densities as 2SC. A captured probe is a no-op wearing a candidate's
  name, which is worse than no candidate.
- Keeping the probe honest by re-hunting it cold every 10 densities (V2) costs
  **6x of the 12.5x**: 4496 against 730 ms/pt.

### What this ticket has to weigh

- **One benchmark is one parameter set, at T = 0, in beta equilibrium.** uSC
  losing by 30 MeV/fm^3 here says nothing about a larger m_s mismatch, a
  different G_D, or T > 0 — which is where uSC/dSC are argued for in the
  literature. What evidence would be enough?
- `DEFAULT_PATTERNS` is in `eos/general/pairing.py` and is shared with every
  model that pairs, not only `njl`. A change there is library-wide.
- CLAUDE.md section 4's rule that a sector is never switched off implicitly
  bears on this: if `free` leaves the default, the statement has to be made
  somewhere a caller can see it, not by a quiet edit to a tuple.
- The map's own gate is "same pattern, P to 1e-8". `free` passes that test by
  contributing nothing — which is exactly why the gate cannot decide this one.

### Options on the table 

1. **Drop `free` from `DEFAULT_PATTERNS`**; callers who want it pass it. 12.5x,
   and the probe is gone unless asked for.
2. **Keep it, re-hunted on a bounded window** (V2's shape). ~2x, the probe
   survives with a stated window, and the window length becomes a documented
   number: a uSC region narrower than it can be missed.
3. **Keep it as it is** and restrict the benchmark, conceding that a default
   CSC table costs 9 s/pt.
4. **Seed it differently** — from a converged CFL/2SC root perturbed
   asymmetrically rather than from the unpaired state. Unmeasured; it may
   collapse the hunt cost without capture, or may simply bias it back onto CFL.

## Gate

- A decision, with the reason stated in terms of what `free` is FOR rather
  than what it costs.
- If it leaves the default: where the statement lives, and what a caller who
  needs uSC/dSC is told to pass.
- If it stays: the number that bounds its cost, and the ticket that measures
  whether option 4 makes option 2 cheap.

## Resolution, 2026-09-14

**`free` leaves `DEFAULT_PATTERNS`**, in both models, and the asymmetric sector
is reached under the names of its states. The reason is not the 12.5x: **`free`
is a dominated probe**. Measured cold, every candidate solved alone,
`rg_njl1`, beta-eq, `backend="reference"`, 12 points over T = 0/30/50 and
n_B = 0.8-2.0 (`../t12_probe.py`, `../t12_probe.json`) -- `f - f_winner` in
MeV/fm^3, `X` = did not converge:

```
   T   n_B |   unpaired          2SC          CFL          uSC          dSC         free
   0  0.80 | unpr +163.8   2SC  +24.2   CFL   +0.0   2SC  +24.2   2SC  +24.2   2SC  +24.2
   0  1.20 | unpr +284.1   2SC  +86.0   CFL   +0.0   uSCX +64.6   dSC  +71.8   uSCX +64.4
   0  1.60 | unpr +384.2   2SC +134.0   CFL   +0.0   uSCX +90.8   dSC  +94.1   uSCX +90.8
   0  2.00 | unpr +473.2   2SC +173.2   CFL   +0.0   uSCX+112.7   dSC +114.4   uSCX+112.7
  30  0.80 | unpr +137.0   2SC  +11.9   CFL   +0.0   2SC  +11.9   2SC  +11.9   CFL  +12.4
  30  1.20 | unpr +246.9   2SC  +66.9   CFL   +0.0   uSC  +51.5   dSC  +56.8   2SC  +66.9
  30  1.60 | unpr +340.0   2SC +110.9   CFL   +0.0   uSC  +75.9   dSC  +78.6   uSC  +75.9
  30  2.00 | unpr +422.4   2SC +146.6   CFL   +0.0   2SC +146.6   dSC  +97.1   uSC  +95.7
  50  0.80 | unpr +101.5   2SC   +0.0   CFL   +3.6   2SC   +0.0   2SC   +0.0   2SC   +0.0
  50  1.20 | unpr +186.8   2SC  +37.7   CFL   +0.0   uSC  +31.3   2SC  +37.7   uSC  +31.3
  50  1.60 | unpr +266.6   2SC  +74.2   CFL   +0.0   2SC  +74.2   2SC  +74.2   2SC  +74.2
  50  2.00 | unpr +337.3   2SC +103.4   CFL   +0.0   2SC +103.4   2SC +103.4   2SC +103.4
```

### What `free` is FOR, and why that is the argument for dropping it

The ticket's premise was wrong in its most load-bearing clause. `free`'s
purpose is stated as "the states no named seed aims at: uSC, dSC, sSC" -- but
**`uSC` and `dSC` are named patterns with their own seeds** (`PATTERNS` in
`eos/general/pairing.py:1636`: `uSC = (0, 0.6, 1.0)`, `dSC = (0.6, 0, 1.0)`).
Only `sSC`, `usSC`, `dsSC` and unequal-gap CFL states have no seed aiming at
them. And both named seeds are in `_REALISED`, so **both can carry a warm
start**, which `free` structurally never can -- that is not a quirk of `free`,
it is the only member of `DEFAULT_PATTERNS` whose name no solved gap vector can
return.

Against its two rivals at the same densities, `free` is beaten on every axis
that matters to a probe:

- **It finds nothing they do not.** In **11 of 12** points every asymmetric
  state `free` reached was reached by `uSC` or `dSC` as well. In **1 of 12**
  (T = 30, n_B = 2.0) it found a uSC state neither named seed did -- **losing
  by +95.7 MeV/fm^3**.
- **It is less reliable.** At T = 30, n_B = 1.2 it collapsed to 2SC while
  *both* named seeds held their own asymmetric layouts. At T = 0 it and `uSC`
  stall on the same uSC state, while `dSC` converges in 0.5 s where `free`
  takes 5.5-8.5 s and fails.
- **It is not even cheap where it works.** Cold at T = 0 it costs +5.7 s
  against the pair's +5.3 s; at T = 30, +1.0 s against +1.5 s. The 12.5x comes
  from the sweep, where it is re-hunted cold at every density forever.

So the decision is not "the probe is too expensive". It is that a probe which
is dominated by the named candidates on 11 of 12 points, cannot be cached, and
is the *worse* finder where an asymmetric root demonstrably exists, is not a
probe -- it is an uncacheable duplicate. **A caller who wants the asymmetric
sector asks for it by the name of the state.**

No asymmetric state wins anywhere in the box; `dSC` is a converged real state
at T = 0 and 30, always losing to CFL by 57-114 MeV/fm^3. The melting region
of Gholami, Hofmann & Buballa PRD 111, 014021 (2025) -- `njl.md` section 17.8,
the passage that says `free` exists so a CFL-layout solve can fall onto a dSC
state -- **is simply not inside n_B = 0.8-2.0, T <= 50.**

### The evidence question the ticket asked

Twelve cold points at one parameter set is **enough to call `free` dominated**:
that claim is about one seed against two rivals at the same densities and does
not need the phase diagram. It is **not enough to say the asymmetric sector
never wins** -- the melting region is outside the box, and nothing here varies
`G_D`/`eta_D` or `m_s`, which is where uSC/dSC are argued for. Locating it is
owed by [Should `uSC` and `dSC` join the default enumeration?](14-usc-dsc-in-the-default.md),
not by this ticket: it is a physics result, and this map has no business
producing one under a speed ticket.

### The baseline exposure is two numbers at one density, and they survive

The njl baseline's sweep half is `csc=False`, and its paired half is explicitly
pattern-restricted at four points. **Exactly one entry uses the default
enumeration** -- `enumeration.n1.2.{f,Delta}` -- and
`test/baseline/generate_baseline.py:665` says it is there "so a change in the
enumeration is still visible". Measured, 4-pattern against 3-pattern at
n_B = 1.2, T = 0: the winner's LABEL flips `free` -> `CFL` (**both realise
CFL**, and no label is stored), while

| | |
|---|---|
| rel `df` | 1.93e-13 |
| rel `dP` | 7.59e-13 |
| rel `dDelta` | <= 1.8e-13 |

-- inside `test/baseline`'s rtol = 1e-10 by ~500x. Ticket 05's worst 6.654e-10
was a warm-started sweep point, not this one.

### Scope: one default, both models

`DEFAULT_PATTERNS` is consumed by `eos/njl/solver.py:846`,
`eos/ccdm/solver.py:682` and **both** `eos/mixed/adapters.py` phases
(`njl_phase` 1225, `ccdm_phase` 1467). The default moves for both. The
argument for dropping is structural -- `realised_pattern` can never return
`'free'`, so the candidate can never be cached -- and that is a property of the
gap matrix, which is precisely why `pairing.py`'s own comment puts the table in
`general/`: "a property of the gap matrix above, not of the Lagrangian that
supplies G_D". A model-local default would be the first fork in a table that
exists to not be forked. `ccdm` gets a confirmation run, not its own policy.

### Where the statement lives

`free` stays a legal pattern and nothing raises, so the statement goes where
the default is already stated -- four sites, **no new constant**. Notably NOT
anything like `table.py`'s `NOT_A_BRANCH`: that constant exists to make a name
RAISE, and `free` must stay requestable.

1. `eos/general/pairing.py:1631-1650` -- the comment block above `PATTERNS`
   already says what `free` is for; it gains why it is not enumerated by
   default. Plus the `DEFAULT_PATTERNS` docstring.
2. `eos/njl/njl.md:1147` and `:1198` (the `patterns` API row spells the tuple
   out), `eos/njl/njl.tex:1242`.
3. `eos/ccdm/ccdm.md:921` and its `.tex`.
4. `eos/njl/njl.md` section 17.8 -- the passage making the physics claim, which
   gains the sentence that the asymmetric sector is reached by `uSC`/`dSC`
   under their own names.

Plus **one `docs/DEFERRED.md` entry, for the open physics question only** (where
the asymmetric sector wins), because `.scratch/njl-speed/` is untracked and
that question would otherwise die with this map. Not for the default change,
which is stated where the default is stated.

**CLAUDE.md section 4 does not bind this.** Section 4 governs flags and
sectors; the sector switch is `csc`, untouched -- `csc=True` still enumerates
the pairing sector. What moves is which seeds the enumeration tries, and that
is already an explicit, caller-passable argument with a documented default.
This is a seed list, not a sector.

### What a caller is told

Ask by the name of the state:

    patterns=("unpaired", "2SC", "CFL", "uSC", "dSC")

`free` stays legal, documented as the seed for what no pattern names -- the
`sSC`, `usSC`, `dsSC` masks `_REALISED` lists but `PATTERNS` offers no seed
for, and unequal-gap CFL states.

### One wrinkle handed forward

`_left_layout` (`eos/njl/solver.py:822`) drops a collapsed candidate only for
`("2SC", "CFL")`. A `uSC` or `dSC` candidate that falls to 2SC is kept and
competes -- exactly the "reported `pattern` column is not stable" defect
ticket 05 found with `free`, and the table above shows it in every T = 50 row.
**That filter must widen before uSC/dSC could join a default**, so it is a
precondition on ticket 14 rather than something for it to discover.

### Out of this ticket

- [Should `uSC` and `dSC` join the default enumeration?](14-usc-dsc-in-the-default.md),
  blocked on 13 -- the physics-honest successor, deliberately NOT bundled here
  because cold it costs what `free` costs (+5.3 s against +5.7 at T = 0; +1.5
  against +1.0 at T = 30) and its whole case rests on warm starts, which are
  unmeasured, and on ticket 13's ladder bound.
- [Land the pattern-default decision](15-land-the-pattern-default.md) -- the
  one-line edit, the four doc sites, the `ccdm` confirmation run and the
  reachable suites.
