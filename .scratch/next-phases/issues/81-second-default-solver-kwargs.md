# Should the bare solver `include_*` kwargs follow §4's flags to False?

Type: grilling
Status: resolved
Blocked by: -
Parent: ../map.md

## Question

Raised by [ticket 65](65-species-flag-defaults.md), which unified §4's six
`SpeciesFlags` defaults on all-False and was explicitly scoped to the
dataclass. It exposed that the same sectors carry a **second default one layer
below**, and the two now disagree.

`zl`, `vmit` and `alphabag` take bare solver keyword arguments defaulting
**True**:

    solve_beta_eq_neutrinoless(n_B, T, params=None, include_photons=True, ...)
    # alphabag also: include_gluons=True, include_thermal_neutrinos=True

and `dd2`'s `solver.solve` carries `include_photons=True` and
`include_muons=True`. `SpeciesFlags` reaches these only through `api.py` /
`table.py`, which translate `species.photons -> include_photons` correctly.

So §4's "a sector that is off is off because its flag says so" is honoured at
the dataclass and violated below it. Measured: `eos_point` with no `species`
gives ΔP = 2.85e-04 MeV/fm³ against the bare `solve_beta_eq_neutrinoless` at
T = 10 — exactly the photon pressure — which is what turned
`test/vmit/test_uniform_api.py` and `test_tables.py` red in ticket 65. Both
were made explicit on **both** sides as a holding fix; neither now inherits
any default, so this ticket is free either way.

**A coverage gap rides along.** `zl`, `vmit` and `alphabag` moved **0**
baseline keys under ticket 65 *because the generator calls their raw solvers
and never constructs a `SpeciesFlags`*. Their public default did move —
`alphabag.eos_point` at T = 30 went 156.3823 -> 156.2985 MeV/fm³. So
`test/baseline/` does not exercise the `SpeciesFlags` -> solver wiring for
those three models at all, and whatever is ruled here should say whether that
gap is closed by a baseline case or left as a `verify/` entry.

**The candidates.**

1. **Flip them to False**, so there is one default per sector everywhere. The
   consistent reading of §4, and it moves numbers: `zl`, `vmit` and `alphabag`
   baselines would move for the first time, needing the same measure-then-
   regenerate gate ticket 65 used.
2. **Delete the kwargs**, making the flags object the only way to name a
   sector. Largest diff; the raw solvers are called directly in `verify/`
   suites and the baseline generator, so every call site would have to route
   through `SpeciesFlags`.
3. **Rule them internal**, on the ground that §5's public boundary is
   `eos_point` / `eos_table` / `eos_response` and those are already correct.
   Costs no numbers; requires saying so where a physicist calling
   `solve_beta_eq_neutrinoless` directly would read it, because the baseline
   generator itself calls them that way.

Whichever is chosen, the drift check belongs beside
`test_the_six_species_flags_all_default_to_off` in `test/test_imports.py`,
which is the precedent ticket 65 set.

## Ruling

Agreed with the user across five rounds of grilling. The ticket asked whether
one default should flip; the measurement said the question was mis-scoped in
four separate ways, and each is ruled below.

**The headline.** `include_electrons` is not one of these kwargs at all — it is
§3's orthogonal `leptons=` flag wearing an `include_*` name, and ticket 70 has
already ruled on it. Of the rest, the answer is not a default at all: a solver
that accepts a `SpeciesFlags` must HONOUR it and carry no parallel kwarg, and a
solver that lacks one grows one. Then there is no second default to disagree.

### 1. The premise was wrong: dd2 does not disagree, it IGNORES

`dd2.solver.solve(par, n_B, flags, ...)` accepts a `SpeciesFlags` and never
reads `.photons` — only `api.py:112` and `table.py` translate it. Measured at
n_B = 0.32, T = 30:

    solve(par, 0.32, SpeciesFlags(photons=False), T=30)      P = 36.84136685
    solve(par, 0.32, SpeciesFlags(photons=True),  T=30)      P = 36.84136685
    solve(par, 0.32, SpeciesFlags(photons=False), T=30,
          include_photons=False)                             P = 36.81824551
    one photon gas at T = 30                                     0.02312133

Flipping the flag changes nothing to eight decimals; the difference between the
first and third rows is exactly 1.000000 photon gases. A flag accepted and
silently ignored is §4's failure in its strongest form, and invisible.

**This cannot be fixed for free.** `generate_baseline.py:293` builds
`SpeciesFlags(hyperons=False, deltas=False, muons=True)` — since ticket 65,
`photons=False` — and never passes `include_photons`. The frozen dd2 numbers
therefore CONTAIN a photon gas the flags say is off. Honouring the flag moves
**456 of 976 dd2 baseline rows** (81 at T=10, 321 at T=30, 27 each at T=40 and
T=60). The 429 T=0 rows, the golden SNM point and the published NMP/TOV values
do not move: photons vanish at T = 0.

Ruled: honour `flags.photons`, delete `include_photons` from `solve` and its
four wrappers, regenerate the 456 rows under ticket 65's measure-then-
regenerate gate. No double counting exists to worry about — verified
end-to-end, `eos_point(photons=True) - eos_point(photons=False)` is exactly one
photon gas in `zl`, `vmit`, `alphabag` and `dd2` (dP/P_gamma = 1.0000000000),
`default_guess` and both `eos/mixed/adapters.py` seeds already pass
`include_photons=False`, and those seeds discard P/eps/s.

### 2. The three §4 kwargs are deleted, not re-defaulted

`zl`, `vmit` and `alphabag` already carry a `SpeciesFlags` with exactly the
fields needed, `gluons` included. The kwargs are a parallel spelling of an
object the models already own and whose `api.py`/`table.py` already translate.

Ruled: their solvers take `par` (first, required), `flags: SpeciesFlags`
(required), and `leptons=` on the fixed-Y_C modes only. `include_photons`,
`include_gluons` and `include_thermal_neutrinos` are deleted into the flags
object; roughly 40 lines of pass-through plumbing in `alphabag/solver.py` go
with them. 72 call sites outside the three `solver.py` files (zl 19, vmit 21,
alphabag 32) plus 6 in the baseline generator.

**Zero baseline movement in these three.** Because `flags` is REQUIRED rather
than defaulted, the ruling stops deciding the numbers and the call site decides
them: the generator passes `SpeciesFlags(photons=True, ...)` at its 6 sites and
reproduces the frozen rows exactly. Verified by the existing baselines staying
green, not by a regeneration. This is what candidate 1 could not offer.

**Ticket 82 becomes decisive.** This ruling does NOT rule on the VALUE of
`alphabag.gluons`; it removes the second place that value could be stated, so
[ticket 82](82-alphabag-gluons-default.md)'s ruling on the dataclass field is
the whole answer rather than half of one.

### 3. `include_electrons` is §3's `leptons=`, and stays its own argument

§5: the `leptons` flag "is NEITHER \[a condition nor a freeze target\], and is
an explicit named argument". It never enters `SpeciesFlags`. It only looked
like a member of the `include_*` sector family by sitting next to them in the
signature — which is why this ticket swept it up.

The two are independent, and all four combinations are distinct:

    dd2  fixed_YC  n_B=0.32  Y_C=0.3  T=10   (Y_C n_B = 0.096 fm^-3 to neutralize)
     leptons=  flags.muons |      n_e      n_mu  n_e+n_mu   P_total
         True         True |  0.056088  0.039912  0.096000  32.826772
         True        False |  0.096000  0.000000  0.096000  34.503713
        False         True |  0.000000  0.000000  0.000000  27.738909
        False        False |  0.000000  0.000000  0.000000  27.738909

`leptons=` asks whether a neutralizing lepton sector is added; `muons` asks
which families exist. Rows 1 and 2 differ by 1.68 MeV/fm^3 at the same total
lepton density; `leptons=False` collapses the muon distinction entirely.

Ruled: rename `include_electrons` -> `leptons`, still a separate named
argument. And its DEFAULT, which §3 never stated and nine models disagree on
(`False` in dd2/sfho/did/alphabag, `True` in enjl/njl/ccdm/zl/vmit and
`mixed/solver.py:716`), is **False** — ticket 65's "off unless asked", and
`leptons=True` is the one direction that silently ADDS a sector. Free of §12
cost: all 16 `leptons=`/`include_electrons=` sites in the baseline generator
pass it explicitly, so no row moves either way.

### 4. §5 binds `api.py`; §13 binds the solvers, and harder

Candidate 3's reading is correct — §5's public boundary is
`eos_point`/`eos_table`/`eos_response`, and those were already right. But §13's
"the same job carries the same name in every model, so a physicist who has read
one can read the next without a translation table" binds `solver.py` too, and
the `include_*` default was the SMALLEST of the divergences there. Three
argument orders exist: `(par, n_B, flags, ...)` in dd2/sfho/did,
`(n_B, T, params=None, ...)` in zl/vmit/alphabag, and
`(n_B_fm, Y_C, T=0.0, par=None, flags=None, ...)` in njl/ccdm/enjl. `par` is
required in two models and optional in seven, and spelled `params` in three.

Ruled by the user: **`par` first and required everywhere**, `params=` renamed
to `par=`, `n_B_fm` renamed to `n_B` (87 sites). Break the signature once,
since it is being broken anyway.

### 5. Renaming `n_B_fm` exposed a natural-units leak in three models

`n_B_fm -> n_B` collides with the natural-units twin in four functions —
`enjl/solver.py:186,520`, `enjl/verify/run_full_check.py:130`, and
`dd2/solver.py:101` where the sense is reversed. In `enjl/solver.py:576` the
bare name is already taken by the other unit: `n_B = n_B_fm * hc3`.

Ruled: fm-based names carry no suffix; natural-units working variables carry
`_nat`, the convention `dd2/solver.py:159` already uses unprompted. This covers
the whole family — `n_C_fm` (12), `n_S_fm` (8), `eps_fm` (7), `P_fm` (7),
`s_fm` (6) — and `n_b_fm`, which is a RESULT FIELD and therefore **234 keys in
`test/baseline/enjl.npz`**: a key rename with identical values, verified by
comparing old and new arrays key-for-key.

**And the audit it prompted found a real §5 violation in three models.** The
user's rule — results carry MeV/fm^3 for P and fm^-3 for n_B — measured over
every public result surface:

                     outer (fm-based)      inner record        ratio
    njl   .state     P    146.939710       1.128356e+09      7679041
                     n_B    0.800000       6.146804e+06      7683505  <- exactly hc3
    ccdm  .state     P     21.167100       1.619120e+08      7649229
                     n_B    0.898148       6.900926e+06      7683505  <- exactly hc3
    enjl  .point     P      2.058714       1.581814e+07      7683505
                     eps  152.450900       1.171357e+09      7683505

`n_B` divides out to exactly `hc3`; **P, eps and s do not**. The inner record
is matter-only, with the lepton and photon sectors missing:
`njl .state.P / hc3 = 146.854334` against an outer `146.939710` — a silent
0.085376 MeV/fm^3. A caller who spots the unit problem and corrects by `hc3`
still gets a wrong answer. `enjl` compounds it: `BetaPoint.point` is reached as
`got.point.point` at `test/enjl/test_enjl_modes.py:201`, because
`api.PointResult.point` holds a `BetaPoint`.

Ruled: **drop the natural-units record from every public result** in all three,
lifting what callers need onto the outer point in fm. Usage is small — 24 sites
across njl/ccdm reaching for `n_*` (10), `euler_residual` (5), `n_q` (3) and
`P` (1); `euler_residual` is dimensionless and stays reachable through an
explicitly internal path. No values move. `dd2`, `sfho`, `zl`, `did`, `vmit`,
`alphabag`, `abpr` and `eos/mixed` are clean on every path audited.

The rule reaches `n_s` and `s` as well — both are densities, both are returned
by every model under §11, and leaving them out reopens the hole one field over.
Masses are MeV in both systems and need no rule.

### 6. What keeps it from drifting back

The precedent this ticket named — `test_the_six_species_flags_all_default_to_off`
in `test/test_imports.py` — gains two neighbours: the `leptons=` default check,
and a parametrised units check over all ten models that walks every float field
on `eos_point`'s result and asserts fm-plausible MAGNITUDE BANDS. It catches a
natural-unit field by six orders of magnitude, so it needs no tolerance and can
never become a second baseline to maintain. It is the check that found §5 above.

### Sequencing: three execution tickets, split by what must be re-measured

The map's hard rule is "only the changes a ticket asks for", and one commit
spanning all of this makes a bisect useless the next time a baseline moves.

- **[Ticket 89](89-dd2-honours-species-flags.md)** — the dd2 ignored flag and
  its 456-row regeneration. The ONLY commit that moves frozen values.
- **[Ticket 90](90-solver-signature-and-units-sweep.md)** — the signature and
  units sweep. No values move, so a full-suite green plus unmoved baselines is
  the whole gate. Carries the 234-key enjl rename.
- **[Ticket 91](91-leptons-default-and-drift-checks.md)** — the `leptons=`
  default, the legacy `TableSettings` layer, and the three `test_imports`
  checks.

Three sentences are owed to CLAUDE.md via
[ticket 85](85-claudemd-sentences-owed.md): §2's Naming block gains the
`_nat` convention, §3 states that `leptons=` defaults to False, and §5's units
sentence names `s` alongside `n`, `T`, `mu`, `eps` and `P`.

**Noticed** (map hard rule — these go to the Stage 7 report): two tickets
shared the number 88, `88-fixed-composition-coexistence.md` and
`88-invert-nmp-basin-lottery.md`. **Fixed when [ticket 67](67-dd2-t0-adoption.md)
landed**, which is the ticket that spun the second one out: it is now
[ticket 93](93-invert-nmp-basin-lottery.md), and the map, ticket 67 and
`docs/DEFERRED.md` all point at the new number.

Open for execution.

---

## Note from [ticket 82](82-alphabag-gluons-default.md) (2026-08-26)

Written after this ticket was resolved, and it does not reopen anything —
§2's answer (delete the three kwargs into the flags) already covers `gluons`.
What it adds is a measurement §2 could not have had, and one consequence for
[ticket 90](90-solver-signature-and-units-sweep.md), which executes §2.

Ticket 82 flipped `alphabag.SpeciesFlags.gluons` `True -> False`, on the rule
that a flag with two legal values is a default and is off. So all three of
alphaBag's thermal sectors now read the same way: flag `False`, bare solver
kwarg still `True`, pending §2.

**The consequence.** `test/baseline/case_alphabag` calls the raw solvers and
names none of the three kwargs. Once §2 deletes them into the flags, those
rows pick up the flag defaults, which are now all `False` — so the deletion
MOVES alphaBag baseline values. Ticket 90 currently says "no value moves in
any of it". That was already untrue for `photons` and `thermal_neutrinos`
(ticket 65 flipped those defaults); ticket 82 adds `gluons` to the same list.
Sizes, at n_B = 0.8 through `eos_point`:

    beta.T0     P unchanged (every thermal sector vanishes at T = 0)
    beta.T10    P  -1.465838e-03 MeV/fm^3   gluons alone
    beta.T30    P  -1.187329e-01 MeV/fm^3   gluons alone

Ticket 82 itself moved NO baseline key, because the flag never reaches those
raw solvers today — which is exactly the coverage gap §2 closes.
