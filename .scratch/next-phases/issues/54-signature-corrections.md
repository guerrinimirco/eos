# The public-signature corrections §5 and §3 require

Type: task
Status: resolved
Blocked by: 11, 44, 45
Parent: ../map.md

## Question

Five public-signature rows ruled (a) by [ticket 11](11-conformance-triage.md).
Blocked on [44](44-rename-dd2.md) and [45](45-rename-sfho.md) so it does not
collide with the renames already in flight through the same files — and because
tickets 42 and 43 both proved the same trap: **a rename onto a name this repo
already used for a LOCAL adapter fails silently.** Run their AST check here too.

1. **`leptons` smuggled through `**conditions`** (finding 16b).
   `eos/sfho/api.py:56`, `eos/dd2/api.py:53` and `eos/did/api.py:57` pop
   `leptons` out of the conditions bag, where it then mutates the mode into a
   name §3 does not define (`fixed_YC_neutral`, `eos/dd2/table.py:54`). §5 fixes
   the condition names at `n_B, T, Y_C, Y_S, Y_Le, Y_Lmu`, and §3 defines
   `leptons` as an orthogonal **flag**. Six models already make it an explicit
   named argument — `zl/api.py:66`, `vmit:66`, `alphabag:71`, `njl:122`,
   `enjl:79`, `mixed:98`. Make the three match the majority, and retire the
   invented mode name with them.

2. **`mode` acquired a default** (finding 15). `njl/api.py:73,121,154`,
   `ccdm/api.py:81,137,171` and `enjl/api.py:78,135,232` default to
   `"beta_eq_neutrinoless"`. §5 shows `mode` as a required positional, and the
   reasoning that makes `par` non-optional applies exactly: a default mode is a
   physics choice made on the caller's behalf. Drop it in the three.
   `abpr/api.py:73,146,203` keeps its default — one mode exists — and §5 gains
   the sentence permitting it via [ticket 22](22-phase5-claudemd.md).

3. **`zl.thermo_from_n(n_B, Y_C, T, params)` takes a mode's held fraction**
   (finding 8). `eos/zl/thermodynamics.py:374`, which then does
   `n_p = Y_C * n_B` / `n_n = (1 - Y_C) * n_B` at `:386-387`. It is the only
   non-docstring hit in the whole §5 purity grep (`grep -n "beta\|Y_C\|neutral\|
   trapped" eos/*/thermodynamics.py`), and it is exported publicly
   (`eos/zl/__init__.py:24,45`) and consumed by `eos/mixed/adapters.py:913`.
   Becomes `thermo_from_n(n_n, n_p, T, params)`, with the one adapter line
   following. The ruling's reasoning: `(n_B, Y_C)` is a legitimate
   re-parameterisation of `(n_n, n_p)` and the physics is not wrong — but it
   makes the grep test §5 publishes return a false positive a reader cannot
   distinguish from a real one, and that is the cost being paid.
   **Checked against `test/baseline/` for `zl` and `mixed`** at rtol = 1e-10.

4. **`TC_COEFF` has no override path** (finding 17a).
   `eos/alphabag/thermodynamics.py:50 TC_COEFF = 0.57 * 2**(1.0/3.0)` is the CFL
   critical-temperature coefficient, feeds `:410 T_critical(Delta0)`, is not a
   field of `eos/alphabag/parameters.py:37-61`, and `T_critical` takes no
   override — so an inference run over CFL pairing cannot vary it (§6). Move it
   into the parameter dataclass. **The default must reproduce
   `0.57 * 2**(1/3)` exactly**; checked against `test/baseline/alphabag`.

5. **`thermal_neutrinos` + the trapped mode: five models, two answers**
   (finding §3-ii). It **raises** in `sfho/solver.py:576` and `did/solver.py:213`
   and **succeeds** in `njl:275`, `ccdm:307` and `enjl:224-236`. §4 defines
   `thermal_neutrinos` as "neutrino flavors **NOT tracked in the matter
   composition** (e.g. the tau family …)" — under the trapped mode the e and mu
   families *are* tracked, so the flag legitimately means the tau family and the
   combination is meaningful. **The three that succeed are right**; `sfho` and
   `did` drop the raise. §4 gains the sentence saying so via
   [ticket 22](22-phase5-claudemd.md). If wiring the tau gas into sfho or did
   turns out to be more than a raise to delete, stop and report rather than
   guessing at the physics.

Items 1, 2 and 5 change no converged number; 3 and 4 are gated as stated. Report
added failures against `output/_audit/pytest_before_with_crust.txt`.

## Noted by [ticket 20](20-phase5-api-readme.md)

Item 1 has a table half the text does not state. `eos_table` accepts no
`leptons=` at all — in dd2 it goes straight into `TableSpec(mode=...)`
(`eos/dd2/api.py:129`), so the ONLY way to ask a table for the neutralizing
flavour today is the invented mode name `fixed_YC_neutral` this item retires.
Retiring it without giving `eos_table` the flag would make the neutral
fixed-Y_C table unreachable. Both entry points take the argument, or neither
name goes.

## Resolution

**All five rows landed; 0 added failures; `test/baseline/` unmoved.** Measured
on **python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0** (`python3`), collection
**1692** — not the 1665 or 1677 the map's earlier numbers imply; tickets 49, 61,
12, 52 and this one have all moved the denominator since.

**The measurement is an isolated PAIR, because the live tree was not mine
alone.** Ticket 52 held `eos/general/fermi_integrals.py`,
`eos/dd2/thermodynamics.py` and `eos/sfho/backends/jacobian.py` modified for
most of this session (they have since landed as `ffae9db` / `f1484b6`), so a
run in the repo measured both sessions and had no before-image. Two copies were
built from `git archive HEAD` (`407c984`) plus one snapshot of the gitignored
`test/`: a **control** carrying HEAD only, and a **mine** carrying HEAD plus
exactly this ticket's 22 files. Both ran the same subset.

    control (HEAD)              3 failed, 1291 passed   8:44
    mine    (HEAD + ticket 54)  3 failed, 1297 passed   8:45
    test/baseline/, both        6 failed, 10 passed

The **failure sets are identical**, diffed node id by node id. The three are
`test/dd2/test_api.py::test_inversion_without_Q_sat_predicts_it`,
`::test_inversion_with_Q_sat_still_imposes_it` and
`test/dd2/test_dd2_m8.py::test_restarts_recover_a_seed_limited_inversion` — the
dd2 NMP inversion, which [ticket 47](47-dd2-nmp-inversion.md) established is a
STACK artifact and not a code state. The six baseline failures (`ccdm`, `dd2`,
`enjl`, `njl`, `tov`, `zlvmit`) are the same six at HEAD, in the same set. The
**+6 passes are this ticket's two new test files**, three tests each.

`sfho`, `did`, `alphabag`, `zl`, `enjl`, `abpr`, `njl` and `ccdm` `verify/`
suites all report every check `[ok ]` on the live tree — including enjl's
`refusals`, which asserts in as many words that `photons` and
`thermal_neutrinos` do NOT raise.

### The AST check, run before and after

Clean both times, on both shapes. The cross-module shape (a name both imported
and defined in one file) found nothing for any of `thermo_from_n`, `T_critical`,
the five `cfl_*` functions, `mode_spec` or the three entry points. The
ticket-43 shape — a LOCAL binding of a name being introduced — found seven
pre-existing rebinds of `leptons` (`did/solver.py:659`, `njl/solver.py:541`,
`dd2/solver.py:285,707`, `sfho/solver.py:463`, `ccdm/solver.py:616`, and my own
`sfho/table.py:498`), and **none of them is a collision**: no function that
rebinds `leptons` also takes it as a parameter. A third check confirmed that
directly — 0 functions in `eos/` overwrite their own `leptons` or `tc_coeff`
argument. The two `_normalize` rebinds the pre-check flagged were exactly the
code item 1 deletes.

### Item 1 — `leptons` out of the condition bag

`sfho/api.py`, `dd2/api.py` and `did/api.py` take `leptons` as a named argument
on **all three** entry points (`eos_response` too, in sfho and did; dd2's
implements only freezes where the flag is meaningless). `_normalize` — which
popped the flag and then mutated the mode name — is `_check` in both sfho and
dd2, and now validates rather than rewrites: finding `leptons` in the bag is a
`TypeError` naming the named argument. That guard is not decorative and it is
not dead: with `leptons` a named parameter of `eos_point` the only live path is
`eos_table(fixed={...})`, and it **caught a real caller on its first run** —
`test/dd2/test_photons_flag.py` was passing `fixed=dict(Y_C=0.1, leptons=True)`,
which is precisely the smuggling [ticket 20](20-phase5-api-readme.md) predicted,
forced because `eos_table` had no `leptons=` to pass it to. Now it does.

**The invented mode names are gone from the repository.** `fixed_YC_neutral`
(dd2, sfho) and `fixed_YC_YS_neutral` (sfho) are deleted from `MODES`,
`MODE_FRACTIONS`, `_mode_kwargs`, `mode_spec`, `dd2/solver.py`'s
`solve_hadronic` docstring and sfho's `_settings_to_spec`. The flag rides beside
the mode instead: `MODES` carries `takes_leptons=True` on the fixed-fraction
entries, `TableSpec` gained a `leptons` field in both models, and asking for the
flag where it does not apply raises rather than being quietly dropped. A grep
for either name over `eos/`, `test/` and `docs/` returns nothing.

Ticket 20's note is satisfied on its own terms: **both entry points take the
argument**, so the neutral fixed-Y_C table stayed reachable. Proved rather than
asserted — `eos_table(..., "fixed_YC", ..., leptons=True)` on the new tree and
`eos_table(..., "fixed_YC_neutral", ...)` on the control were compared as raw
hex floats over 3 densities x 2 temperatures x {P, eps, s} in **both** dd2 and
sfho: **bit-identical, 12/12 rows**.

One incidental gain: the `progress` dict's `mode` key now always carries a §3
name, in both models, because there is no other kind of name left to carry.

### Item 2 — `mode` is required again

Nine defaults dropped, three per model in `njl`, `ccdm` and `enjl`. `abpr` keeps
its `mode="cfl"`, the exception §5 gains a sentence for via
[ticket 22](22-phase5-claudemd.md). An AST sweep over every `eos_point` /
`eos_table` / `eos_response` call in `eos/` and `test/` reports **0 call sites
with fewer than two positional arguments and no `mode=` keyword** — the two that
relied on the default were both in `eos/enjl/verify/run_full_check.py`
(`eos_response(par, n_B=0.3)` at :757 and `eos_point(par, n_B=8.0)` at :778) and
both now name `beta_eq_neutrinoless`.

### Item 3 — `zl.thermo_from_n(n_n, n_p, T, params)`

The signature is the species densities; the `n_p = Y_C * n_B` / `n_n = (1 - Y_C)
* n_B` lines moved out to the three callers (`zl/nmp.py:38`,
`mixed/adapters.py:973`, and two lines of `test/zl/test_zl_interaction.py`). The
§5 purity grep that motivated the ruling —

    grep -n "beta\|Y_C\|neutral\|trapped" eos/*/thermodynamics.py

— now returns **no non-docstring hit in `eos/zl/`**. What is left there is
`Y_C=n_C / n_B` on a returned record at :331, which is a computed OUTPUT and is
what `alphabag` and `vmit` already carry.

Gated as the ticket required and then some: `test/baseline/` for `zl` and
`mixed` both PASS in control and mine alike, and a direct hex-float comparison
of `thermo_from_n` over 4 densities x 4 Y_C x 3 temperatures, `nuclear_matter`
at three asymmetries and the full `compute_nmp` map is **bit-identical, 52/52
lines**. It has to be: the arithmetic did not move, it changed address.

### Item 4 — `TC_COEFF` becomes `Parameters.tc_coeff`

The constant left `thermodynamics.py` for `parameters.py`, where it is the
default of a new `tc_coeff` field and keeps its literal spelling
`0.57 * 2**(1.0/3.0)` and its citation. `T_critical`, `cfl_gap`, `cfl_dgap_dT`,
`cfl_P_correction`, `cfl_n_correction` and `cfl_s_correction` each take a
trailing `tc_coeff=TC_COEFF`, and the two places that hold a parameter set —
`cfl_thermo_from_mu` and `solver.py`'s `paired_density` — thread `params.tc_coeff`
through. So the override actually REACHES the gap, which was the point: a
sampler varying CFL pairing now moves T_c.

`Parameters.default().tc_coeff == 0.57 * 2**(1.0/3.0)` exactly, and the whole
CFL sector is **bit-identical, 33/33 lines** against the control across 3 gaps x
5 temperatures of `T_critical`, `cfl_gap`, `cfl_dgap_dT` and `cfl_thermo_from_mu`.
`dataclasses.replace(par, tc_coeff=0.70)` moves `cfl_thermo_from_mu(...).P` —
the override is live, not decorative. `test/baseline/alphabag` and
`test/baseline/abpr` both pass.

### Item 5 — the raise goes, and the COUNT goes with it

Re-measured against the current tree first, as instructed: ticket 61 landed
dd2's six §4 flags but did not touch this, and both raises were still there
(`sfho/solver.py:576`, `did/solver.py:213`). Both are deleted.

**Deleting the raise alone would have been wrong, and the ticket's own ruling is
what says so.** The refusal's stated reason — a three-flavour mu = 0 gas on top
of a tracked nu_e double-counts it — is correct physics; both models multiplied
by a flat `N_NEUTRINO_FLAVOURS = 3.0`. The ticket rules that "the three that
succeed are right", and what those three do is
`3 - (1 if the mode holds Y_Le)`: `enjl/solver.py:236` computes it in a named
function, `njl:275` and `ccdm:307` inline it as `2.0 if trapped else 3.0`. So
adopting their answer means adopting their count. Both models now do, in one
line each. This is not guessing at the physics — it is the count three
conformant models already ship, and the ticket's instruction to stop and report
applies to wiring that turns out to be more than this, which it did not.

`sfho.md`, `sfho.tex` are updated. Both already carried a paragraph saying the
refusal "is a DEFECT, not a design choice, and it is due to be removed",
naming this exact resolution — so the documents were written against the answer
and only had to catch up to it. Nothing in `docs/DEFERRED.md` changed: sfho.md
said in as many words that this was not a ledgered gap, and a grep confirms
there is no entry for it.

Two new test files, per §12 (`test/` is gitignored, so they stay local):
`test/sfho/test_sfho_thermal_neutrinos.py` and
`test/did/test_did_thermal_neutrinos.py`, 3 tests each. They assert the
combination converges and that switching the flag on moves eps, P and s by
exactly 3x the single-flavour gas in `beta_eq_neutrinoless` and exactly 2x in
`beta_eq_neutrino_trapped` — which is the invariant, since the gas carries no
conserved charge and enters no row of the residual. **The basenames are
model-prefixed deliberately**: `test/` has no `__init__.py`, so two files named
`test_thermal_neutrinos.py` collide at collection with an import-file-mismatch
error, which is how they were first written and how it was found.

### Two things found and NOT fixed, per the map's hard rule

- **njl and ccdm have item 1's defect too.** `njl/api.py:68` and
  `ccdm/api.py:76` carve `leptons` out of `**conditions` in `eos_point` and
  `eos_response` exactly as sfho/dd2/did did. The audit missed them because
  `njl/api.py:122` — the line finding 16b cites as conformant — is `eos_table`,
  and `eos_table` in both models does take the named argument. Now
  [ticket 68](68-njl-ccdm-leptons-condition.md).
- **`leptons=True` on a beta mode still gets three different answers.** sfho and
  dd2 raise (their pre-existing guards, preserved verbatim rather than
  redesigned), zl and did silently ignore, njl and ccdm forward it. Converging
  them is a behaviour change item 1 does not authorise and the reading is not
  obvious — in a beta mode the leptons are constitutive, so `leptons=True` is
  redundant rather than unimplemented, which is not quite the case §4's
  no-silent-no-op rule is written for. Recorded on ticket 68 as the half that
  needs a decision, not a diff.
