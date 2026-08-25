# The public-signature corrections §5 and §3 require

Type: task
Status: open
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
