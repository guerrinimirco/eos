# Three routes to a parameter set, in every model

Type: task
Status: resolved
Blocked by: 46
Parent: ../map.md

## Question

Stated by the user while ruling [ticket 46](46-api-changes.md): *"for all the
models I want the possibility of use some standard parametrizations or to use a
new set of parameters, for e.g. hadronic ones I want also the possibilities of
fix them using nmp."*

Three routes, and §6's "MODEL PARAMETERS ARE ARGUMENTS" is why they matter — an
inference run varies couplings across millions of calls, and a route that only
exists in one model is a route a sampler cannot take uniformly.

**Measured across all ten** with `dataclasses.fields` and `hasattr`:

| model | `default()` | `named()` | new set direct | `compute_nmp` | inverse |
|---|---|---|---|---|---|
| `dd2`      | yes | yes | **partial** | yes | `invert_nmp`, `from_nmp` |
| `sfho`     | yes | yes | yes | yes | `invert_nmp`, `from_nmp` |
| `did`      | yes | yes | **partial** | yes | **NONE** |
| `zl`       | yes | **NO**  | yes | yes | raises by design (ticket 26) |
| `vmit`     | yes | **NO**  | yes | — | — |
| `alphabag` | yes | **NO**  | yes | — | — |
| `abpr`     | yes | **NO**  | yes | — | — |
| `njl`      | yes | yes | yes | — | — |
| `ccdm`     | yes | yes | yes | — | — |
| `enjl`     | yes | yes | yes | — | — |

Four gaps:

1. **`zl`, `vmit`, `alphabag`, `abpr` have no `named()`.** §13 lists
   `Parameters.named(name)` in the mandatory vocabulary. Each ships exactly one
   published set today, so `named()` is either a one-entry map or the docstring
   says in as many words that there is one set and `default()` is it. Pick one
   and apply it to all four — two models answering the same question two ways is
   what §13 exists to prevent.
2. **`did` is hadronic and has no inverse map.** The user's requirement names
   hadronic models specifically. `docs/DEFERRED.md:947` records it as not
   implemented and not published. Decide whether it is written (and against what
   closure) or whether the refusal becomes explicit the way
   [ticket 26](26-zl-nmp.md) made `zl`'s: raising with the reason, never absent.
   **An absent name and a refusing name are not the same thing** — the first is
   an `AttributeError` a caller cannot interpret.
3. **`dd2` and `did` cannot be constructed field-by-field** without supplying
   required fields. For a DD-RMF with many couplings that may be right, but then
   the documented route to "a new set" is `dataclasses.replace(default(), ...)`
   and the docstring must say so.
4. **`vmit` loses `get_vmit_custom`** to ticket 46 item 1. That deletion is
   correct — `Parameters` carries identical defaults, so the helper was a pure
   alias — but it removes the *documented* route to a new vmit set, and this
   ticket owes the replacement sentence.

Not a rename ticket: [ticket 07](07-naming-sweep.md)'s sweep covered the names.
This is about which routes EXIST.

Done when all three routes are available in every model or refused with a reason
in the ones where a route is meaningless, and each model's document (§11) states
which routes it offers.

## Answer

**All three routes exist in every model, and where one is refused the refusal
has a name.** Commit `ff7888e` on `main`, 29 files.

**Gap 1 — `named()`, and the ruling that applies to all four.** `zl`, `vmit`,
`alphabag` and `abpr` get a **one-entry map**, not a docstring note. The shape
is the one `dd2` and `did` already use — a local dict, a `KeyError` naming what
is available — so nothing new was invented for it. The docstring-only option
was rejected on gap 2's own argument, which applies just as well here: an
absent attribute is an `AttributeError` a caller cannot interpret, and six of
the ten models already answer `named()`, so leaving four silent is precisely
the "two models answering the same question two ways" §13 exists to stop.

One detail the ticket did not specify: **what the key is.** It is the set's own
`name` field, so `Parameters.named(par.name)` round-trips — `'ZL_Constantinou'`,
`'vMIT_default'`, `'alphabag_default'`, `'abpr_default'`. `dd2` and `did` key
on the model name instead (`'DD2'`, `'DIDY'`) because their `Parameters` carry
no `name` field; that is a difference in the dataclasses, not a second answer to
this question, and `test_named_round_trips_on_the_sets_own_name` skips the five
models without the field rather than pretending otherwise.

**Gap 2 — `did`'s inverse is REFUSED, not written.** `nmp.invert_nmp` and
`nmp.from_nmp` now exist in `eos/did/nmp.py`, are exported from `eos.did`, and
raise `NotImplementedError`. Ticket 26's `zl` pattern exactly: it says which,
it names the two ways to close it, and it is never a silent no-op.

Writing it was considered and is not close. `docs/DEFERRED.md` already had the
first reason — the couplings are the maximum-likelihood point of a Bayesian
analysis over 18 observables, so an inversion has to choose which to impose,
and the paper makes no such choice. **The session found a second the ticket did
not anticipate:** DID carries two inequivalent symmetry energies, the quadratic
coefficient `S_2` and the full ISM-to-neutron-matter difference `S`, which
differ by 2.72 MeV at saturation (its own `nmp.py` computes both, and the
paper's Table VI publishes both). So even the LIST of data to impose is
undetermined — `{n_0, B, K, S, L}` and `{n_0, B, K, S_2, L_2}` are different
inversions with different answers. `zl`'s refusal is a one-parameter family;
`did`'s is a family whose defining equations are not agreed on either. Both are
now recorded in `docs/DEFERRED.md`, which also states that the refusal is
NAMED rather than absent.

**Gap 3 — `dd2` and `did` are right as they are, and now say so.** 18 of dd2's
27 fields and 29 of did's 34 carry no default, so field-by-field construction
means supplying all of them. That is correct for a DD-RMF, where a coupling is
meaningless without the four shape coefficients that go with it and a partially
specified set is a silently wrong one rather than a convenient one. Both
`default()` docstrings now name `dataclasses.replace(default(), ...)` as the
route, alongside the constructors that do exist — `from_microscopic` (which
derives the omitted `a_i`, `d_i` from the internal constraints and validates
any supplied) and `with_deltas`.

**Gap 4 — the vmit sentence is paid.** `Parameters.default()`'s docstring now
carries it: every field has a default so `Parameters(B4=..., a=...)` names only
what changes, and because the dataclass is frozen, `dataclasses.replace` is how
a set already in hand is modified — there is no setter and no mutating helper.

**Documents (§11).** A uniform **"Three routes to a parameter set"** block in
all ten `.md` and all ten `.tex`, at the end of each model's parameter section.
Each names its three routes concretely: which keys `named()` answers to, what
builds a new set, and either the inverse map or why there is none. The quark
models get "no route, and none is missing — no nuclear sector, so no `nmp.py`
and nothing to invert", which is an absence rather than a gap; `did` and `zl`
get the refusal in full. All ten `.tex` compile under
`pdflatex -halt-on-error`. `sfho`'s block was repositioned after its existing
published-sets table rather than duplicating the five keys above it, and
`abpr`'s first draft was corrected: its "inverse maps" section inverts STATE
VARIABLES (`mu` from `n_B`, `mu` from `P`), not parameters.

## Measurement

**Interpreter: python.org 3.14.2** (numpy 2.3.5, scipy 1.17.0).

Ticket 65 held `eos/*/species.py` and the baselines, and ticket 29 was live in
`api.py` / `table.py` / `solver.py` across most models plus
`eos/general/modes.py`, so a run in the repo would have measured three
sessions. Gate was an **isolated `git archive HEAD` PAIR**: a control copy at
HEAD beside a work copy carrying only this ticket's eight `.py` files, both
with one snapshot of the gitignored `test/`. `diff -rq` confirmed the two
copies differed by exactly those eight files.

    test/baseline test/zl test/vmit test/alphabag test/abpr test/did
    test/dd2 test/test_imports.py

    HEAD control  892 collected   2 failed, 890 passed   172.52s
    with 79       892 collected   2 failed, 890 passed   172.41s

The two outputs are **identical byte for byte** apart from the copy path in
three warning lines and the elapsed time. **0 added failures**, and the
`test/baseline` gate at rtol = 1e-10 holds: adding a route moves no number.

**The two failures are pre-existing and neither was touched.**
`test_baseline[enjl]` is red on purpose (ticket 72; `enjl.npz` is still the 3.9
file). `test_baseline[ccdm]` is NEW relative to this map's Suite-status block
and is **ticket 65's, not this ticket's**: all twelve non-enjl `.npz` were
regenerated at 16:47 today, and the failure is 20 `state.field_residual` keys —
solver residuals at 1e-6 to 1e-10 ABSOLUTE, compared at rtol = 1e-10 against a
stored value that is zero by construction, giving "max rel. change 7.000e+00"
on an absolute difference of 1.7e-06. That is exactly the shape
[ticket 76](76-nucleation-golden-tolerances.md) named as a comparison to
CORRECT rather than loosen, and it belongs to whoever closes 65.

**The new gate.** `test/test_parameter_routes.py`, cross-model at the `test/`
root beside `test_imports.py`, since the invariant spans all ten. It asserts
`default()`, `named()` with a recoverable `KeyError`, the `name`-field round
trip, `dataclasses.replace` as the universal new-set route, and — for the four
hadronic models — that the inverse map is present or refuses with a
`NotImplementedError` pointing at the forward map that does exist.

    with 79    45 passed, 5 skipped
    at HEAD     9 failed, 36 passed, 5 skipped

The 9 are the four missing `named()` (×2 tests each) and `did`'s absent
inverse — the whole of what the ticket measured, and nothing else. `test/` is
gitignored, so the file is local and not in the commit.

Status: resolved.
