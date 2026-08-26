# Three routes to a parameter set, in every model

Type: task
Status: open
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
