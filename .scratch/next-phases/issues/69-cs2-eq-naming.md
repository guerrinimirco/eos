# `cs2_eq` names the freeze where §5 requires the thermal variable

Type: task
Status: open
Parent: ../map.md

## Question

Four models return the equilibrium sound speed under the key **`cs2_eq`**:
`zl` and `dd2` ([ticket 13](13-hadronic-figures.md)), `vmit` and `alphabag`
([ticket 15](15-quark-notebook.md) item 2). The other six — `sfho`, `did`,
`njl`, `ccdm`, `abpr` and `mixed`'s hadronic side — return
`cs2_isothermal`, and `did`/`njl`/`ccdm` return `cs2_adiabatic` beside it.

`cs2_eq` names the *composition* axis of §5's conditioning (nothing held, the
composition re-equilibrates) and leaves the *thermal* axis unsaid, although the
derivative is taken at fixed `T`. §5 requires the returned name say which
thermal condition it was taken at, "never a bare `cs2` whose meaning depends on
the arguments" — and at `T > 0` the isothermal and adiabatic speeds differ by
`C_P/C_V`, so `cs2_eq` is exactly that bare name with a freeze word in front of
it. At `T = 0` the two coincide, which is why every notebook that reads the key
has been able to plot it and label it correctly anyway.

The freeze is not lost by the rename: it is the `frozen=` argument the caller
passed, and §5's three axes are conditioning, not return-key components.

So: rename the key to `cs2_isothermal` in the four, keep `cs2_adiabatic`
wherever the model can compute it (and say in its docstring where it cannot),
and sweep the callers — each model's `verify/`, `responses.py` docstrings, the
`.tex`/`.md` documents, `test/`, and the four notebooks, which currently all
carry a two-key reader for exactly this reason.

Cross-check against `eos/mixed` and `eos/astro/gmode` before renaming:
[ticket 49](49-nonconvergence-return.md) records `mixed.eos_response` returning
`cs2_eq = nan` outside the mixed window, and
[ticket 53](53-gmode-contract.md) asks whether `cs2_equilibrium` /
`cs2_frozen` is the g-mode surface — a third spelling, and this ticket should
not settle its own four while leaving that one open.

Done when a grep for `cs2_eq` over `eos/` returns nothing outside a changelog,
every `verify/` suite still passes, and `test/baseline/` is unmoved (a key
rename moves no number).
