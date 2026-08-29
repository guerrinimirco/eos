# `nucleation` cannot run here, and the suite that would say so cannot collect

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

Nothing to decide about naming; two defects found by
[ticket 101](101-pressure-and-energy-field-names.md)'s consumer sweep, neither
caused by it, and the second is what hid the first.

### 1. The one `eos`-point consumer calls a signature `eos` removed

`nucleation/analysis/filters.py` is the ONLY file in `nucleation` that reads a
solved `eos` point (verified by sweeping every import, function-local ones
included; everything else takes `eos.general.*`, sector blocks, table rows, or
`EOSTable_for_TOV`). Its three solver calls pass keyword arguments that no
longer exist:

    filters.py:53   solve_cfl(p, nB, T, Delta0, include_photons=False,
                              include_gluons=False, initial_guess=guess)
    filters.py:76   solve_beta_eq_neutrinoless(p, nB, T, include_photons=...,
                                               include_gluons=True, ...)
    filters.py:110  solve_fixed_yc_ys(p, nB, yc, 0.0, T, include_photons=...,
                                      include_gluons=True,
                                      include_electrons=True, ...)

against, today:

    solve_cfl(par, n_B, T, Delta0, flags, initial_guess=None)
    solve_beta_eq_neutrinoless(par, n_B, T, flags, initial_guess=None)
    solve_fixed_yc_ys(par, n_B, Y_C, Y_S, T, flags, leptons=False,
                      initial_guess=None)

`cfl_eos_at_params(0.16, 145.0, 80.0, cfg)` raises `TypeError` on the first
call. The sectors are now a `SpeciesFlags` object (tickets 91/95/96), so the
fix is to build one per call site -- and it is a real decision at
`filters.py:76` and `:110`, which asked for `include_gluons=True` where the
`cfl` path asks for `False`.

**The file was already dirty and mid-edit** when ticket 101 opened it: an
uncommitted par-first / exception-narrowing / gluons-flag change was in
`nucleation`'s working tree, by someone else. 101 added exactly three renamed
field reads and left the rest alone. Whoever owns that edit owns this.

### 2. The suite that would have caught it cannot collect

    PYTHONPATH=<eos> python3 -m pytest test -q      # in ../nucleation
    9 errors in 0.24s -- every test module

    E ImportError: Error importing numpy: you should not try to import numpy
    E   from its source directory; ...

`import numpy` succeeds in the same interpreter outside pytest (3.14.2, numpy
2.3.5 from the python.org site-packages), succeeds with `<eos>` on PYTHONPATH,
and succeeds with `test/` prepended to `sys.path`. It fails only under pytest,
and numpy's message masks the real cause with `raise ImportError(msg) from e`.
**Identical under a HEAD-`eos` control**, so it is a property of that checkout,
not of anything `eos` changed.

This is the ticket-24 shape exactly -- a green `eos` suite over a `nucleation`
that does not import -- except that here the caller is broken while
`nucleation`'s own modules all still import (25/25), so nothing announces it.

## Gate

- The three call sites run against today's `eos` and return converged points.
- `pytest test` in `../nucleation` collects, and its result is reported with
  the interpreter named -- or the reason it cannot is written down here.
- Whatever the suite reports is stated as measured; ticket 76's two goldens
  comparing round-off are known and are not this.
