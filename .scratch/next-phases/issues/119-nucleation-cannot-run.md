# `nucleation` cannot run here, and the suite that would say so cannot collect

Type: task
Status: closed
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

## Resolution (2026-08-29)

**Both defects closed. Defect 2's diagnosis in the Question above is WRONG in
its last line and the correction is the whole finding: it is not a property of
that checkout, it is a property of the COMMAND — `timeout` drags the
interpreter under Rosetta.**

### 0. The uncommitted edit: found, and its owner cleared it

The par-first / exception-narrowing / gluons-flag edit was written by the
`nucleation` session `d1826fe2` on **2026-08-28 at 14:43 UTC** (peer session
`nucleation-22`), and its transcript shows it **finished and verified** the
work in the next minute -- "all three converge 12/12 where they previously
returned 0/12", "72 tests pass" -- then moved on to the notebook. So the edit
was **done-but-uncommitted, not mid-flight**, and ticket 101 stopping short of
it was still the right call: nothing in the working tree says which of those
two it is.

Asked before touching it. The owner replied: *"Go ahead -- no conflict. I'm
done in filters.py and won't touch it."* It then re-ran the finished call sites
itself and confirmed the migration is numerically inert.

An aside worth keeping: the mtime of `filters.py` is **not** evidence of who
wrote that edit. It reads 11:20:30 local, which is 09:20:30 UTC -- ticket 101's
own `perl -pi` rename, two hours *after* the edit being hunted. The transcripts
are timestamped UTC and the tree is local (+2); comparing them directly points
at the wrong session.

### 1. Three call sites, three `SpeciesFlags`, one `leptons=`

Built on top of the owner's edit, preserving its argument order, its narrowed
`except (RuntimeError, ValueError)` and its CFL physics call:

    :53   solve_cfl(p, nB, cfg.T_eos, Delta0,
                    SpeciesFlags(photons=False, gluons=False), ...)
    :76   solve_beta_eq_neutrinoless(p, nB, cfg.T_eos,
                    SpeciesFlags(photons=False, gluons=True), ...)
    :110  solve_fixed_yc_ys(p, nB, yc, 0.0, cfg.T_eos,
                    SpeciesFlags(photons=False, gluons=True),
                    leptons=True, ...)

The `gluons` asymmetry the Question flags is **real physics and is kept**, not
smoothed over. The CFL path must say `False` -- `alphabag.SpeciesFlags.gluons`
RAISES in `cfl`, because locking leaves a single unbroken U(1)_Qtilde and all
eight gluons are Meissner-massive. The two unpaired paths keep the published
`True`: there the gluons are massless and the sector is a genuine choice.

**The asymmetry is free only at `T_eos = 0`, and that is load-bearing.**
`FilterConfig.T_eos` defaults to `0.0`, where both gases contribute exactly
zero, so the CFL and unpaired branches are still compared on equal terms --
which matters, because the Witten and no-rehadronization filters compare them
directly. At finite `T_eos` that stops being true and the comparison is no
longer like-for-like. Left as a comment at the call site rather than an
`assert`, which would break a legitimate finite-T run; recorded here as the
known ceiling.

`include_electrons` became `leptons=True`, a separate named argument, never a
species flag (CLAUDE.md §5). It has to be on: the root `ud_eps_per_nB`
brackets is `mu_d - mu_u - mu_e`.

**Scope held to exactly the three direct `eos` solver calls.** `nucleation`
still has ~20 `.P_total`/`.e_total` reads and ~40 `include_photons=`/
`include_gluons=` kwargs, and every one of them is nucleation's OWN API -- its
`DropletThermo` dataclass and its own solver wrappers in `nucleation/quark.py`.
Propagating either rename into them would break the package.

### 2. `timeout` is an x86_64 binary, so numpy is the wrong architecture

    $ file $(which timeout)
    /usr/local/bin/timeout: Mach-O 64-bit executable x86_64

On Apple Silicon that starts under Rosetta, and the child interpreter inherits
the x86_64 preferred architecture:

    $ timeout 60 python3 -c "import platform; print(platform.machine())"
    x86_64
    $ python3 -c "import platform; print(platform.machine())"
    arm64

numpy's own C extension is then the wrong slice, and `numpy/__init__.py:117`
masks it with `raise ImportError(msg) from e`. Underneath:

    ROOT: ImportError : dlopen(.../numpy/_core/_multiarray_umath
      .cpython-314-darwin.so, 0x0002): tried: ... (mach-o file, but is an
      incompatible architecture (have 'arm64', need 'x86_64'))

This explains every symptom exactly, including the ones that made it look like
a checkout property: `import numpy` succeeded outside pytest because those
probes were run **without** the `timeout` wrapper; the HEAD-`eos` control
reproduced it because that control was run **with** it; and `0.24s` is a
process that died before collecting anything. There is no `conftest.py`
problem, no stray `numpy` directory, and no rootdir shadowing -- the searches
for those were sound and correctly found nothing.

Measured both ways, same shell, same command otherwise:

    PYTHONPATH=<eos> timeout 300 python3 -m pytest test -q  -> 9 errors in 0.23s
    PYTHONPATH=<eos>            python3 -m pytest test -q  -> 72 passed in 2.61s

### Gate

- **The three call sites run against today's `eos` and converge.** On the
  production default grid (`n_B_grid` = 250 points, `T_eos = 0.0`):
  `cfl_eos_at_params` **250/250**, `unpaired_eos_at_params` **250/250**,
  `ud_eps_per_nB` = **959.56 MeV**, above the 930 MeV line the 2-flavour
  filter requires. On a 12-point grid, 12/12 and 12/12, matching the owner's
  independent run.

  **`ud_eps_per_nB` moves with the grid, and the gate number is the converged
  one.** It is a P = 0 crossing interpolated between grid nodes, so it depends
  on both the resolution and the SPAN. Measured twice, independently, same
  (alpha, B4) = (0.16, 145):

      linspace(0.05, 2.5,   12)  ->  985.07 MeV
      linspace(0.05, 2.5,  250)  ->  959.56 MeV     <- the gate, production default
      linspace(0.05, 2.5, 2000)  ->  959.51 MeV
      linspace(0.2,  1.2,   12)  ->  965.12 MeV     <- a different span, not just
                                                       a coarser grid

  So the production default is converged to ~0.05 MeV, and **all four sit above
  930: the Witten conclusion never depended on the resolution.**
- **`pytest test` in `../nucleation` collects**, and passes. As measured,
  with the interpreter named:

      CPython 3.14.2 (python.org), pytest 9.1.1, numpy 2.3.5, scipy 1.17.0
          -> 72 passed in 2.61s

  Also collects and passes on the other two stacks -- `/usr/local/bin/python3`
  (the same 3.14.2) 72 passed in 2.49s, and anaconda 3.9.7 72 passed, 1
  warning in 5.60s (a pandas `np.find_common_type` DeprecationWarning, not
  ours). `/usr/bin/python3` (Xcode 3.9.6) has no pytest and cannot run it.

  **Those two runs are corroboration for the COLLECTION claim only, and the
  anaconda one is not corroboration of correctness.** They establish that the
  `9 errors` were the wrapper and not the checkout, since a second and third
  stack collect the same tree fine. They do not establish that the suite is
  sound, because anaconda 3.9.7 carries scipy 1.13.1 and numpy 1.26.4, **both
  below the floors `nucleation` and `eos` declare** (`scipy>=1.17`,
  `numpy>=2.0`). That floor is a measured correctness bound, not hygiene:
  `eos`'s own `pyproject.toml` records that scipy 1.13's
  `root(..., method="hybr")` REPORTS SUCCESS while returning its seed
  unchanged on one of eos's closures (isoscalar residual 2.2e-03 against
  6.7e-11 on 1.17). Green on anaconda is therefore consistent with the suite
  simply not covering that closure. **The python.org 3.14.2 run is the
  measurement; if the two ever disagree, 3.14.2 is the one to trust.**
- Ticket 76's two goldens are not in this: nothing here is a golden, and the
  72 include them passing.

**Not committed.** The `filters.py` change sits in `nucleation`'s working tree
on `paper-release`, alongside the owner's notebook edits and a large `output/`
regeneration, none of which the user has committed. Durability is the user's
call, not this ticket's.

### For anyone measuring anything on this machine

**Never wrap a measurement in `timeout` here.** It is not a no-op: it silently
switches the interpreter's architecture, and what it breaks (numpy) reports an
error that names a cause it does not have. `test/run_clean_suite.sh` and any
future timing harness must call the interpreter directly.
