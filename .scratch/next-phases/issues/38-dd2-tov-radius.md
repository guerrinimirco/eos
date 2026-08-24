# dd2's TOV pipeline gives R = 12.33 km against an asserted 13.2 ± 0.4

Type: task
Status: resolved
Parent: ../map.md

## Question

The fourth and last failure on `main` ([ticket 01](01-pytest-baseline.md)):

    test/dd2/test_dd2_m4_tov.py:35
    assert 12.332677607118407 == 13.2 ± 4.0e-01

A 0.87 km shortfall, more than twice the tolerance. `test_baseline[dd2]` passes,
so whatever moved is **not** in the baselined quantities — the EoS points
themselves reproduce at rtol = 1e-10. That narrows it to the TOV path: the
sweep's density range, the crust attachment, the interpolation onto the P grid,
or the radius extraction.

Confirmed independent of the photon fix ([ticket 28](28-photons-silent-ignore.md)):
the test calls `sweep_beta_eq_octet(..., include_photons=False)` explicitly at
line 55, and it runs at T = 0 where the photon gas vanishes anyway. Verified by
running the file alone both before and after that change.

§12 makes the DD2 published TOV values ground truth, so the asserted 13.2 km is
presumed right until shown otherwise. Two questions, in order:

1. Is 12.33 km what DD2 nucleonic matter actually gives through this pipeline,
   or is the pipeline losing something — a truncated density range, a missing
   crust, a non-monotone P segment silently dropped? CLAUDE.md §8 requires the
   P-monotonicity and `0 <= c_s^2 <= 1` check to run *before* integration and
   return a status rather than a meaningless mass; check that it does here.
2. If the number is real, is the assertion quoting a different configuration
   than the test builds — a different parameter set, crust, or M value?

Do not widen the tolerance to make it pass (§12).

## Answer

**Same cause as [ticket 37](37-did-failures.md) Finding 2: the BPS crust table is
absent and `test/dd2/dd2_tov_sequence.py` silently falls back to no crust.**

`have_crust("BPS")` is `False` here — the search path is only `<repo>/data/crust`,
which does not exist, and `EOS_CRUST_DIR` is unset. The file is on disk at
`/Users/mircoguerrini/Desktop/Research/Crust/BPST0.dat`, outside the repo.

With `EOS_CRUST_DIR` pointed at it, `test_dd2_m4_tov` passes. Both questions the
ticket posed are answered: the pipeline was losing the crust, not truncating a
density range or dropping a non-monotone segment, and the asserted 13.2 km was
right all along.

Neither the dd2 code nor `eos/astro/tov` needs any change. The remaining defect —
the silent downgrade in the test helpers — is
[ticket 39](39-crust-silent-fallback.md).

Status: resolved.
