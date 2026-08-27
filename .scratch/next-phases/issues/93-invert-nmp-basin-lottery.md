# `invert_nmp`'s verdict is decided in the target's last bits

Type: task
Status: open
Blocked by: -
Parent: ../map.md

## Question

Split out of [ticket 67](67-dd2-t0-adoption.md), which found this while
measuring something else and could not fix it from where it stood.
[Ticket 47](47-dd2-nmp-inversion.md) found the same floor from the other side:
47 is the STENCIL noise in the forward map, this is the SOLVER's basin
selection in the inverse map, and the fix candidates are different.

### What was measured

The 5x5 default closure at DD2's OWN nuclear-matter parameters, target scaled
by (1 + eps), no code change, python.org 3.14.2 / numpy 2.3.5 / scipy 1.17.0:

    eps      isoscalar residual        couplings
    0        6.7e-11                   moved 3.9%
    1e-14    2.2007e-3                 moved 0.0 -- the seed, bit for bit
    1e-13    2.2007e-3                 moved 0.0
    1e-12    2.3e-08                   moved 3.9%
    1e-10    2.0e-3 (partial)          moved 0.27%

Three findings, none of which is about any one refactor:

1. **`root(method="hybr")` returns the published seed unmoved** on a target
   perturbed in its fourteenth digit, and `InversionStatus.ok` is True when it
   does, because ISO_GATE = 2e-2 admits the seed's own 2.2007e-3 cross-row
   violation. Not new for the 6x6 closure: `nmp.py:70-78` documents the failure
   mode, `test_api.py` documents hybr stalling with status 5 at DD2's own NMPs,
   and that test already carried the seed comparison
   (`assert inv.b_sigma != par.b_sigma`) this ticket would otherwise propose.
   **What is new is that it reaches the 5x5 closure**, which the same docstring
   describes as converging to 6.7e-11 without qualification -- it does, at
   eps = 0, and returns the seed at eps = 1e-14.

2. **`test_inversion_is_deterministic` gives false comfort.** Two stuck runs
   agree perfectly, so the test passes hardest exactly when the solver has done
   nothing.

3. **The converged branch is not injective at DD2's own point.** Where the
   solve does move, it lands on couplings 3.9% from the published set that
   reproduce the same six NMPs. "Converged" and "recovered DD2" are different
   statements and the return value does not separate them.

And across the (K_sat, Q_sat) plane, the 6x6 closure's 32-restart landing
residual is chaotic in the target's last bits: eight targets over
K_sat in {210, 220, 230, 240} and Q_sat in {300, 350}, each at
eps in {0, 1e-13, 1e-11}, and NOT ONE holds its verdict across its own three
-- residuals scattering 3e-5 to 0.15 with no relation to eps.

### What to decide

1. **Does `ok` learn to say "never moved"?** The cheap version is the
   comparison `nmp.py:70-78` already tells the reader to make by hand:
   `InversionStatus` carries whether the recovered couplings differ from the
   seed, and `ok` is False when they do not. That converts a silent wrong
   answer into a reported failure, which is what section 6 requires of a public
   boundary.
2. **Is ISO_GATE = 2e-2 defensible** when the seed's own cross-row violation is
   2.2e-3, an eighth of it? A gate that admits the starting point cannot
   certify an answer.
3. **Restart coverage.** `test_dd2_m8` gave up asserting on 32 restarts at a
   single target (ticket 67, decision 3) because that verdict is a coin flip.
   The property worth testing is the one `nmp.py:84-96` already records --
   7/68/115 of 187 cells inverting at 0/32/64 restarts -- a count over a grid,
   stable where any single cell is not. Slow; decide whether it is a test or a
   `verify/` entry.
4. **Does the same lottery reach `from_nmp` and the scan callers**, or is it
   confined to targets sitting on their own seed? Partly answered already: it
   reached `test/tov/test_solver_fast_robustness.py`, whose three cases needed
   a soft Delta-rich parametrization and obtained it through
   `build_parametrization` on a hand-tuned NMP target. Ticket 74 had already
   re-tuned that sample once for this reason; ticket 67 froze the resulting
   parametrization as data instead, which removes the prerequisite but does
   not fix the map.

5. **Dropping Q_sat is NOT the workaround it looks like.** The 5x5 keeps the
   system square by PINNING c_omega, so excluding Q_sat constrains the
   isoscalar sector more than imposing it does -- the 6x6 solves for c_omega as
   a sixth unknown. Measured at (K_sat = 220, L_sym = 30) with DD2's other
   NMPs: the 6x6 converges (Q_sat 299.65) where the 5x5 fails at 5.9e-2. Any
   proposal to retire the 6x6 has to answer what reaches those targets
   instead.
