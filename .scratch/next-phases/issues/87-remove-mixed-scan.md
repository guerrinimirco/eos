# Remove `eos/mixed/scan.py`

Type: task
Status: open
Blocked by: 84
Parent: ../map.md

## Question

The user's ruling in [ticket 84](84-vmit-params-in-the-plumbing.md): *"scan is
just a code that help us find parametrizations useful. We can remove it,
rethink it. In future I will have a bayesian code which do it better so probably
we can just remove it."*

626 lines whose own first line reads "Where in parameter space does a DD2 + vMIT
hybrid equation of state exist?" — a declared DD2+vMIT study inside a composite
engine that is meant to be general, and the reason `mixed` imports `dd2.nmp` and
`dd2.solver` at module level.

§6 lists Bayesian inference as use case 3, so a scan is aligned with the
library's purpose. This deletes THIS scan, not the ambition.

### One function must survive, and it relocates

**`build_parametrization(nmp, flags, ...)` moves to `eos/dd2/nmp.py`.** Not a
concession to the deletion — a correction owed anyway: §5 says an NMP-inverting
constructor "is therefore a free function in `nmp.py`, not a classmethod on the
parameter dataclass". It has been living in the composite engine, one layer
further from home than the anti-pattern §5 names.

**Four sites depend on it**, in `test/tov/test_solver_fast_robustness.py:87,94,
188,196` — among the tov tests [ticket 74](74-py314-non-baseline-failures.md)
JUST repaired. Do not break them.

### The rest of the tail, measured

    eos/mixed/__init__.py:124-126,164-165   six re-exported names
    test/mixed/test_scan.py                 271 lines, goes with it
    test/tov/test_rotating.py:373           a `dd2_scan` fixture
    CLAUDE.md:36-37                         names `mixed/scan.py` in the §1
                                            astro carve-out  -> ticket 85
    eos/vmit/parameters.py:66               cites `eos.mixed.scan`
    eos/vmit/vmit.tex:160, vmit.md:102      cite it in the DOCUMENTS (§11)
    eos/mixed/charges.py:109                a comment

The three `vmit` citations are §11 documents describing what the code does. They
do not simply lose a sentence: each says `eos.mixed.scan` is what moves over
(B4, a, m_s), so each needs a replacement statement of how a vMIT parameter
sweep is actually done now.

### The §8 gate is NOT lost

`scan.py`'s `eos_is_physical` is a SECOND implementation used only by the scan.
`eos/mixed/verify/run_full_check.py:231 _check_causality` implements §8's
delivery gate independently and stays.

### Gate

`grep -rn "scan_parameters\|scan_point\|scan_hadronic\|eos.mixed.scan" eos test
docs` returns nothing outside this ticket's own deletions; `test/tov` still
green including the four `build_parametrization` sites; `test/baseline/`
unmoved. CLAUDE.md §1's mention goes via
[ticket 85](85-claudemd-sentences-owed.md), not here.
