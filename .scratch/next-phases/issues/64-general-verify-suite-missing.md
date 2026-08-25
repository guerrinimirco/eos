# eos/general/ has no verify/, which CLAUDE.md section 5 states it has

Type: task
Status: open
Blocked by: 11
Parent: ../map.md

## Question

Found while running [ticket 49](49-nonconvergence-return.md)'s verify sweep,
and pre-existing — this is not a regression from that ticket.

CLAUDE.md section 5 says it in the present tense, as a paragraph of its own:

> **`general/` carries a `verify/` too.** It is not a model either, but it is
> the single home of the Fermi and Bose integrals (section 7), the
> conserved-charge basis maps (section 2) and the thermal meson gas — the
> pieces every model's correctness rests on, and the ones a wrong result is
> hardest to trace back to. Its suite checks those shared pieces against each
> other: JEL against the alternatives section 7 requires be validated against
> it, the basis maps against the species tables, the T = 0 limits against the
> finite-T forms as T -> 0.

It does not exist. `ls eos/general/` shows no `verify`, and

    python3 -m eos.general.verify.run_full_check
    ModuleNotFoundError: No module named 'eos.general.verify'

Nine of the ten `verify/` entry points a full sweep should hit are real and
pass; this is the missing tenth, and it guards the code with the widest blast
radius in the repository.

**Not covered by [ticket 51](51-verify-invariants.md)**, whose four missing
invariants are all inside models — checked, `general` does not appear in it.

The section text already names the three checks the suite owes, so the scope is
written: JEL against the alternatives validated against it, the basis maps
against the species quantum-number tables, and the T = 0 limits of the
finite-T forms as T -> 0. `test/general` (128 passing cases) may already cover
some of this as unit behaviour — the first job is to sort what is there into
"physics invariant, belongs in `verify/`" and "unit behaviour, stays a test",
per section 12, rather than writing three checks that duplicate tests.

Note the second half of section 8's gate does NOT apply: `general/` has no
`table.py` and hands no table to a structure solver, so it does not owe the
monotone-P / causal-c_s^2 delivery check.
