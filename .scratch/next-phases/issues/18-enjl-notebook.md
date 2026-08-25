# notebooks/enjl — skeleton, knobs, figures and the author-table reproduction

Type: task
Status: open
Blocked by: 04, 61
Parent: ../map.md

## Question

Stage 3. Same shape again, for `enjl`, **including its branch pair** — §5 records
the ENJL branch pair as two branches of one functional, and the notebook must
show both.

`docs/enjl/` and `test/enjl/reference/` hold the author tables. **Show the
notebook reproducing at least one of them, with the residual printed.** Those
tables are §12 golden references: code that disagrees with them is wrong.

Finite-`T` ENJL is an open item. Check `docs/DEFERRED.md` and **let the notebook
report the gap rather than work around it** — the §3 raise is caught at the top
of the section, its message printed, and the notebook continues.

Figures to `output/enjl/`. Done when the notebook executes clean, every figure
file exists, and the reproduced author table's residual is printed.

## Added by ticket 05

**The ENJL branch pair belongs to THIS notebook, not `hybrid_eos`.**
`enjl_branch_pair` lives in `eos/mixed/adapters.py` and §5 lists it among the
shipped adapters, so this notebook and [ticket 58](58-hybrid-skeleton.md) overlap
on exactly that object. The two branches are two branches of one functional, not
two models being coupled — the physics is ENJL's and `eos/mixed` is the machinery
it is expressed through. State the boundary in one line here; ticket 58 states it
there.
