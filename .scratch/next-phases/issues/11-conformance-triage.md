# Sort every failing conformance row into fix-code, fix-CLAUDE.md, or defer

Type: grilling
Status: open
Blocked by: 02, 08, 09
Parent: ../map.md

## Question

Ticket 08 is resolved: 24 Fail and 25 Ambiguous cells over 136, already sorted
into 22 (a) / 11 (b) / 12 (c) with file:line evidence in
[conformance-table.md](../research/conformance-table.md). This ticket is the
human ruling on that sort, not a re-derivation of it.

**One (a)-class finding is already carried out of this pile**: the
`photons=False` silent-ignore in dd2 and mixed, which changes numbers and so gets
its own gate at [ticket 28](28-photons-silent-ignore.md). The other 21 stay here.

**The auditor's own note, worth weighing before ruling:** `docs/DEFERRED.md` is
unusually thorough — most of what a naive audit would flag is already recorded
there with reasoning and measurements. That is why the (c) pile is only 12
entries, and why the real work is the (a) fixes, several of which are one-liners.

Take ticket 08's table and put every row that **fails or is ambiguous** to the
user, grouped as:

- **(a)** the code is wrong and should be fixed
- **(b)** `CLAUDE.md` describes a target the refactor settled differently, and the
  document should change
- **(c)** genuinely deferred, and belongs in `docs/DEFERRED.md`

Two rows are already known to be live and are decided elsewhere — carry their
rulings in rather than re-litigating: §11's "one usage notebook per model"
(ticket 02) and §11's mandated `.tex` (ticket 09).

[Ticket 07](07-naming-sweep.md) surfaced §5 layout rows the conformance table may
not repeat — carry them in as findings to be triaged, not re-derived:

- `eos/abpr` has no `table.py` though it carries `TableResult`, `cfl_row` and an
  `eos_table` with the full progress dictionary (`api.py:112,127,146`), and
  `response_at_mu` sits in `solver.py:350` rather than a `responses.py`.
- `eos/vmit/compute_tables.py` is kept by an existing `DEFERRED.md` ruling, but
  its three package-repeating symbol names are not covered by that ruling.
- `docs/DEFERRED.md:320` records the vmit §13 conversion as "DONE"; it is not.
  A stale ledger entry is itself a (c)-class row.

Nothing is edited here. This ticket produces the ruling; ticket 22 applies the
(b)-class edits as a `CLAUDE.md` diff, the (a)-class fixes become their own work,
and the (c)-class entries land in `docs/DEFERRED.md`.

Any (a)-class fix that could move a number must say which §12 golden reference it
is checked against.
