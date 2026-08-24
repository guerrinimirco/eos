# Approve or reject the proposed public renames

Type: grilling
Status: open
Blocked by: 07
Parent: ../map.md

## Question

Ticket 07 produces a list of §13 deviations and proposes renames. This ticket is
the approval gate: **no public name is touched until the user has ruled on the
list.**

Take them one at a time where they differ in kind: a private helper renamed is
cheap; a public entry point renamed breaks `nucleation` call sites (Phase 6) and
any notebook already written against it.

Named suspects the list will contain: `eos/vmit/compute_tables.py` (not a §5
layout name), and whatever `Parameters` subclasses still repeat their package
(§13's example is `eos.vmit.VMITParams` saying "vmit" twice).

Resolved when every proposed rename is approved, rejected or deferred, and the
approved ones are applied with the test suite green and the added-failure count
against `output/_audit/pytest_before.txt` reported.
