# Apply the approved renames — eos/sfho

Type: task
Status: open
Blocked by: 10, 42
Parent: ../map.md

## Question

~60 call sites. sfho has no `Parameters.default()` / `.named()` pair at all —
it carries five `get_sfho*` free functions instead, which is the same job under
five names none of which is §13's.

**Rule 2 (7):**

    add Parameters.default() and Parameters.named(name)     (40 eos / 15 test)
    get_sfho_nucleonic       -> Parameters.named("nucleonic")
    get_sfhoy_fortin         -> Parameters.named(...)
    get_sfhoy_star_fortin    -> Parameters.named(...)
    get_sfho_2fam_phi        -> Parameters.named(...)
    get_sfho_2fam            -> Parameters.named(...)
    get_all_parametrizations() -> PUBLISHED_SETS

Settle the exact `named()` keys while doing it: they become public API and the
model document has to state them.

NOT in this ticket — [ticket 46](46-api-changes.md): `get_sfho_general(...)` and
`create_custom_parametrization(...)` becoming `from_*` constructors, and the
isentropic solvers folding into `SnB=`.

Resolved when sfho is renamed and the added-failure count is reported.
