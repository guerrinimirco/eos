# The verdict: did exact solves reach 1 ms/pt and 60 s, and what ports?

Type: grilling
Status: open
Blocked by: 05, 06, 07, 08
Parent: ../map.md

## Question

The map's closing ticket. Three things to settle, with every number from a
closed ticket rather than re-derived.

### 1. The measurement

Assemble the product of the levers against the two targets:

| target | baseline (02) | after 05 | after 06 | after 07 | reached? |
|---|---|---|---|---|---|
| ms/pt, beta-eq T=0 CSC table | | | | n/a | 1 ms |
| s per 200-point hybrid table | | | | | 60 s |

State it plainly, including if it fell short. A 450-900x ask met at 200x is a
real result and a useful one; reporting it as success is not.

### 2. What the production port is

The prototype lives on `njl-speed` and is a spike. The port is a separate
effort and this ticket **specifies** it rather than doing it:

- which files change, and whether `backends/` stays deletable in the ported
  form (CLAUDE.md section 5)
- what goes in `eos/njl/verify/` and `test/njl/` — every new physics invariant
  needs a `verify/` entry, every new behaviour a test (section 12)
- whether `build_fast_table` and its `FAST_MODES` restriction survive, or are
  retired now that exact solves are cheap
- what `eos/njl/njl.tex` and `.md` owe: section 11 requires every equation the
  code solves to be written out, and a compiled Newton loop with re-inflation
  rescues is solver behaviour the document currently does not describe
- whether `analytic_jac` flips to default-on across `eos/mixed`
- a landing measurement per section 12: `test/run_clean_suite.sh` at a landing
  point, certificate path cited, **including any DISCARDs**

### 3. Which fog graduates

Judged on what was measured, not on appetite:

- **Continuation along the sweep** — graduates if 05's pre-screen left the
  enumeration still dominant.
- **The interpolated phase surface** — graduates if the 60 s mixed target was
  missed, or if the 1 s figure becomes the requirement. It is the only route to
  1 s and it is a different destination, so it graduates as a **new map**, not
  as tickets on this one.
- **C or Fortran via f2py** — graduates only if 06 measured numba short and said
  by how much.
- **The BayEoS `njl` registry entry** — now specifiable, because the per-theta
  table build time is finally a known number. Likely a new map.
- **Finite T and `fixed_YC` as delivered paths** — 08 measured them; making them
  production paths is a separate effort.

## Gate

- The table filled in from closed tickets, each number citing its ticket.
- A written port specification a hand-off session can execute without
  re-deriving anything.
- Each fog entry above ruled: graduated (to where), or left in fog (why).
- The map's Decisions-so-far complete, and the map marked closed.
