# general/ owes a public T = 0 entry point, and one loop is unbounded

Type: task
Status: open
Blocked by: 11
Parent: ../map.md

## Question

Two rows in `eos/general/`, grouped because both sit under every model's numbers
and therefore share one golden-reference gate.

**1. dd2 re-derives the T = 0 Fermi gas (finding 24).**
`eos/dd2/thermodynamics.py:66-94` defines `number_density_t0`,
`scalar_density_t0`, `eps_kin_t0` and `P_kin_t0` — the exact T = 0 ideal Fermi
gas, duplicating `eos/general/fermi_integrals.py:220 _compute_exact_T0`. §7 is
absolute: "All Fermi and Bose integrals, **at T = 0 and finite T**, come from
`eos/general/`. No model implements its own."

[Ticket 11](11-conformance-triage.md) ruled the fix lands in `general/`, not in
`dd2`, and the reason matters: **`general/fermi_integrals.py` exports no public
T = 0 entry point.** Everything at `:220` is private; the public names
(`:426 solve_fermi_jel`, `:453 solve_fermi_gl`, `:546 Fermi_Numerical`,
`:724 kinetic_thermo`) are finite-T. dd2 did not ignore the rule, it found no
door. So: promote the T = 0 closed forms to a public name, then dd2 imports them
and its four functions go.

The formulas are already algebraically identical and the audit found no numeric
discrepancy, so this must be a **no-op on every number**. §12 checks:
the **DD2 golden SNM point at n_B = 0.16 fm^-3**, the **DD2 published NMP/TOV
values**, and `test/baseline/dd2` at rtol = 1e-10. Movement beyond round-off
means the promotion is wrong and gets reverted, not accommodated.

While promoting, decide whether the new public name serves only dd2's four call
shapes or the general T = 0 case — §7 says the integral implementations may be
improved and alternatives added alongside, "each validated against JEL", and
**JEL is never removed**.

**2. The one unbounded loop (finding 20).**

    eos/general/fermi_integrals.py:519   while n_hi < n_target:   (mu_hi *= 1.5)
    eos/general/fermi_integrals.py:524   while n_lo > n_target:

No iteration counter, no cap. §6: "every solver has a bounded iteration count."
Geometric growth makes a hang practically unreachable, which is why this is a
ledger line as well as a fix — but every other loop in the repository is already
bounded (`grep -rn "while True" eos/` returns nothing; `general/tabulate.py:98`
caps at `T_cap=400.0`; `general/thermodynamics_leptons.py:343-345` raises past
`mu_max`; `general/solve.py:72-83` allows three attempts "because a parameter
scan must always get an answer back"). Bound it in the same style, and make
exhaustion a **returned status**, not a raise (§6).

Report added failures against `output/_audit/pytest_before_with_crust.txt`.
