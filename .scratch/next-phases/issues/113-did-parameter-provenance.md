# How each DID parameter is fixed: theory, fit, or astrophysical data

Type: research
Status: open
Blocked by: -   (103 ruled it out of its own table and into here)
Parent: ../map.md

## Question

Raised by the user on [ticket 103](103-nmp-closures-four-models.md)
(2026-08-29): before DID's parameters can be sampled, it has to be clear which
of them are fixed by theoretical constraints, which by a fit, and which by
astrophysical data -- "so that we can properly constrain them with our
bayesian inferences".

103 confirmed the prerequisite: DID already reproduces its own paper.
`did/verify/run_full_check.py::check_nuclear_matter_parameters` compares
Table VI of arXiv:2511.15646 and passes at **0.84x tolerance** (B = -15.40,
K = 227.07, L = 59.85, S_2 = 32.12). So this ticket is provenance, not
reproduction.

103 kept it out of its own deliverable deliberately: that table is one row per
CLOSURE for four models, and DID has no closure to put in it. A per-coupling
provenance list for one model is a different table.

What it owes, per coupling: the quantity that fixed it, the kind of constraint
(theoretical identity, laboratory datum, Bayesian posterior over the paper's
18 observables, astrophysical observation), and whether it is free to vary.
Two DID-specific items 103 surfaced and did not settle:

- **The hyperon N-branches.** Each coupling carries a symmetric branch
  `g^S(0)` and a neutron-matter branch `g^N(0)`, and DID ties the hyperon
  N-branches to the nucleon one "in the same proportions as their ISM
  counterparts" (`parameters.py:79`, `_branch_pair`). A depth measured in
  symmetric matter constrains the S-branch ONLY. Whether that proportionality
  is a fit result or a modelling choice decides whether a non-symmetric `U_Y`
  is an input a sampler may vary.
- **The two symmetry energies.** 103 ruled `E_sym` means the CompOSE beta^2
  coefficient (arXiv:2203.03209 Eq. 6.4), i.e. DID's `S_2`, everywhere. DID's
  `S` -- the full ISM-to-PNM difference, -2.72 MeV away in its own Table VI --
  keeps its own name. Which of the two the paper's 18 observables actually
  constrained is a provenance question and belongs here.

## Gate

One table in `eos/did/did.md` (and `.tex`): every parameter, what fixed it,
what kind of constraint that is, and whether an inference may vary it. No code
change is required by this ticket.
