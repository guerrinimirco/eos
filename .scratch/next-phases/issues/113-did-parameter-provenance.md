# How each DID parameter is fixed: theory, fit, or astrophysical data

Type: research
Status: resolved (2026-08-29)
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

## Resolution

Researched 2026-08-29 against arXiv:2511.15646 itself. No code touched; the
gate is met by `eos/did/did.md` (subsection "Provenance: what fixed each
number", the 19-row table) and `eos/did/did.tex`
(`Table~\ref{tab:provenance}` and `Table~\ref{tab:evidence}`), with the
working record in
[`research/did-parameter-provenance.md`](../research/did-parameter-provenance.md).

### The eighteen observables — and NONE of them is astrophysical

The count is 18, and it is the paper's own (Sec. IV C, values in Tables III
and IV). Nine are hyperon single-particle potentials `U_Y` at |k| = 0 from
Brueckner-Hartree-Fock on HAL QCD baryon-baryon interactions (Kohno et al.,
PRC 110, 054001): three iso-multiplet averages in ISM (Lambda -28.15 +/- 2.02,
Sigma +14.62 +/- 1.82, Xi -3.60 +/- 2.14 MeV) and six individual species in NM
(Lambda -25.42 +/- 1.78, Sigma+ +8.24 +/- 3.68, Sigma0 +15.73 +/- 1.70,
Sigma- +24.86 +/- 1.39, Xi0 -12.19 +/- 1.46, Xi- +5.79 +/- 2.59 MeV). The
other nine: n_0 = 0.150 +/- 0.010 fm^-3 (PREX, 208Pb), B = -15.6 +/- 0.6 MeV,
K = 240 +/- 20 MeV (GMR of 208Pb and 90Zr), S_2 = 32.0 +/- 1.1 MeV, the
finite-nucleus crossover M(0.11 fm^-3) = 1100 +/- 70 MeV, the two chiral-EFT
neutron-matter pressures P(0.08) = 0.472 +/- 0.036 and
P(0.16) = 2.898 +/- 0.404 MeV/fm^3, and the two heavy-ion ISM pressures
P(0.32) = 19.0 +/- 14.3 and P(0.56) = 106.8 +/- 22.0 MeV/fm^3.

Tally: **7 laboratory, 11 theoretical, 0 astrophysical.** This is the finding
that is easiest to get wrong, because the abstract names NICER and GW170817.
They enter twice, and neither time through the likelihood: as the discrete
post-hoc selection of `b_omega` from {0.60 ... 0.80} by the tidal
deformability, and as a check on the finished model. So `b_omega` is the one
parameter an astrophysical observation set, and it was set by picking the best
of five separate fits rather than by a posterior — which is why the table's
"may an inference vary it?" column reads "only by redoing the five fits".

### The hyperon N-branches: a MODELLING CHOICE

Settled from the paper, Sec. IV A, inside the parameter COUNT: "The
g^{N(0)}_{sigma Y}'s are in the same proportions to g^{N(0)}_{sigma N} as
their ISM counterparts", preceded by "We constrained all hyperon couplings to
fixed ratios relative to nucleon couplings, and similarly fixed the ratios
g^N_{MN}/g^S_{MN}." It is what holds the hyperon sigma sector at three
parameters instead of six; it appears in no table and no posterior.

The consequence for a sampler is the sharp half. The six NM potentials WERE
fitted — but through the sectors still free in neutron matter (the nucleon N
branches, `z`, and the two isovector hyperon vertices; Sec. VII says so
explicitly), never through a hyperon N branch of its own. So a non-symmetric
`U_Y` is legitimate DATA a sampler may feed DID, while `g^{N(0)}_{sigma Y}` is
NOT a dial: freeing it changes the model, not its parameters.

### Which symmetry energy: S_2, and only S_2

Sec. IV C names "the quadratic symmetry energy at saturation
S_2 = 32.0 +/- 1.1 MeV", Table III's row is `S_2`, and Sec. VII repeats it.
The full ISM-to-PNM difference `S` was never evidence: it appears only in
Table VI as the derived row S - S_2 = -2.72 against 1.2 +/- 1.5 MeV, a
comparison. This confirms
[ticket 103](103-nmp-closures-four-models.md)'s ruling at the source.

**`L` was not an observable either.** All of Table VI — L, L_2, Q, K_sym,
K_sym2 — is prediction; L is constrained only indirectly, through the two
chiral-EFT neutron-matter pressure points.

### The repository was right about all twelve of its claims

Every established item verified against the source and nothing was corrected.
Three things worth recording anyway:

- **`did.md`'s "Eq. 52" only LOOKS wrong.** The PDF prints "(4.1)"; the
  repository uses a consistent global sequential count — Sec. II (2.1)-(2.19)
  = 1-19, Sec. III (3.1)-(3.32) = 20-51, so Sec. IV's two equations are 52 and
  53, consecutive exactly as `couplings.py:188` and `nmp.py:127` have them.
  Left alone.
- **Two locational corrections to this ticket's own wording**, not repository
  errors: the branch-tying quote is in the `Parameters` class docstring at
  `parameters.py:77-82`, not `_branch_pair`'s (that is at :156-166); and
  `e = 1/3` is `couplings.E_ISOSPIN`, a module constant unreachable from a
  parameter set — which is the right encoding for a number nobody may sample.
- **Two wrinkles inside the paper itself.** Sec. VII says "17-parameter" where
  Sec. IV A says fifteen free (17 is the Table II row count; the documents'
  "fifteen" is the correct one), and `K` is quoted as 240 +/- 20 in the
  likelihood but 230 +/- 40 in Table VI's comparison column.

### What the paper does not say

The VALUE `e = 1/3` has no fit, datum or derivation behind it — the paper
argues only for the necessity of the tanh factor. Recorded in the table as a
modelling choice with no stated reason. And `alpha = 1` was fixed "to simplify
the Bayesian analysis" rather than on physics grounds, so it stays legal in
[0, 1] for a future sampler; the paper does not say what varying it costs.
