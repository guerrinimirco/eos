# One NMP vocabulary across dd2, sfho, did and zl: what is imposed, what is
# free, and what is a prediction

Type: grilling
Status: resolved (2026-08-29)
Blocked by: -   (93 is a sibling, not a blocker)
Parent: ../map.md

## Question

Raised by the user (2026-08-27), re-opening the NMP <-> couplings map in both
directions across all four models with a nuclear sector. §5 already requires
`compute_nmp` / `invert_nmp` of these four and only these four; what it does
NOT say is which NMPs each closure may take, and today each model answers
differently for reasons that are partly physics and partly history.

**The counting is the whole argument and it is not a matter of taste.** A
closure can impose exactly as many independent conditions as it has free
couplings in that sector. Fewer leaves a direction unfixed; more is
over-determined and has no solution except by accident. So the question "can
{n_sat, m*/m, E_sat, K_sat, L_sym, E_sym, Q_sat, K_sym} fix a model
univocally?" has a per-sector arithmetic answer before it has a numerical one:

    model   isoscalar couplings          isovector couplings
    dd2     6  (G_s, b_s, c_s, G_w,      2  (G_rho(n_sat), a_rho)
               b_w, c_w) - 1 structural
               cross-constraint  ->  5 free
    sfho    4  (g_sN, g_wN, g2, g3),     2  (g_rhoN, b1)
               c3 held as a 5th
    did     ~10 per meson-branch pair    ~6, and TWO symmetry energies
    zl      3  (a0, b0, gamma)           3  (a1, b1, gamma1)

Consequences that follow immediately, none of which is currently written in
one place:

1. **K_sym cannot be imposed anywhere except `zl`.** dd2 and sfho have two
   isovector knobs and three isovector NMPs on offer (E_sym, L_sym, K_sym), so
   two is the maximum and K_sym is a prediction by arithmetic, not by
   preference. `zl` has three and is the one model that can take it.
   **No longer hypothetical**: [ticket 104](104-zl-analytic-inversion.md)
   shipped it on 2026-08-28, in closed form -- gamma1 falls out linearly from
   `K_sym,V = 9[gamma1(gamma1-1) b1 - gamma(gamma-1) b0]` because E_sym and
   L_sym alone already fix the product `(gamma1-1) b1`. So this row of the
   table is a built API, not an arithmetic claim, and this ticket's job on it
   is now vocabulary rather than feasibility.
2. **`zl` cannot take m*/m at all** — it has no scalar field and therefore no
   effective mass (`thermodynamics.py:16` states this). So the eight-NMP list
   in the question is not one list: the imposable set is model-shaped.
3. **Q_sat is where the conditioning dies**, and it is measured, not
   suspected. In `zl` it does not merely condition badly, it is not FREE:
   [ticket 104](104-zl-analytic-inversion.md) confirmed
   `Q_sat,V = 3(gamma - 2) K_sat,V` exactly, one power-law term supplying
   both, so {n_sat, E_sat, K_sat} already determine it and a prior over
   (K_sat, Q_sat) in zl lives on a curve rather than in a plane. That is a
   fourth answer to "can this NMP be imposed?" beside imposed / predicted /
   absent, and the vocabulary this ticket settles needs a word for it.
   The same ticket also confirms row 2: zl's isoscalar sector is exactly
   three conditions -- `a0 + b0 = Eb`, `P(n_sat) = 0`, `9 gamma(gamma-1) b0 =
   Kb` -- against its three isoscalar couplings, with no room for a fourth. It is a third density derivative of E/A: `h`-sweeps under
   [ticket 67](67-dd2-t0-adoption.md) put its stencil floor at 1.5e-3
   relative, three orders above the four h-exact keys, and dd2's 6x6 closure
   that imposes it inverts 7 of 187 (K_sat, Q_sat) cells at zero restarts and
   115 of 187 at sixty-four (`dd2/nmp.py:84-96`).
   [Ticket 93](93-invert-nmp-basin-lottery.md) is the same floor seen in the
   solver's basin selection.
4. **sfho's Q_sat option is absent for a stated reason** — it would need a
   fifth isoscalar knob and the candidate is `c3`, a high-density vector
   parameter saturation says little about (`docs/DEFERRED.md`). That is the
   same trade dd2 makes by pinning `c_omega`, and the two are not described in
   the same words anywhere.
5. **`did` has no inverse map and the obstacle is not effort.** Its couplings
   are a Bayesian posterior over 18 observables, its NMPs are published as
   predictions, and it carries TWO symmetry energies (S and S_2, differing by
   -2.72 MeV in its own Table VI) because B is not quadratic in beta. An
   inversion must first say WHICH symmetry energy `E_sym` means. Until that is
   ruled, "did has no `invert_nmp`" is a physics statement, not a gap.

### What has to be decided

- **The recommended closure per model, named once**, so that a sampler varying
  NMPs knows what it is allowed to vary. The candidate, on the evidence above:
  {n_sat, E_sat, m*/m, K_sat} + {E_sym, L_sym} imposed everywhere it fits,
  Q_sat and K_sym reported as predictions, and the model's structural
  condition (dd2's cross-constraint, sfho's held `c3`) named in the same
  breath as the NMPs so it is visible that the sector was closed by seven
  conditions and not six.
- **Whether "imposed vs predicted" becomes API**, i.e. whether
  `InversionStatus.predictions` is the shape all four models return, so a
  caller reads one field rather than a per-model docstring.
- **Whether the published set is required to be a root of its own closure.**
  For dd2 it is NOT: the published couplings violate the cross-constraint by
  2.2e-3, so no seed recovers them and a round trip cannot be a test
  (`dd2/nmp.py`). If inference is to be run "around DD2", the point being
  sampled around is not in the closure's solution set, and that deserves
  stating on the public surface rather than in a module docstring.
- **The hyperon and Delta sectors are a separate closure and are not covered
  by any NMP.** They are fixed by single-particle potentials — U_Lambda,
  U_Sigma, U_Xi at saturation in SNM, and U_Delta — with the VECTOR couplings
  taken from SU(6) and only the scalar ones inverted
  (`dd2/nmp.py:489-520`, `sfho/parameters.py:37`). SU(6) is an ASSUMPTION, not
  a measurement, and the ratios it fixes (x_omegaLambda = 2/3,
  x_phiLambda = -sqrt(2)/3, x_rhoSigma = 2) are exactly the numbers a hyperon
  inference would want free. Decide whether they stay pinned, become
  parameters with SU(6) as the default, or get a documented breaking
  parameter.
- **`did`'s hyperon sector needs two numbers per hyperon where the others need
  one**, and it currently assumes its way out of the second: each coupling
  carries a symmetric branch g^S(0) and a neutron-matter branch g^N(0), and
  DID ties the hyperon N-branches to the nucleon one "in the same proportions
  as their ISM counterparts" (`parameters.py`). A depth measured in SNM
  constrains the S-branch only. Whether the proportionality is a fit result or
  a modelling choice, and whether a non-symmetric U_Y should be an input, is
  the DID-specific half of this ticket.
- **A failed inversion is a `None` in dd2 and a `RuntimeError` in zl**, and
  the two cannot both be right. Surfaced by
  [ticket 93](93-invert-nmp-basin-lottery.md) (2026-08-28), which made
  `ok=False` reachable where it previously was not and then hit its own
  crash: `dd2.nmp.from_nmp` returns `None` without `return_status`, and the
  None travels until `solver.py` raises `'NoneType' object has no attribute
  'kernel_masses'` — a §6 non-convergence that reads as an AttributeError two
  layers down. `zl.nmp.from_nmp` raises `RuntimeError` on the same event.
  93 documented dd2's None rather than changing it, because which of the two
  §6 means is a question about all four models and this ticket owns it.

- **sfho's inversion silently invalidates a hyperonic base** and says so in
  `docs/DEFERRED.md`: hyperon couplings are stored absolutely, derived from
  ratios against nucleon couplings the inversion has just changed, so
  `from_potential_depths` must be re-run by hand. Decide whether holding the
  DEPTHS or holding the RATIOS is the default, because right now neither is.

## Gate

A ruling, plus the closure of each model stated in ONE table — in
`docs/STRUCTURE.md` or a shared docstring, not four times — naming for each
sector: the free couplings, the imposed conditions, the structural conditions,
and the predictions. Any code change it authorises is a separate ticket; this
one is the ruling.


## Resolution (2026-08-29)

**Ruled by the user.** The deliverable is one table of six closures for four
models in `docs/STRUCTURE.md` section 6, "The nuclear-matter-parameter
closures". Every cell in it is measured at the model's own published point;
nothing was carried over from the ticket's own arithmetic without checking it,
and two of the ticket's four counting rows were wrong.

### The counting table, corrected against the source

- **dd2's row was stale.** [Ticket 105](105-dd2-isoscalar-conditioning.md)
  removed the cross-constraint, so dd2 has **6 isoscalar couplings and no
  structural condition** -- not "6 minus 1 structural -> 5 free". The default
  closure pins `b_sigma` and `c_omega` and frees four.
- **sfho's isovector row undercounted by eight.** It is not two couplings; it
  is **ten** (`g_rho_N, a1..a6, b1..b3`) of which two are freed and eight
  pinned at published values. And a sixteenth, **`c4`, is invisible to every
  nuclear-matter parameter** -- its Jacobian column is exactly zero, because
  the rho field vanishes in symmetric matter and `esym`'s closed form contains
  no `c4`. Six free, nine pinned, one unreachable.
- **The "structural condition" row was deleted from the deliverable.** After
  105 no model has one, and an empty row invites a refill.

### Q1 -- the selection rule, and it is not 105's

105's rule was "pin the most collinear, free the most orthogonal", scored by
the condition number. **Run on SFHo for the first time, it picks the wrong
pair.** Isovector Jacobian at the published point, all 45 pairs ranked:

    (a1, a2)     cond 2.413   sigma_min 0.069     <- cond's winner
    (g_rho, b1)  cond 2.665   sigma_min 0.483     <- shipped, and right
    (a1, a4)     cond 4.717   sigma_min 0.029
    (a1, b1)     cond 5.547   sigma_min 0.107

`(a1, a2)` freezes `g_rho_N` and its columns are ten times weaker. The cause is
that **`cond` is sigma_max / sigma_min, so it divides out the absolute strength
of the weakest knob** -- exactly the number that decides whether the knob can
reach an inference prior. sfho's columns span four orders of magnitude and
dd2's span a factor 2.7, which is why the flaw never showed in dd2: re-scored
under `sigma_min`, dd2's six pin choices rank in the identical order
(`b_sigma + c_omega` first at 128.0 / 0.471, reproducing 105's numbers to three
digits).

**Adopted: pin whichever subset leaves the largest `sigma_min`, then confirm
with a basin scan over a grid of targets, which may veto a locally-best
choice.** 105's collinearity rule survives inside it as the equal-strength
special case. The user's addition (Q16): the confirmation is where "how fast"
and "are the solutions findable" enter -- `sigma_min` is a local statement at
one point, basin coverage is the global one, and a rule with only the local
half repeats the mistake `cond` made.

The rule immediately overturns one shipped choice, which is
[ticket 115](115-dd2-qsat-pin-recheck.md).

### Q2 -- the vocabulary, ruled by the user

Two words, not five: **input** (fed in to determine couplings) and **computed**
(obtained from them). "rigid" was refused -- zl's `Q_sat` is computed, and the
identity that makes it rigid is a footnote, not a category. "blocked" became
moot when [ticket 111](111-dd2-analytic-nmp-derivatives.md) landed mid-ticket.
**absent** survives for a quantity the model does not have (zl's `m*/m`).

### Q3 / Q12 -- six closures, not four models

`input` and `computed` are properties of a CLOSURE, and two models offer a
choice: dd2 (`impose_Q_sat`) and zl (`gamma1` named, or `K_sym` imposed). One
row per closure, so no footnote carries physics. **zl's `gamma1` gets its own
cell** as a caller-named input: it is the one place the API demands a decision
from the sampler, and a footnote is how that gets defaulted by accident.

Two factual answers the ticket asked for:

- **zl cannot use `Q_sat` as its sixth datum.** `Q_sat = 3(gamma - 2) K_sat` in
  the interaction part, and gamma1 is isovector while Q_sat is isoscalar. The
  alternative is `K_sym`, which [ticket 104](104-zl-analytic-inversion.md)
  already shipped in closed form.
- **`Q_sat` is a legal input in dd2 alone.** Confirmed independently of 111:
  the closure converges at `iso_res = 1.5e-12` and returns the published
  couplings to 9.1e-5. sfho's only candidate fifth isoscalar knob is `c3`,
  whose column is **550x weaker** than `g_sigma`'s (0.033 against 18.2);
  admitting it drops `sigma_min` from 0.051 to 0.0063. That is structural --
  saturation does not constrain a high-density omega^4 term -- so unlike dd2's
  it is not a floor analytic derivatives would remove.
  `docs/DEFERRED.md`'s prose reason is now that number.

### Q4, Q5 -- already the shape, and one model out of step

`InversionStatus.predictions` is **already** the contract in all three models
that have an inversion (`dd2:313`, `sfho:463`, `zl:221`) -- same field, same
`{Q_sat, K_sym}` contents. 104 and 105 shipped it without anyone writing it
down. Nothing is built; it is recorded.

`from_nmp` raises in sfho and zl and returns `None` in dd2. **Ruled: dd2
conforms.** `invert_nmp` is the section-6 boundary and returns a status;
`from_nmp` is the face for a caller that has declared it will not score
failures. Carried to [ticket 114](114-nmp-api-conformance.md).

### Q6 -- the published set as a root, measured on all three

The user's test was "use the NMP equal to literature and check the couplings
are compatible". Run:

    dd2    1.6e-4 from the literature quote     (DD2 prints 4-6 digits)
    sfho   2.8e-2 from the quote
           1.3e-3 with an exact m*/m, everything else still rounded
           0      at full precision
    zl     7.2e-3 exact, 7.15e-3 at four figures, 3.8e-3 at three

**The test measures the paper's rounding, not the closure.** SFHo's entire
factor of 22 is one two-digit entry: `m*/m = 0.76` against a true 0.761564.
ZL's 7.2e-3 does not move with precision at all, because its published
couplings saturate 0.3% below their own `n0` and so are a root of no closure --
which is why this is a **reported property** and not a gate. Both quotes get
shipped, in [ticket 114](114-nmp-api-conformance.md).

### Q7 / Q9 / Q21 -- the hyperon and Delta sectors

One rule for every RMF, stated by the user twice and identically: **`x_sigma_H`
from the depth `U_H`; `x_omega/rho/phi_H` = SU(6) x a free factor.** The count
is **nine** -- one per (meson, multiplet) -- and the shipped sets are why three
would not do: SFHoY is 1.5 on omega AND phi for Lambda and Sigma and 1.875 on
both for Xi, with rho left at SU(6) (`sfho/parameters.py:391-410`), which
neither a per-meson nor a per-multiplet factor can express. **The Delta sector
takes no factors**: `x_Delta_sigma`, `x_Delta_omega`, `x_Delta_rho` are free
variables directly. Carried to
[ticket 112](112-su6-vector-ratios-as-parameters.md), which shipped it on
2026-08-29 and folded [ticket 106](106-su6-breaking-rescaling.md) in, closing
that one as superseded rather than unblocking it.

### Q8 -- `E_sym` means the CompOSE definition, checked at the source

Read from the CompOSE Reference Manual v3.01 (Typel et al., arXiv:2203.03209)
section 6.1 rather than assumed:

    Eq. (6.4)   E_sym(n_b) = (1/2) d^2 E(n_b, alpha) / d alpha^2 |_{alpha=0}
    Eq. (6.1)   alpha = (n_n - n_p)/(n_n + n_p) = 1 - 2 Y_q
    (6.10) J = E_sym(n_sat)      (6.11) L = 3 n_b dE_sym/dn_b
    (6.7)  K = 9 n_b^2 E''       (6.9)  Q = 27 n_b^3 E'''
    (6.12) K_sym = 9 n_b^2 d^2 E_sym / dn_b^2

So **`E_sym` is the beta^2 expansion coefficient -- DID's `S_2`** -- in every
model, and DID's `S` (the full ISM-to-PNM difference, -2.72 MeV away in its own
Table VI) keeps its own name. Every derivative convention already matches what
dd2, sfho and zl compute.

One wrinkle the manual raises and the ruling declines: Eq. (6.5) prescribes a
finite-difference form, the symmetric `(1/2)[E(-1) - 2E(0) + E(1)]` over pure
neutron AND pure proton matter. That is CompOSE's approximation for when the
quartic term is negligible, and DID is the model where it demonstrably is not;
it also needs pure proton matter, which empties DID's neutron sector. **(6.4)
is the definition; how each model differentiates it is numerics and stays in
its docstring with the measured spread.**

**The prerequisite the user set for DID is already met.**
`did/verify/run_full_check.py::check_nuclear_matter_parameters` compares
Table VI of arXiv:2511.15646 and passes at **0.84x tolerance** (B = -15.40,
K = 227.07, L = 59.85, S_2 = 32.12). The provenance half -- which parameter is
fixed by theory, by a fit, or by astrophysical data -- is
[ticket 113](113-did-parameter-provenance.md), kept out of this table because
that table is one row per closure and DID has no closure to put in it.

### What the deliverable says

`docs/STRUCTURE.md` section 6 gains "The nuclear-matter-parameter closures":
the two-word vocabulary, the CompOSE definitions, the six-closure table, the
three footnoted cells (zl's rigid `Q_sat`, sfho's unreachable `c4`, dd2's sole
claim on `Q_sat`), the selection rule with the dd2/sfho disagreement that
motivates it, a second table of what each closure is worth, the hyperon/Delta
sector as a separate closure, and the section-6 boundary sentence.

### Tickets this authorises

Code changes were out of scope by the ticket's own last line. Raised:

- [112](112-su6-vector-ratios-as-parameters.md) SU(6) factors as parameters,
  Delta ratios free -- shipped 2026-08-29, subsuming
  [106](106-su6-breaking-rescaling.md)
- [113](113-did-parameter-provenance.md) DID parameter provenance
- [114](114-nmp-api-conformance.md) dd2's `from_nmp` raises; full-precision
  published quotes in all three
- [115](115-dd2-qsat-pin-recheck.md) the Q_sat pin the new rule overturns
- [116](116-sfho-analytic-nmp-derivatives.md) sfho analytic derivatives, for
  interpreter reproducibility and explicitly NOT for imposing `Q_sat`

### Gate

The gate was "a ruling, plus the closure of each model stated in ONE table".
Met: one table, six rows, in `docs/STRUCTURE.md` section 6, not repeated in any
model's document. No code changed and no number moved -- everything measured
here was measured against the shipped code as it stands.

Status: resolved (2026-08-29).
