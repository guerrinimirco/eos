# One NMP vocabulary across dd2, sfho, did and zl: what is imposed, what is
# free, and what is a prediction

Type: grilling
Status: open
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
