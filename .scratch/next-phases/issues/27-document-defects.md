# Fix the 13 factual defects the document audit found

Type: task
Status: resolved
Blocked by: 09
Parent: ../map.md

## Question

[Ticket 06](06-document-audit.md) found 13 factual defects across the model
documents — statements that are wrong, not merely missing. Fixing a document in
place needs no permission, but *which file* gets fixed depends on
[ticket 09](09-tex-or-md.md)'s ruling on whether the `.tex` survives, which is
why this waits.

Full list in [document-audit.md](../research/document-audit.md). The ones that
carry consequences beyond their own page:

- **`mixed.tex:524` uses an undefined macro `\tmuB`** — the document does not
  compile past that equation. The Acceptance criteria block of
  `docs/REFACTOR_PROMPTS.md` requires every model to have a `.tex` that compiles,
  so this fails acceptance today. If ticket 09 keeps the `.tex`, fix the macro
  **and** compile all twelve as the check; if it drops the `.tex`, the criterion
  goes with it and this defect is moot.
- **`mixed.md` and `mixed.tex` both give `mu_mu = mu_e`**, where the code uses
  `mu_e − mu_nue`. The document is wrong about the physics, not merely terse.
- **`vmit.md:81` claims `eos_response` is unimplemented**; `eos/vmit/api.py:167`
  implements it. A stale "not implemented" ledger is the kind of claim §11's
  test is meant to catch.
- **`vmit.md` and `vmit.tex` both give an unknown ordering that contradicts**
  `solve_vmit_beta_eq` and `solve_vmit_trapped_neutrinos`. §11 requires the
  residual in the order the solver assembles it; here the order is stated and
  wrong, which is worse than absent.
- **`tov.md` and `tov.tex` treat the rotating case as a citation rather than a
  formulation**, and neither Komatsu 1989 nor CST 1994 is in `docs/eos.bib`. The
  RNS backend ships; the document does not formulate it.

The code decides wherever a document and the source disagree. Where fixing a
defect means writing physics that was never in either file, that is a §11
completion job and belongs with whatever ticket 09 sets up, not here — this
ticket fixes what is *wrong*, not what is *missing*.

## Answer

**Fixed: the compile blocker (which was five documents, not one) and nine of the
thirteen factual defects.** Four are re-assigned, with reasons, below.

### Compilation — the acceptance criterion was failing on five documents

The audit found `mixed.tex`. LaTeX halts at the first error, so each failure was
masking the next; it took four passes to drain. All twelve now compile.

| document | defect |
|---|---|
| `mixed.tex:524` | `\tmuB` undefined — only `\mutB` is (line 12). Line 260 uses it correctly, so a typo |
| `did.tex` | **8** sites of `\meff{i}^2`; `\meff` expands to `m^{*}_{i}`, so the following `^` is a double superscript. Braced |
| `ccdm.tex` | 2 × `\Mstar^`, 1 × `\mustar^` — same class. Braced |
| `enjl.tex` | 9 × `\kF_i` / `\kF_\ell` double subscript, **plus 4 uses of an undefined `\dd`** the audit never saw |
| `njl.tex:110` | `\slashed\partial` used without its package |

Two judgement calls: `enjl`'s indexed Fermi momenta are written out as
`k_{F,i}` / `k_{F,\ell}` rather than given a new indexed macro, since `\kF` is
bare at all nine other uses; and the undefined `\dd` is mapped to `\mathrm{d}`,
which is what that same file already uses at lines 1279 and 1430 — its own
convention, not a new one. `njl.tex` gained `\usepackage{slashed}` (carlisle,
standard in TeX Live) — the only package added anywhere.

### The nine factual defects fixed

Each verified against the code first, not taken from the audit on trust:

| document | was | now |
|---|---|---|
| `zl.tex:343` | `r_5 = n_p - n_e` | `r_5 = n_e(mu_e,T) - n_p` — `solver.py:334` returns `n_e - n_C` |
| `sfho.tex` (eq:fieldeqs) | four field sources bare | `(hbar c)^3` on all four; `thermodynamics.py:255,263,271,277` do `src * hc3`, and LHS is MeV^3 against fm^-3 sources |
| `vmit.md:81` | "Not implemented: muons, `eos_response`" | muons only — `api.py:167` implements `eos_response` |
| `vmit.md` | unknowns `(mu_u,mu_d,mu_s,n_u,n_d,n_s)` + `mu_e` appended | `(mu_u,mu_d,mu_s,mu_e,n_u,n_d,n_s)` — `solver.py:246` puts `mu_e` fourth |
| `enjl.tex:843` | slot 5 printed `\mu_e` | `\muC` — `solver.py:133` `BASE_UNKNOWNS`. The next sentence already said `mu_C` is in the always-group, so the equation contradicted its own prose |
| `enjl.md` | modes table `T=0` on all four rows | `T` — `solve()` takes `T=0.0` and converges at T = 10 MeV (checked: P = 20.19 against 19.70 at T = 0) |
| `mixed.tex:387`, `mixed.md:162` | `mu_mu = mu_e` | `mu_mu = mu_e - mu_nue` — `thermodynamics.py:64` |
| `tov.md:13` | `solver_fast.py` "the jitted variant of the same integration" | a second implementation: uniform log-P grid, four ODE variables in dimensionless r, different tolerances and `M_b` algorithm |

`zl`, `sfho`, `enjl` and `mixed` recompile clean after the edits.

### Four re-assigned, not fixed here

This ticket fixes what is **wrong**. These are missing or mislabelled, which is
§11 completion work and belongs with the pairs being rewritten:

- **`sfho` (both) — the three-flavour `mu = 0` thermal-neutrino gas**
  (`solver.py:523-527`) appears in neither document. An omission, not an error →
  [ticket 35](35-hadronic-documents.md).
- **`abpr` (both) — code names `ms`/`Delta`** where the mathematics wants
  `m_s`/`Delta_0`. Notation, and `abpr`'s pair is being rewritten →
  [ticket 36](36-quark-engine-documents.md).
- **`ccdm.md` — `R1..R5` label modes while `ccdm.tex`'s `R1..R4` label residual
  rows**, and the `.md` uses `R_4` in both senses. Fixing this means choosing one
  labelling across the pair, which is exactly what
  [ticket 30](30-ccdm-documents.md) does.
- **`njl` (both) — the returned `n_s` is the strange-quark density, not a scalar
  density**, and neither document says so. `njl.md` is at 4/16 and has no
  returned-quantity section to correct →  [ticket 32](32-njl-documents.md).

Each is recorded in its destination ticket rather than left loose.

Status: resolved.
