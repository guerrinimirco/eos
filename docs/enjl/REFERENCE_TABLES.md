# The Xia reference tables — column meanings and verified identities

The five files in `test/enjl/reference/` were produced by the author's own Maple
worksheet for the extended NJL model of

> C.-J. Xia, *Extended NJL model for baryonic matter and quark matter*,
> Phys. Rev. D **110**, 014022 (2024), arXiv:2405.02946

They are beta-equilibrium, charge-neutral stellar matter along a density grid,
one file per parameter set. Because they come from the code that made the
paper's figures, they constrain the model far more tightly than the two or three
significant figures the paper prints — which is why they are the validation
target for `eos/enjl/` rather than Table II alone.

Every identity stated below was checked numerically against the tables before
this document was written; the tolerance quoted is the observed agreement, not
an aspiration. Where the tables are internally inconsistent, that is said
explicitly rather than smoothed over.

---

## 1. Files and parameter sets

| file | `f_q` = f_u = f_d = f_s | `B` [GeV/fm³] | rows | notes |
|---|---|---|---|---|
| `Beta_fq1.0_B0.dat` | 1.0 | 0 | 280 | |
| `Beta_fq1.0_B1.dat` | 1.0 | 1 | 279 | |
| `Beta_fq0.7_B0.dat` | 0.7 | 0 | 253 | coarse grid above n_b ≈ 3 fm⁻³ |
| `Beta_fq0.7_B1.dat` | 0.7 | 1 | 273 | |
| `Beta_fq0.5_B1.dat` | 0.5 | 1 | 383 | extra columns, see §6 |

`f_Λ = 1.0626` and `f_p = f_n = 1` in all of them. The nominal grid is
`n_b = 0.01 … 10 fm⁻³` in steps of 0.01 below 1 fm⁻³ and coarser above, plus a
handful of **off-grid densities that are the Maxwell coexistence endpoints** —
see §5.

## 2. Columns

Tab-separated, one header line. Stray empty column names appear where Maple
wrote a bare tab; `test/enjl/reference.load_reference` drops them.

| column | meaning | unit |
|---|---|---|
| `nB` | total baryon density n_b, Eq. (7) | fm⁻³ |
| `muu, mud, mus` | quark chemical potentials µ_q, Eq. (15) | MeV |
| `mup, mun, muL` | baryon chemical potentials µ_b, Eq. (14) | MeV |
| `mue, mumu` | lepton chemical potentials, Eq. (16) | MeV |
| `epa` | energy per baryon E/n_b (**includes** the rest mass) | MeV |
| `E` | energy density E, Eq. (13), with E₀ subtracted | MeV/fm³ |
| `P` | pressure, Eq. (19) | MeV/fm³ |
| `nu, nd, ns, np, nn, nL, ne, nmu` | number densities n_i, Eq. (11) | fm⁻³ |
| `Mp, Mn, ML` | baryon effective masses M_i, Eq. (4) | MeV |
| `Mu, Md, Ms` | quark effective masses M_q, Eq. (5) | MeV |
| `Sigmau, Sigmad, Sigmas` | **effective** quark scalar density n̄ˢ_q, Eq. (6) | fm⁻³ |
| `nsp, nsn, nsL` | baryon scalar densities nˢ_i, Eq. (12) with Λ = 0 | fm⁻³ |
| `nsu, nsd, nss` | quark-flavour scalar densities, vacuum term **removed** — see §3 | fm⁻³ |
| `munr` | the **coexistence** µ_b — see below | MeV |
| `fq` | quark baryon fraction n_b^Q / n_b | — |

`mumu` is written as the muon *mass* (105.66) on rows where no muons are
present; on rows with `nmu > 0` it equals `mue` exactly. Treat `mumu` as
meaningful only when `nmu > 0`.

**`munr` is not a duplicate of `mun`, and the difference matters.** On rows
where baryons are still present the two are bit-identical, which is why it looks
redundant. Once the baryons have dissolved — `n_p + n_n + n_Λ` down at
~10⁻⁵ fm⁻³ or below, past the deconfinement transition — they part company, and
there `munr = µ_u + 2µ_d` to 10⁻¹² MeV while `mun` continues to report the
vanishing neutron's own potential. So:

* **`munr` is the baryon chemical potential to use throughout**, in both
  phases. It is what a solver should match across a phase boundary.
* `mun` is the neutron's µ_i, which coincides with µ_b only while neutrons
  exist in appreciable number.

In `Beta_fq0.7_B0.dat` this affects the 33 rows at n_b ≥ 5.60 fm⁻³, and in
`Beta_fq0.5_B1.dat` the 70 solved rows at n_b ≥ 2.0 fm⁻³ (where `fz = n_p/n_b`
is also written as `--`, the file's own marker for "no protons left").

## 3. The two scalar-density columns are different quantities

This is the single most important thing to get right, and the naming actively
misleads. Write the T = 0 scalar density of one species as the sum of a medium
piece and a vacuum piece,

```
n^s(k_F, M, g, Λ) = (g M³/4π²) [ x√(x²+1) − arcsinh x ]      # medium, x = k_F/M
                  − (g M³/4π²) [ y√(y²+1) − arcsinh y ]      # vacuum, y = Λ/M
```

Then, verified over every surviving row of every file — to 8×10⁻⁵ fm⁻³ or
better for `nsq`, and to between 7×10⁻⁷ and 6×10⁻³ fm⁻³ for `Sigmaq` depending
on the file (§4 has the per-file breakdown; `fq0.7_B0` is the loose one):

```
nsq     =  medium(q)                    + α_S Σ_{i=p,n,Λ} N^q_i nˢ_i
Sigmaq  =  medium(q) + vacuum(q)        + α_S Σ_{i=p,n,Λ} N^q_i nˢ_i   =  n̄ˢ_q
```

`Sigmaq` is therefore the n̄ˢ_q of paper Eq. (6) — the quantity that enters the
gap equation and the condensate energy — while `nsq` is the same object with the
Dirac-sea term dropped. **The gap equation must be fed `Sigmaq`, not `nsq`.**

Confirmation: substituting the `Sigmaq` columns into Eq. (5),

```
M_q = m_q0 − 4 G_S n̄ˢ_q + 2K n̄ˢ_u n̄ˢ_d n̄ˢ_s / n̄ˢ_q
```

reproduces the `Mu, Md, Ms` columns to a relative error ≤ 2×10⁻⁵ across the
whole density range — i.e. the tables satisfy their own gap equation, and
`g_σ σ_q = 4 G_S n̄ˢ_q` is the correct reading of Eq. (5) combined with Eq. (8).

**One clamp to know about.** On rows where a quark mass has reached its current
mass exactly (`Mu == Md == 5.5`, i.e. 215 of 280 rows in `Beta_fq1.0_B0.dat`)
the Maple run writes `Sigmau = Sigmad = 0` rather than the small residual value
the formula gives (≈ 0.009–0.012 fm⁻³). Those rows cannot be used to check the
gap equation, and a solver that reproduces them will not produce an exact zero.
Mask on `abs(Mq − m_q0) < 1e-9`. The `s` channel is never clamped in these
files.

`nsp, nsn, nsL` are plain Eq. (12) with Λ = 0, agreeing to ≤ 7×10⁻⁴ relative.

## 4. Verified identities — what a reimplementation must reproduce

Using only table columns and the published parameters, these all hold:

**Agreement varies by two to five orders of magnitude between files**, so a
single universal bound would be either useless or wrong. Worst case per file,
over the rows that survive the exclusions of §4b-bis and §6, straight from
`verify_reference_tables.py`:

| identity | fq1.0_B0 | fq1.0_B1 | fq0.7_B0 | fq0.7_B1 | fq0.5_B1 |
|---|---|---|---|---|---|
| n_b sum rule, Eq. (7) [fm⁻³] | 1.7e-6 | 9.4e-11 | 3.0e-6 | 3.8e-6 | 3.9e-11 |
| `epa` × n_b = `E` [MeV/fm³] | 1.1e-2 | 8.8e-7 | 4.4e-2 | 6.0e-3 | 8.4e-7 |
| P = Σ µ_i n_i − E, Eq. (19), abs [MeV/fm³] | 4.3e-2 | 4.9e-6 | 1.0e-1 | 3.6e-2 | 1.4e-6 |
| — same, relative to P | 2.3e-4 | 7.1e-9 | 7.2e-6 | 5.1e-4 | 6.6e-9 |
| gap equation, Eq. (5), relative | 2.0e-5 | 3.6e-7 | 2.3e-5 | 3.3e-6 | 1.6e-5 |
| `nsq` identity, §3 [fm⁻³] | 8.3e-5 | 7.7e-12 | 6.8e-8 | 3.5e-6 | 7.6e-8 |
| `Sigmaq` identity, §3 [fm⁻³] | 1.5e-4 | 6.6e-7 | 6.4e-3 | 1.3e-5 | 9.5e-4 |
| `nsp/nsn/nsL`, Eq. (12), relative | 6.7e-4 | 9.8e-8 | 9.1e-3 | 3.9e-5 | 7.9e-8 |
| `fq` = (n_u+n_d+n_s)/3/n_b | 5.0e-16 | 4.9e-17 | 5.6e-16 | 5.1e-16 | 4.9e-15 |

Charge neutrality (Eq. 24) and beta equilibrium (Eq. 23) need more explanation
and get their own tables below.

Two readings of this table matter. `Beta_fq1.0_B1.dat` and the solved rows of
`Beta_fq0.5_B1.dat` are converged to 10⁻⁶ or better on nearly everything and are
the files to gate tightly against. `Beta_fq0.7_B0.dat` is the loosest even after
its nine bad rows are removed — its `Sigmaq` and baryon scalar-density residuals
sit at 10⁻³-10⁻², two to five orders of magnitude worse than the same identity
in the other files — so gate it at 1% and do not read anything into the
difference.

The beta-stability check needs care in three ways, and getting any of them wrong
produces residuals of hundreds to thousands of MeV that look like a broken
model rather than a misread column.

1. **Restrict to species that are present.** For a species at vanishing density
   the tables still print a µ_i, and it is not B_i µ_b − q_i µ_e but the
   threshold value the solver last held. A cut at n_i > 10⁻⁴ n_b works across
   all five files; a cut at absolute 10⁻¹⁰ fm⁻³ does not (it admits rows with
   n_p ~ 10⁻¹⁰ where the residual is 2700 MeV).
2. **Use `munr`, not `mun`, as µ_b** (§2).
3. **Do not read µ_e off the `mue` column when no electrons are present.**
   Exactly like `mumu`, `mue` is written as the electron *mass* (0.511) on those
   rows — but µ_e is still nonzero and still fixes the isospin splitting.
   Recover it as µ_e = µ_d − µ_u, which follows from Eq. (23) with
   q_d − q_u = −1. In `Beta_fq0.5_B1.dat` above n_b = 2 fm⁻³ the true µ_e is
   ≈ 18.7 MeV while the column says 0.511; using the column leaves a 6-12 MeV
   residual in all three quark channels.

Done all three ways, the worst-case residual over all six species is 0.52 MeV
on `fq1.0_B0`, 1.19 MeV on `fq1.0_B1`, 0.19 MeV on `fq0.7_B1`, 0.17 MeV on the
solved rows of `fq0.5_B1`, and 0.49 MeV on `fq0.7_B0` once the nine bad rows of
§4b-bis are excluded (28 MeV if they are not). The worst channel is always a species
near its onset threshold, where the density is small and the printed µ is least
resolved — d in `fq1.0_B0` and `fq0.7_B1`, Λ in `fq1.0_B1`, s in `fq0.7_B0` and
`fq0.5_B1`. The n channel is exact by construction in every file (µ_n ≡ µ_b),
and p is the next tightest at 3×10⁻⁷ to 2×10⁻³ MeV. A gate at 1.5 MeV passes all five files
as a whole; if you want a tight gate, put it on p and n and let the onset
species carry a looser one.

Per-channel worst cases, for choosing a gate:

| file | p | n | Λ | u | d | s |
|---|---|---|---|---|---|---|
| `Beta_fq1.0_B0.dat` | 2.3e-3 | 0 | 4.7e-2 | 1.1e-2 | **5.2e-1** | 6.9e-3 |
| `Beta_fq1.0_B1.dat` | 1.2e-4 | 0 | **1.19** | — | 2.0e-7 | — |
| `Beta_fq0.7_B0.dat` | 7.4e-5 | 0 | 4.5e-3 | 9.4e-4 | 4.8e-4 | **4.9e-1** |
| `Beta_fq0.7_B1.dat` | 8.4e-4 | 0 | 6.3e-2 | 1.7e-3 | **1.9e-1** | 1.7e-1 |
| `Beta_fq0.5_B1.dat` | 3.0e-7 | 0 | 1.7e-6 | 2.3e-8 | 2.3e-8 | **1.7e-1** |

(MeV; "—" = never above the 10⁻⁴ n_b presence cut in that file.)

Charge neutrality (Eq. 24) also needs a *relative* tolerance in the
quark-dominated regime, because there the residual is a small difference of
large numbers and the Maple solve converged less tightly on nearly-symmetric
quark matter. Worst case per file, over solved rows:

| file | worst \|Σ q_i n_i\| [fm⁻³] | at n_b | worst relative to n_u+n_d+n_s | at n_b |
|---|---|---|---|---|
| `Beta_fq1.0_B1.dat` | 5×10⁻¹¹ | 7.0 | 5×10⁻¹² | 0.670 |
| `Beta_fq1.0_B0.dat` | 2.5×10⁻⁶ | 0.666 | 1.9×10⁻⁵ | 0.666 |
| `Beta_fq0.7_B1.dat` | 6.2×10⁻⁶ | 0.449 | 6.7×10⁻⁶ | 0.534 |
| `Beta_fq0.7_B0.dat` | 1.0×10⁻² | 3.900 | 1.3×10⁻³ | 3.900 |
| `Beta_fq0.5_B1.dat` | 3.8×10⁻² | 9.700 | 1.4×10⁻³ | 6.400 |

The absolute and relative worst cases sit on different rows in the two loosest
files, so quote them separately rather than as one number. Test relative to
n_u+n_d+n_s above n_b ≈ 5 fm⁻³. Note that `fq0.7_B0`'s worst row here is
n_b = 3.9 fm⁻³ — inside the same 3.3-4.7 fm⁻³ band that §4b flags as the
least-converged region of that file.

### 4b. The mean field reproduces the printed chemical potentials

This is the strongest single test, because it exercises the couplings, the
rearrangement terms and the ω/ρ normalizations at once. Build, from the table's
own densities and masses,

```
J_ω = Σ_i f_i N_i n_i = 3(n_p + n_n + f_Λ n_Λ) + f_q (n_u + n_d + n_s)
J_ρ = Σ_i f_i τ_i n_i = (n_p − n_n) + f_q (n_u − n_d)
g_ω ω = Γ_ω J_ω,   g_ρ ρ = Γ_ρ J_ρ
Σ^R_b = ½ (dΓ_ω/dn_b) J_ω² + ½ (dΓ_ρ/dn_b) J_ρ²
      + Σ_{i=p,n,Λ} (dα_S/dn_b) [Σ_q N^q_i (M_q − m_q0)] nˢ_i
Σ^R_q = ⅓ B Σ_{i=p,n,Λ} nˢ_i + ⅓ Σ^R_b
```

and evaluate Eqs. (14)-(15). Then, on rows where the species is present:

| file | max abs error in µ_i over p, n, Λ, u, d, s |
|---|---|
| `Beta_fq1.0_B1.dat` | 0.0047 MeV |
| `Beta_fq0.7_B0.dat` | 0.018 MeV excluding the nine bad rows of §4b-bis (2.0 MeV including them) |
| `Beta_fq0.7_B1.dat` | 0.032 MeV (7×10⁻⁵ relative) |
| `Beta_fq1.0_B0.dat` | 0.048 MeV (7.9×10⁻⁵ relative) |
| `Beta_fq0.5_B1.dat` | 0.20 MeV overall; 0.0005 MeV below n_b = 1 fm⁻³ |

**This is the test to build the numerical gate on.** The worst case across all
five files is 0.20 MeV, and four of the five are at 0.05 MeV or below — on
potentials of 1000-2500 MeV, i.e. 8×10⁻⁵ relative at worst. That leaves no room
for a sign error, a wrong factor, or a missing rearrangement term. A gate at
0.05 MeV works on `fq1.0_B0`, `fq1.0_B1`, `fq0.7_B1` and (with the §4b-bis
exclusions) `fq0.7_B0`; `fq0.5_B1` needs 0.25 MeV, or 0.001 MeV if restricted to
n_b < 1 fm⁻³.

### 4b-bis. `Beta_fq0.7_B0.dat` has eight rows with a stale `nB` column

The `fq0.7_B0` degradation is not a convergence problem. On eight rows the
printed `nB` **does not equal the sum of that row's own species densities**,
being high by one grid step:

| printed `nB` | Σ from densities | printed `nB` | Σ from densities |
|---|---|---|---|
| 3.7 | 3.6 | 4.5 | 4.3 |
| 3.9 | 3.7 | 4.6 | 4.5 |
| 4.1 | 3.9 | 5.5 | 5.4 |
| 4.3 | 4.1 | 5.5757 | 5.4757 |

The `epa` column on these rows equals `E`/(sum of densities), not `E`/`nB` — to
10⁻¹⁰ — which identifies the sum as the true density and the `nB` column as the
stale one. Most likely the Maple loop wrote a density label from the following
iteration. Substituting the corrected density does not rescue the rows (the
mean-field residual stays ≈ 2 MeV, because the *other* columns are also from a
mixture of iterations), so they cannot be repaired, only excluded.

**Exclude these eight rows, plus the isolated row at n_b = 3.4.** That row's
columns are self-consistent — `nB` is right and `epa × nB = E` to 8×10⁻⁸ — but
it fails Eq. (23) badly: 9.6 MeV in the proton channel, 6.4 in u, 3.3 in d,
3.2 in s. The two rows below it (n_b = 3.0 and 3.1) satisfy the same identity to
8×10⁻⁸ and 4×10⁻⁹, and the two above it are stale-`nB` rows, so n_b = 3.4 reads
as the first row of a non-converged run rather than an isolated glitch. With
those nine rows out, `Beta_fq0.7_B0.dat` behaves like the others across its full
0.01-10 fm⁻³ range:

| identity | worst over the remaining 244 rows |
|---|---|
| `epa` × n_b = `E` | 2.1×10⁻⁶ relative (0.044 MeV/fm³ absolute, on the coexistence row at n_b = 5.6010) |
| n_b sum rule (Eq. 7) | 3.0×10⁻⁶ fm⁻³ |
| mean-field µ_i rebuild | 0.018 MeV |
| Eq. (23) residual | 0.49 MeV |

Detect them programmatically rather than hard-coding densities:

```python
n_sum = col["np"] + col["nn"] + col["nL"] + (col["nu"]+col["nd"]+col["ns"])/3
stale = np.abs(n_sum - col["nB"]) > 1e-5          # eight rows
```

and drop the n_b = 3.4 row explicitly with a comment.
`verify_reference_tables.py` does exactly this, in `stale_density_rows()` and
the `BAD_ROWS` table. Build the tight gates on `fq1.0_B0` and `fq0.7_B1`, which
need no exclusions at all.

### 4c. The ρ coupling carries a factor 9 — confirmed independently

The factor-9 normalization already recorded in `eos/enjl/parameters.py`
(`RHO_FACTOR`) can be read straight off the tables without any symmetry-energy
argument. On the low-density rows where only nucleons are present, the isospin
splitting of the printed potentials gives

```
g_ρ ρ = ½ [ (E_F^n − E_F^p) − (µ_n − µ_p) ]
```

and dividing by J_ρ = n_p − n_n yields **exactly 9.0000 ×** the coupling of
paper Eq. (22), at every one of those densities. The paper prints
g_ρ²/m_ρ² = 4G_S[a_TV e^{−n_b/n_TV} + b_TV]; the implementation behind these
tables uses nine times that with J_ρ = Σ_i f_i τ_i n_i and τ_p = +1, τ_n = −1.
The ω channel needs no such factor when J_ω is built with N_i = 3 for baryons.

### 4d. The vacuum constant E₀

Assembling Eq. (13) from the table columns and comparing to the printed `E`
gives a density-independent offset

```
E₀ = −4263.849 MeV/fm³     (per-file means −4263.848 … −4263.856;
                            spread within a file 5×10⁻⁶ - 1.6×10⁻⁵ relative)
```

matching `eos/enjl/uniform.vacuum_energy_density()` = −4263.8455 MeV/fm³ to
8×10⁻⁷ relative. **Both are negative**, in the sense that the assembled Eq. (13)
sum minus the printed `E` is negative — i.e. E₀ is subtracted as written in the
equation. Do not flip the sign to make a number look tidy.

That all five parameter sets, spanning three values of f_q and two of B, return
the same constant is itself a check: E₀ depends only on (Λ, m_q0, G_S, K) and
must not move with the parameters that vary between files. The Maple worksheet hard-codes an `E0` constant in its own units; the
repo's value, computed from the vacuum gap solution, is the same number. Vacuum
masses agree too: worksheet `M_u = 367.648260165719`,
`M_s = 549.479210995025`; repo `vacuum_solution()` gives 367.6482504954,
549.4792013744.

## 5. Off-grid rows are Maxwell coexistence endpoints

A few densities in each file are not on the round grid. They come in pairs at
equal pressure and equal µ_b, and they are the two edges of a first-order
transition located by Maxwell construction:

| file | n_b low | n_b high | µ_b = `munr` [MeV] | P [MeV/fm³] | transition |
|---|---|---|---|---|---|
| `Beta_fq1.0_B0.dat` | 0.643752 | 0.665923 | 1381.2899 / 1381.2898 | 186.5964 | chiral (u, d) |
| `Beta_fq1.0_B1.dat` | 0.637711 | 0.676728 | 1411.0842 / 1411.0842 | 202.1530 | chiral / quark onset |
| `Beta_fq0.7_B1.dat` | 0.448564 | 0.534224 | 1168.4748 / 1168.4747 | 69.6419 | chiral |
| `Beta_fq0.7_B0.dat` | 5.575706 | 5.600980 | 6348.7561 / 6348.7562 | 14262.3172 | deconfinement |

Both P and µ_b match across each pair to the printed digits — **provided you
read µ_b from `munr`.** The deconfinement pair in `fq0.7_B0` is exactly where
this matters: its `mun` values are 6348.76 and 5852.31, apparently a 500 MeV
mismatch, because on the high-density row the baryons have dissolved to
~10⁻⁵ fm⁻³ and `mun` is that vanishing neutron's own potential. `munr` reports
6348.7562 there and the pair matches. This is the practical demonstration that
`munr` is the quantity a Maxwell construction should equate.

`Beta_fq0.5_B1.dat` has two transitions in it, at P = 20.1226 and
1106.8075 MeV/fm³, spanning n_b = 0.322–0.459 and 1.248–1.852 fm⁻³ — but those
windows are filled with interpolated rows, so read the endpoints and ignore the
interior (§6).

Two tables also contain one step with dP/dn_b < 0 (`fq1.0_B1` at
n_b ≈ 0.677, `fq0.7_B0` at n_b ≈ 0.59–0.60): the unstable branch is *retained*
in the file rather than replaced by the plateau. A monotonicity check on the raw
file will fail; that is the file's structure, not an error.

## 6. The f_q = 0.5 file contains interpolated rows — do not fit to them

`Beta_fq0.5_B1.dat` is the parameter set of the paper's Figs. 4-6, so it is the
most interesting file physically. It is also the only one that is **not purely
solver output**, and using it naively will produce nonsense.

Of its 383 rows, only **180 are solved states** — exactly the rows on the round
0.01 fm⁻³ grid, and exactly the rows where `munr` is filled in. The other **203
rows are the two Maxwell mixed-phase plateaus, filled in by linear
interpolation**, at n_b ≈ 0.322–0.459 and 1.248–1.852 fm⁻³ (101 rows each, plus
one isolated row at n_b = 0.0857). On those rows:

* `P` is constant to 7×10⁻¹⁵ — the plateau.
* every other column (`E`, `epa`, all `n_i`, all `M_i`, all `mu_i`, `Sigmaq`) is
  **exactly linear in n_b**, residual from a straight-line fit ≤ 5×10⁻⁸.
* `munr` is blank, which is the reliable flag: `numpy.isnan(col["munr"])`
  selects precisely the interpolated rows.

Linear interpolation is the correct lever rule for the *densities* — that is
what a mixed phase does — but it is wrong for `E`, `epa` and the `mu_i`, and the
tables show it: on the plateau rows `epa` disagrees with `E/n_b` by up to
48 MeV/fm³, and the gap equation, Eq. (23) and Eq. (19) all fail there by large
margins. **Restrict every quantitative check on this file to
`~numpy.isnan(col["munr"])`.** Restricted that way it behaves like the others:
`epa × n_b = E` to 8×10⁻⁷, and the gap equation to 2×10⁻⁵ relative.

The extra columns:

* `fz` = n_p / n_b, the proton fraction (verified to 5×10⁻¹⁶ on solved rows).
  Written as `--` on the eight solved rows at n_b ≥ 1.85 where no protons
  remain.
* `Derivative Y1/Y2/Y3` are Maple plot-derived columns (d n_d/d n_b, d P/d n_b,
  and its derivative). Ignore them; `load_reference` drops them.

The plateau rows are still useful for one thing: they hand you the coexistence
pressures and endpoint densities of the two first-order transitions of this
parameter set directly (P = 20.1226 and 1106.808 MeV/fm³), which is a free check
on any Maxwell construction you build.

## 7. What the tables do *not* pin

* **Anything at T > 0.** All five files are T = 0.
* **The meson masses.** Paper 1 leaves m_σ, m_ω, m_ρ undetermined; they enter
  only through g²/m². The Thomas-Fermi paper (arXiv:2409.12489) assigns
  m_σ = 630 MeV, m_ρ = 769 MeV and m_ω = 10⁵ MeV — the last deliberately
  unphysical, chosen large to suppress density fluctuations in that paper's
  Thomas-Fermi solve. These values matter only once gradient terms are switched
  on, and 10⁵ is easily misread as 105.
* **The crust.** Paper 1 stitches a DD-LZ1 crust below the core-crust
  transition by Maxwell construction; the tables are pure core matter.
* **TOV output.** No M-R information is in these files.
