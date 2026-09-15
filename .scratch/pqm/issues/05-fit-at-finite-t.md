# Fit at T > 0: the thermal terms, Delta(T), and s as a fitted row

Type: prototype
Status: open
Blocked by: 01, 02, 04
Parent: ../map.md

## Question

With ticket 04's data in hand, turn the T = 0 model into a finite-T one. Three
pieces, and only the third is genuinely open.

### 1. The thermal terms are already written, just not used

`basis()` carries them (`notebooks/csc_bag_map.py:168-169`):

```python
"aT2": scale * sum(m ** 2 for m in mu_f) * T ** 2,
"aT4": scale * T ** 4,
```

and they are absent from `TERMS` (`:709-710`). Putting them in is one line.

**The form is exact, and that is measured rather than assumed.**
`mu^4/4pi^2 + mu^2 T^2/2 + 7 pi^2 T^4/60` IS the antiparticle-inclusive
massless Fermi gas -- verified against `fermi_thermo` to 1.000000 at T = 0, 30
and 60 MeV. So `a4`, `aT2` and `aT4` are one gas written in three linear
pieces, and at T = 0 the last two vanish identically, which is why the T = 0
fit was not wrong to omit them. What is NOT known is whether the NJL medium's
thermal behaviour sits on those two coefficients -- that is what the data
answers.

The three massive strange gases need no change at all: `fermi_thermo(mu_s, T,
m)` already carries T (`csc_bag_map.py:172`), and after ticket 01 its `n` and
`s` supply the derivatives.

### 2. Delta(T) is the BCS closure, and it belongs to the model

The gap is currently `Delta_star (mu_B/mu_star)^sigma`
(`csc_bag_map.py:157`, `340-348`) -- a power law in `mu_B` with no temperature
in it at all. The docs already name the extension
(`docs/csc_bag_mapping.md:187-189`):

    Delta(mu_B, T) = Delta_star (mu_B/mu_star)^sigma * sqrt(1 - (T/T_c)^2)
    T_c            = tc_coeff * Delta(mu_B, 0)

`eos/alphabag/thermodynamics.py:434-448` implements exactly this shape
(`cfl_gap`), with `TC_COEFF = 0.57 * 2**(1/3)` (`alphabag/parameters.py:42`) --
the BCS relation with the CFL multiplicity factor. **Read it, do not import
it**: a model may not import another model (CLAUDE.md section 1), and this is
three lines of textbook physics, not shared machinery. `tc_coeff` becomes a
`pqm` parameter with the alphabag value as its default, and `pqm.tex` states
the relation with its citation. Above `T_c` the gap is zero and the pattern is
unpaired -- which is a BRANCH TERMINUS in temperature and ticket 08 will have
to know about it.

**Check it against the data rather than assuming it.** The NJL gaps come back
on every row (`th.fields["Delta_1..3"]`), so the sqrt law is testable directly:
fit `(Delta_star, sigma)` on the T = 0 line as now, then measure how far the
NJL `Delta(T)/Delta(0)` departs from `sqrt(1 - (T/T_c)^2)` on the T grid. If it
departs, `tc_coeff` is the one free knob and a second exponent is the next
cheapest fix. Either way this is a MEASUREMENT, and its result belongs in
`pqm.tex`.

### 3. `s` becomes a fitted row -- and this is the open part

The residual today carries `P` and the three charge densities
(`csc_bag_map.py:431-445`), four rows per grid point, each scaled by the grid
maximum. At T > 0 there is a fifth: `s = -dOmega/dT`, which the adapter returns
(`PhaseThermo.s`, `eos/general/state.py:122`) and which after ticket 01 the
model produces in closed form.

`s` is the sharpest test of the thermal terms for the same reason `n` is
sharper than `P`: it is a derivative, so a constant offset that `P` absorbs
shows up in it. It is also the quantity a supernova table is actually sensitive
to, and the fourth acceptance benchmark asks for it within 10%.

Open questions the prototype must answer, not the plan:

- **Does `s` need its own scale row, or does `max|s|` over the grid suffice?**
  The other rows use the grid maximum because `P` crosses zero near the
  surface; `s` does too, at T = 0, where it vanishes identically on every
  point. A grid with 1/6 of its points at exactly `s = 0` may need those rows
  dropped rather than scaled.
- **Does the strange threshold structure survive at T > 0?** The three
  fixed-mass gases were introduced because `n_S` is the worst residual row --
  the NJL constituent `M_s` runs ~460 to ~200 MeV across these branches and the
  s sea is switching on, which is a THRESHOLD and not a power (measured: giving
  s its own `mu^4` coefficient moved `n_S` 9.5% -> 9.2%, while the three gases
  moved it to 7.6%). Temperature smears a threshold. Either the gases still
  help and carry their own T dependence for free, or the smearing means fewer
  of them are needed -- worth one term-set comparison, in the style of
  `TERM_SETS` at `csc_bag_map.py:684-707`.
- **Do `aT2`/`aT4` peg?** If they run to the free-gas values the thermal
  sector is trivially right and the ticket is short. If they do not, the NJL
  medium's entropy is not a massless gas's and the deviation is physics worth
  recording.

## Gate

- `aT2`, `aT4` in `TERMS`; `Delta(T)` implemented with its `tc_coeff`
  parameter; `s` a fitted residual row.
- **The T = 0-100 MeV benchmark slice at target**: median `|dP|/max|P|` <= 3%
  over `Y_C` = 0.01-0.5 at `mu_S` = 0, and **`s` within 10%** of `eos.njl`.
- **T = 0 results unchanged** against ticket 03's recorded appendix, to within
  the refit noise. The thermal terms vanish at T = 0, so a T = 0 regression
  here means something else moved and must be explained.
- A measured statement of how far the NJL `Delta(T)/Delta(0)` departs from the
  BCS sqrt law, and the `tc_coeff` the data prefers.
- A term-set comparison at T > 0 in the existing style, so the massive-gas
  count is a decision with a number under it rather than an inheritance.
- Whatever is learned about `s`'s scale row written down -- the next model to
  fit an entropy will hit the same T = 0 degeneracy.
