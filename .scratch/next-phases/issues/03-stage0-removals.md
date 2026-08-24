# What goes from notebooks/, and what is lost with it

Type: grilling
Status: resolved
Blocked by: 02
Parent: ../map.md

## Question

Stage 0. Produce the table before removing anything: path, size, last commit that
touched it, whether it holds outputs or results not reproducible from the
replacement notebooks, and a one-line recommendation each.

Candidates: `CCDM_usage`, `DD2_usage`, `DD2vMIT_general1oPT`, `DID_usage`,
`ENJL_usage`, `NJL_usage` (each `.ipynb` + jupytext `.py`),
`mass distribution.ipynb`, `notebooks/eos_tables_DD2vMIT/`, and
`eos/dd2/notebook_api.py`.

**Keep** `ZLvMIT_hybrid.ipynb` and `zlvmit_test.ipynb`.

`DD2vMIT_general1oPT.ipynb` (3.1 MB) and `mass distribution.ipynb` (194 kB) carry
stored outputs — say explicitly what would be lost and whether the replacement
notebooks reproduce it.

**`test/dd2/test_notebook_api.py` must go with it** — it does
`from eos.dd2 import notebook_api as api` at line 13, so deleting the module
alone breaks the suite ([ticket 28](28-photons-silent-ignore.md)).

`eos/dd2/notebook_api.py` is forbidden by §11 and is imported only by
`notebooks/DD2_usage.{py,ipynb}`, both on the removal list — so nothing survives
it. `docs/DEFERRED.md:299` already records its deletion as outstanding.

Resolved when the table is in the answer and the user has named what goes.
Deletion itself happens under this ticket (both gates are lifted for this map).

## Answer

**Fifteen files removed, one 46 MB folder deliberately held, and one defect the
ticket did not know about.**

### The table, as asked

| Path | Size | In git? | Last commit | Stored outputs | Ruling |
|---|---|---|---|---|---|
| `notebooks/CCDM_usage.{ipynb,py}` | 28K + 16K | tracked | `971f6ad` | none (0/13 cells) | removed |
| `notebooks/DD2_usage.{ipynb,py}` | 24K + 20K | tracked | `82e587a` | none (0/16) | removed |
| `notebooks/DID_usage.{ipynb,py}` | 16K + 12K | tracked | `a086177` | none (0/11) | removed |
| `notebooks/NJL_usage.{ipynb,py}` | 20K + 16K | tracked | `9186be5` | none (0/10) | removed |
| `notebooks/ENJL_usage.{ipynb,py}` | 40K + 28K | tracked | `4e89355` | none (0/10) | removed |
| `notebooks/DD2vMIT_general1oPT.{ipynb,py}` | 3.0M + 144K | tracked | `d9f8eec` | 27/30 cells, 14 figures | removed |
| `notebooks/mass distribution.ipynb` | 192K | tracked | `d9f8eec` | 12/12 cells, 2 figures | removed |
| `eos/dd2/notebook_api.py` | 28K | tracked | `68b3632` | -- | removed |
| `test/dd2/test_notebook_api.py` | 3.0K | **gitignored** (`/test/`) | -- | -- | removed |
| `notebooks/eos_tables_DD2vMIT/` | **46M** | **gitignored, 0 tracked** | -- | 32 tables + 42 figures | **HELD** |
| `notebooks/ZLvMIT_hybrid.ipynb` | 112K | tracked, KEEP | `d9f8eec` | 0 outputs | kept -- **but corrupt** |
| `notebooks/zlvmit_test.ipynb` | 688K | **gitignored**, KEEP | -- | 8/11 cells, 17 figures | kept |

**What is lost.** Nothing that was reachable. The five `_usage` pairs carried
**zero** stored outputs -- every one had been cleared. `DD2vMIT_general1oPT.ipynb`
and `mass distribution.ipynb` did carry outputs (27 and 12 cells, 16 inline
figures between them), but those outputs have been **unreadable since
`d9f8eec`** (below) and remain reachable at `d9f8eec^` in git history. Every
`.py` jupytext pair parsed as valid Python, so no code was `.ipynb`-only.

### The defect the ticket did not know about

`d9f8eec` mechanically rewrote `eos.tov` -> `eos.astro.tov` across `notebooks/`
and **broke the JSON of all three `.ipynb` files it touched** -- it replaced one
JSON source line with text split across real newlines without re-escaping,
leaving an unterminated string (plus, in two files, a dropped comma). `d9f8eec^`
is valid JSON in all three; `d9f8eec` is not, and `nbformat` refuses all three.

Two were on this ticket's removal list, so their breakage no longer matters. The
third, `ZLvMIT_hybrid.ipynb`, is on the KEEP list -- kept for published results
nobody can currently open. Split to [ticket 41](41-corrupt-notebooks.md), which
also carries the Stage 7 observation that nothing in the repository checks an
`.ipynb` is loadable, so this survived five days and four commits undetected.

### `notebooks/eos_tables_DD2vMIT/` is held, not removed

This is the one item where the map's "both gates are lifted, git history is the
undo" **does not hold**: `.gitignore:46` excludes the folder and **zero files in
it are tracked**, so deleting it is permanent. It holds the 32 computed
mixed-phase tables (`.csv` + `.h5`) and the 42 published figures
(`fig01`--`fig11`, png + pdf) that `DD2vMIT_general1oPT` produced.

[Ticket 05](05-notebook-coverage.md) has not yet ruled whether `mixed` gets a
notebook at all, so no replacement is guaranteed to regenerate these. **Held
until 05 rules and a replacement has actually regenerated equivalents** --
recorded on the map's Not-yet-specified section, not silently skipped.

### `notebook_api.py` was a three-part removal

Deleting the module alone breaks the suite; deleting the test too leaves a stale
exemption. All three went together:

- `eos/dd2/notebook_api.py` -- its only importer was `notebooks/DD2_usage`,
  removed in the same change
- `test/dd2/test_notebook_api.py` -- `from eos.dd2 import notebook_api` at line 13
- `test/test_imports.py` `_EXEMPT_FILES` -- the entry naming
  `dd2/notebook_api.py` as a recorded violation that "dies with the file". The
  dict is now empty; no file is exempt.

`docs/DEFERRED.md` carried **three** references, all now closed: the dd2 layout
entry that called this "the last one outstanding", the surviving `dd2 -> sfho`
import edge (`notebook_api` reaching for `eos.sfho.table`), and the
`notebook_api imports astro` line. No model-to-model edge remains.

### Sequencing accepted

`notebooks/` now holds only the two `zlvmit` files until tickets 12--19 land the
three grouped notebooks. That window is the point of Stage 0: it stops 12--19
copying the old shape, and git is the undo for every tracked file removed.
