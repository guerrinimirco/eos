# What goes from notebooks/, and what is lost with it

Type: grilling
Status: open
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
