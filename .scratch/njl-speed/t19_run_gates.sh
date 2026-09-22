#!/bin/sh
# Ticket 19's gate runs (not timings), four chains in parallel.
cd /Users/mircoguerrini/Desktop/Research/Python_codes/eos
D=.scratch/njl-speed
P=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
export PYTHONPATH=.
chainA() {
  for pat in default 2SC CFL; do for arm in control landed; do
    $P $D/t19.py table $arm $pat fast $D/t19_table_${pat}_fast_${arm}.json
  done; done
  for arm in control landed; do $P $D/t19.py gapless $arm fast $D/t19_gapless_fast_${arm}.json; done
  for arm in control landed; do for be in fast reference; do
    $P $D/t19.py finiteT $arm $be $D/t19_finiteT_${be}_${arm}.json; done; done
}
chainB() {
  for pat in 2SC CFL; do for arm in control landed; do
    $P $D/t19.py table $arm $pat reference $D/t19_table_${pat}_reference_${arm}.json
  done; done
}
chainC() {
  for arm in control landed; do
    $P $D/t19.py table $arm default reference $D/t19_table_default_reference_${arm}.json
  done
}
chainD() {
  for arm in control landed; do $P $D/t19.py gapless $arm reference $D/t19_gapless_reference_${arm}.json; done
}
chainA > $D/t19_chainA.log 2>&1 &
chainB > $D/t19_chainB.log 2>&1 &
chainC > $D/t19_chainC.log 2>&1 &
chainD > $D/t19_chainD.log 2>&1 &
wait
echo ALL DONE
