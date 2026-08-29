#!/bin/sh
# Certify the certifier.
#
# run_clean_suite.sh exists so a suite count cannot be believed without
# evidence the tree held still. It shipped with three defects in twenty
# minutes, TWO OF THEM FAILING OPEN -- a guard that reports the all-clear when
# it cannot run is worse than no guard, because it is indistinguishable from
# the good answer. These cases are the ones that would have caught them.
#
# Case 3 is the important one and is a regression test for a specific trap:
# `find -newermt '-2 minutes'` under the interactive shell's bfs shim errors to
# stderr AND exits 0, so it is correct exactly when nothing is fresh and blind
# exactly when something is. Any check of it against a quiet tree passes.
#
# Usage: sh test/test_run_clean_suite.sh
cd "$(dirname "$0")/.."
# The script under test. A seam, so the mutation check below is reproducible
# by a third party rather than a thing its author did once and reported:
#
#     cp test/run_clean_suite.sh test/.mutant.sh   # then break it
#     SCRIPT=test/.mutant.sh sh test/test_run_clean_suite.sh   # mutation arm
#     sh test/test_run_clean_suite.sh                          # control arm
#
# THE MUTANT MUST LIVE INSIDE test/. run_clean_suite.sh does `cd $(dirname
# $0)/..` to find the repo, so a copy in /tmp resolves that to `/` and the run
# dies on `mkdir: test: Read-only file system` -- a failure that looks like the
# mutation and is not. Relocating the HARNESS instead fails the same way and
# louder: a control arm of the UNMUTATED script through a relocated harness
# gives 4 passed, 7 failed, i.e. it measures its own relocation. Both arms,
# always: without the control, a broken rig reads as a finding.
SCRIPT=${SCRIPT:-test/run_clean_suite.sh}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# Never the real store. A rig that writes there leaves certificates naming the
# real HEAD and the real interpreter over a stub count -- and a MUTANT arm
# leaves one stamped CLEAN.
export CERT_DIR="$TMP/certs"
pass=0; fail=0
check() {  # check <name> <expected-exit> <expected-substring> <actual-exit> <output>
    if [ "$2" = "$4" ] && echo "$5" | grep -q "$3"; then
        echo "  ok    $1"; pass=$((pass + 1))
    else
        echo "  FAIL  $1 (wanted exit $2 containing '$3', got exit $4)"
        echo "$5" | sed 's/^/          /'
        fail=$((fail + 1))
    fi
}

# A fake tree, old enough to pass the 120s guard.
mkdir -p "$TMP/tree"
echo "x = 1" > "$TMP/tree/mod.py"
touch -t "$(date -v-10M +%Y%m%d%H%M)" "$TMP/tree/mod.py"

echo "case 1: a quiet tree certifies CLEAN"
out=$(TREE="$TMP/tree" SUITE="echo '1 passed'" sh $SCRIPT 2>&1); rc=$?
check "verdict is CLEAN" 0 "CLEAN: the count above is a measurement" "$rc" "$out"
cert=$(echo "$out" | sed -n 's/^certificate: //p')
check "certificate written" 0 "." "$?" "$cert"
check "certificate records CLEAN" 0 "verdict     CLEAN" 0 "$(cat "$cert" 2>&1)"
check "certificate names the interpreter" 0 "interpreter CPython" 0 "$(cat "$cert" 2>&1)"
check "certificate names HEAD" 0 "HEAD        [0-9a-f]" 0 "$(cat "$cert" 2>&1)"
rm -f "$cert"

echo "case 2: a tree written DURING the run certifies DISCARD"
out=$(TREE="$TMP/tree" SUITE="echo 'y = 2' >> $TMP/tree/mod.py; echo '1 passed'" \
      sh $SCRIPT 2>&1); rc=$?
check "exits non-zero" 1 "DISCARD" "$rc" "$out"
cert=$(echo "$out" | sed -n 's/^certificate: //p')
check "certificate records DISCARD" 0 "verdict     DISCARD" 0 "$(cat "$cert" 2>&1)"
rm -f "$cert"

echo "case 3: a freshly-written tree is REFUSED (the fail-open regression)"
touch "$TMP/tree/mod.py"
out=$(TREE="$TMP/tree" SUITE="echo 'ran anyway'" sh $SCRIPT 2>&1); rc=$?
check "refuses, and does not run the suite" 2 "the tree is live" "$rc" "$out"
check "the suite really did not run" 2 "^" "$rc" \
      "$(echo "$out" | grep -c 'ran anyway' | grep '^0$' && echo ok)"

echo "case 4: --check reports without running"
touch -t "$(date -v-10M +%Y%m%d%H%M)" "$TMP/tree/mod.py"
out=$(TREE="$TMP/tree" SUITE="echo 'ran anyway'" sh $SCRIPT --check 2>&1); rc=$?
check "reports the window" 0 "WINDOW OPEN" "$rc" "$out"
check "started nothing" 0 "^" "$rc" \
      "$(echo "$out" | grep -c 'ran anyway' | grep '^0$' && echo ok)"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
