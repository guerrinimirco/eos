#!/bin/sh
# A full-suite run that certifies its own window, and writes the certificate.
#
# The map's record is that a run spanning ANY source edit is not a measurement
# -- it has cost four discarded runs (two on 2026-08-25, one on 2026-08-29,
# plus the concurrent-suite flake that falsely reds test_baseline[ccdm]).
# Several sessions share this checkout, so "I have nothing in flight" is a
# claim about one session and not about the tree. This checks the tree.
#
# What the four incidents have in common is not that a run was invalidated but
# that the invalidation was discovered AFTER the number was believed. So the
# verdict is written beside the count, to a file a resolution can cite: a
# citation beats a session's word that a window existed.
#
# The certificate names the INTERPRETER and the HEAD SHA as well as the count,
# because a failure count without them names nothing -- this machine has two
# Python stacks that disagree, and HEAD moves under you while others commit.
#
# Usage: sh test/run_clean_suite.sh [--check] [pytest args...]
#
# THREE SEAMS EXIST FOR test_run_clean_suite.sh, and for nothing else: TREE is
# the directory fingerprinted, SUITE is the command run between the two
# fingerprints, CERT_DIR is where the certificate is written. The certifier has
# to be certified -- it shipped with three defects in its first twenty minutes,
# two of them failing OPEN -- and it cannot be tested against the real tree
# (dirtying eos/ mid-run would fire every other session's guard) or the real
# suite (twenty minutes per case).
#
# CERT_DIR is a seam for a measured reason, not for symmetry. Without it the
# rig writes into the real certificate store, and the first file that store
# ever held was written by the fail-open MUTANT: verdict CLEAN, real HEAD, real
# interpreter, and the string `ran anyway` where the pytest count belongs. An
# evidence store its own test can write to is not evidence.
set -e
cd "$(dirname "$0")/.."

TREE=${TREE:-eos}
SUITE=${SUITE:-"PYTHONPATH=. python3 -m pytest test/ -q"}

fingerprint() { find "$TREE" -name '*.py' -exec shasum {} + | sort | shasum | cut -d' ' -f1; }

# Match the python process itself, never the shell wrapping it. `pgrep -f
# pytest` also matches every session's waiter loop -- their command lines
# contain the word -- so it reports a run that is not running and the window
# never opens. Filtering on comm being python is what separates the two.
running_suites() { ps -Ao pid=,comm=,args= | awk '$2 ~ /[Pp]ython/ && /pytest/'; }

if [ -n "$(running_suites)" ]; then
    echo "REFUSED: another pytest is running -- two concurrent suites falsely"
    echo "red test_baseline[ccdm] in both. Wait for it."
    running_suites
    exit 2
fi
# Age of the most recently written eos/*.py, in seconds. NOT `-newermt
# '-2 minutes'`: real BSD find accepts that, but an interactive shell here
# shims `find` to bfs, which REJECTS the relative timestamp -- and the error
# goes to stderr while command substitution captures only stdout, so the guard
# returns empty and reads as "the tree is quiet". A check that fails open is
# worse than no check. This form works under both.
newest_write() {
    newest=$(find "$TREE" -name '*.py' -exec stat -f '%m' {} + | sort -rn | head -1)
    echo $(( $(date +%s) - newest ))
}

if [ "$(newest_write)" -lt 120 ]; then
    echo "REFUSED: $TREE/*.py written $(newest_write)s ago; the tree is live."
    # sort on the epoch, then format: sorting the FORMATTED time is a
    # lexical sort, which ranks yesterday's 23:54 above today's 01:28.
    find "$TREE" -name '*.py' -exec stat -f '%m %N' {} + | sort -rn | head -5 \
        | while read -r m f; do echo "  $(date -r "$m" +%H:%M:%S) $f"; done
    exit 2
fi

# Both guards passed. `--check` stops here: it lets a session poll the window
# without spending 20 minutes discovering it was open, and it is the only way
# to ask the question without also answering it.
if [ "$1" = "--check" ]; then
    echo "WINDOW OPEN: no pytest, newest eos/*.py $(newest_write)s old."
    echo "Note: an open window is not a quiet ticket. Check whether anything"
    echo "under eos/ is mid-edit -- a two-minute pause is a pause, not an end."
    git status --short -- 'eos/*.py' | sed 's/^/  /'
    exit 0
fi

CERT_DIR=${CERT_DIR:-test/suite_certificates}
mkdir -p "$CERT_DIR"
# Stamp the open time ONCE, here, and reuse it for both the filename and the
# certificate's `opened` line. Calling `date` again inside the heredoc below
# stamps the moment the certificate is WRITTEN, which is twenty minutes after
# the window opened -- the first real certificate carried a close time under
# an `opened` label, disagreeing with its own filename.
OPENED=$(date +%Y-%m-%dT%H:%M:%S)
CERT="$CERT_DIR/$(echo "$OPENED" | tr -d ':-').txt"
PY=$(PYTHONPATH=. python3 -c 'import sys,numpy,scipy;print(f"CPython {sys.version.split()[0]}, numpy {numpy.__version__}, scipy {scipy.__version__}")')
SHA=$(git rev-parse --short HEAD)

before=$(fingerprint)
echo "window opened  $(date +%H:%M:%S)  eos/ $before"
LOG=$(eval "$SUITE" "$@" 2>&1 | tail -25)
after=$(fingerprint)
echo "$LOG"
echo "window closed  $(date +%H:%M:%S)  eos/ $after"

{
    echo "suite certificate"
    echo "  opened      $OPENED"
    echo "  closed      $(date +%Y-%m-%dT%H:%M:%S)"
    echo "  HEAD        $SHA"
    echo "  interpreter $PY"
    echo "  eos/ before $before"
    echo "  eos/ after  $after"
    if [ "$before" = "$after" ]; then
        echo "  verdict     CLEAN -- no eos/*.py changed during the run"
    else
        echo "  verdict     DISCARD -- eos/ changed mid-run; this is not a measurement"
    fi
    echo
    echo "$LOG"
} > "$CERT"

echo "certificate: $CERT"
if [ "$before" = "$after" ]; then
    echo "CLEAN: the count above is a measurement."
else
    echo "DISCARD: eos/ changed mid-run. The count above is not a measurement."
    exit 1
fi
