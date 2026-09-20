#!/bin/sh
# Ticket 13, closing window: VX re-taken against the fixed harness (rung C
# separated from the cold retry), then the cross-mode correctness probe.
cd /Users/mircoguerrini/Desktop/Research/Python_codes/eos
python3 .scratch/njl-speed/proto13_variant.py VX || exit 1
echo "--- gate ---"
python3 .scratch/njl-speed/proto13_gate.py VX || exit 1
echo "--- cross-mode probe: beta-eq and fixed_YC, T = 0 and 30 MeV ---"
python3 .scratch/njl-speed/proto13_modes.py || exit 1
