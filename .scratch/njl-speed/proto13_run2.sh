#!/bin/sh
cd /Users/mircoguerrini/Desktop/Research/Python_codes/eos
python3 .scratch/njl-speed/proto13_variant.py VX || exit 1
python3 .scratch/njl-speed/proto13_variant.py V0 || exit 1
cp .scratch/njl-speed/proto13_V0.json .scratch/njl-speed/proto13_V0second.json
echo "--- gates (baseline = the V0 run that just closed the window) ---"
for v in VX VD VL; do python3 .scratch/njl-speed/proto13_gate.py "$v"; echo; done
