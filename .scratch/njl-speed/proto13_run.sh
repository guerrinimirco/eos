#!/bin/sh
# PROTOTYPE (wayfinder ticket 13) -- one load window, control bracketed.
cd /Users/mircoguerrini/Desktop/Research/Python_codes/eos
for v in V0 VD VL; do
  python3 .scratch/njl-speed/proto13_variant.py "$v" || exit 1
done
cp .scratch/njl-speed/proto13_V0.json .scratch/njl-speed/proto13_V0first.json
python3 .scratch/njl-speed/proto13_variant.py V0 || exit 1
echo "--- control bracket: V0first vs V0 (the window's drift) ---"
python3 -c "
import json
a = json.load(open('.scratch/njl-speed/proto13_V0first.json'))
b = json.load(open('.scratch/njl-speed/proto13_V0.json'))
print(f\"  V0 first {a['wall_s']:.1f}s / cpu {a['cpu_s']:.1f}s\")
print(f\"  V0 last  {b['wall_s']:.1f}s / cpu {b['cpu_s']:.1f}s\")
print(f\"  drift {b['wall_s'] / a['wall_s']:.2f}x wall, {b['cpu_s'] / a['cpu_s']:.2f}x cpu\")"
