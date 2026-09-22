#!/bin/sh
# Ticket 19: the reachable suites, ONE AT A TIME, python.org 3.14, eos.__file__
# printed by the process that runs pytest; eos/*.py fingerprinted either side.
cd /Users/mircoguerrini/Desktop/Research/Python_codes/eos
P=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
D=.scratch/njl-speed
fp() { find eos -name '*.py' -type f | sort | xargs shasum | shasum; }
echo "fingerprint before: $(fp)  HEAD $(git rev-parse --short HEAD)"
for target in test/njl test/baseline test/test_imports.py test/test_nonconvergence_return.py test/general/test_zero_pressure.py test/mixed; do
  name=$(echo $target | tr '/.' '__')
  $P -c "import sys, numpy, scipy, eos, pytest; print(sys.version.split()[0], numpy.__version__, scipy.__version__, eos.__file__, flush=True); sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '$target']))" > $D/t19_suite_$name.log 2>&1
  echo "$target exit $? :: $(tail -1 $D/t19_suite_$name.log)"
done
echo "fingerprint after:  $(fp)"
