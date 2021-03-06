#!/bin/bash
set -e

# measure convergence for SIA obstacle problem

# to save results:
#   $ ./siaconv.sh >& out.siaconv

# convergence runs with F-cycle V(1,2) and tight tolerance
for JJ in 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    python3 ../obstacle.py -problem sia -down 1 -ni -nicycles 3 -J $JJ -irtol 1.0e-10 -cyclemax 1000 -l1err
done
