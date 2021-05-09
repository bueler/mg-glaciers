#!/bin/bash
set -e

# measure convergence for three available exact solutions for the
#   classical obstacle problem
# to save results:
#   $ ./convergence.sh >& out.convergence

# convergence runs with V(0,2) and tight tolerance
for CASE in icelike traditional pde2; do
    echo "*** case $CASE ****"
    for JJ in 2 3 4 5 6 7 8 9 10 11 12 13 14; do
        python3 ../obstacle.py -poissoncase $CASE -J $JJ -irtol 1.0e-8 -cyclemax 1000
    done
done
echo


