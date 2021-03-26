#!/bin/bash
set -e

# compare the error norm from one F-cycle (-ni), or one F-cycle using
#     2 V-cycles at each level (-ni -nicycles 2), with the goal of showing
#     error norms relative to discretization error
# uses "traditional" problem; performance is more inconsistent on icelike
# run with "pde1" to see that F-cycles reach discretization
#     (within factor of 2) when no obstacle

# to save results:
#   $ ./perfni.sh >& out.perfni

for OPTS in "-irtol 1.0e-7 -down 1 -up 1" "-ni -cyclemax 1 -down 1 -up 1" "-ni -nicycles 2 -cyclemax 1 -down 1 -up 1"; do
    echo "*** solver = '$OPTS' ****"
    for JJ in 2 3 4 5 6 7 8 9 10 11 12; do
      python3 ../obstacle.py -poissoncase traditional -J $JJ $OPTS
      #python3 ../obstacle.py -poissoncase pde1 -J $JJ $OPTS
    done
done
