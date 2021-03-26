#!/bin/bash
set -e

# performance (WU and timing) for V-cycles of the classical obstacle problem
# four cases:
#      V(1,0), V(0,1), V(1,1), V(1,1)-Jacobi0.67
# on default problem icelike using default rtol=1e-3

# to save results:
#   $ ./performance.sh >& out.performance

for OPTS in "-down 1 -up 0" "-down 0 -up 1" "-down 1 -up 1" "-down 1 -up 1 -jacobi -omega 0.67"; do
    echo "*** solver = '$OPTS' ****"
    for JJ in 6 7 8 9 10 11 12 13 14 15; do
    #for JJ in 6 7 8 9 10 11 12; do
        /usr/bin/time -f "real %e" python3 ../obstacle.py -poissoncase icelike -J $JJ $OPTS
    done
done

