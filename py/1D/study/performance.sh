#!/bin/bash
set -e

# performance (WU and timing) for the classical obstacle problem with:
#      V(1,0), V(0,1), V(1,1), ni+V(0,1)
#   on default problem icelike using default rtol=1e-3

# to save results:
#   $ ./performance.sh >& out.performance

for OPTS in "-down 1 -up 0" "-down 0 -up 1" "-down 1 -up 1" "-ni -down 0 -up 1"; do
    echo "*** solver = '$OPTS' ****"
    for JJ in 6 7 8 9 10 11 12 13 14 15; do
        /usr/bin/time -f "real %e" python3 ../obstacle.py -poissoncase icelike -jfine $JJ $OPTS
    done
done

