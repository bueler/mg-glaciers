#!/bin/bash
set -e

# measure performance for the classical obstacle problem
# to save results:
#   $ ./performance.sh >& out.performance

FIXME
1 add unconstrained and demo O(1) V(1,1) cycles and discretization in one F-cycle
2 update runs to use ?:   V(1,0), V(0,1), V(1,1), ni+V(0,1), nicycles=2+V(0,1)

# performance (WU and timing) runs with:  V(1,0), V(2,0), ni+V(1,0), nicycles=2+V(1,0),
#   all on default problem (poisson, icelike)
echo "PERFORMANCE"
for OPTS in "" "-down 2" "-ni" "-ni -nicycles 2"; do
    for OBS in "" "-random"; do
        echo "*** solver = '$OPTS', obstacle = '$OBS' ****"
        for JJ in 6 7 8 9 10 11 12 13 14 15; do
            /usr/bin/time -f "real %e" python3 ../obstacle.py $OBS -jfine $JJ $OPTS
        done
    done
done

