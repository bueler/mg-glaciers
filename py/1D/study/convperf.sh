#!/bin/bash
set -e

# measure convergence and performance for the classical obstacle problem
# to save results:
#   $ ./convperf.sh >& out.convperf

# convergence runs with V(1,0) and tight tolerance
echo "CONVERGENCE"
for CASE in icelike traditional; do
    echo "*** case $CASE ****"
    for JJ in 2 3 4 5 6 7 8 9 10 11 12 13 14; do
        python3 ../obstacle.py -poissoncase $CASE -jfine $JJ -irtol 1.0e-7 -cyclemax 1000
    done
done
echo

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

