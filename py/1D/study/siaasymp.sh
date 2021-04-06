#!/bin/bash
set -e

# measure asymptotic rates around exact solution for V-cycles with various smoothers
# to save results:
#   $ ./siaasymp.sh >& out.siaasymp

# quick view results:
#   $ grep -B 2 case out.siaasymp |less

# OTHER PARAMETERS TO CONSIDER:
#   -newtonits 1|2|3
#   -jacobi -omega 0.2|0.3|0.4|0.5|0.6

#for SMOOTHER in "" "-jacobi -omega 0.5"; do
for SMOOTHER in ""; do
    for CYCLE in "" "-down 1" "-symmetric" "-symmetric -down 1" "-symmetric -down 1 -up 2"; do
        for JJ in 6 8 10 12; do
            OPTS="-problem sia -exactinitial -monitor -irtol 1.0e-8 -J $JJ $SMOOTHER $CYCLE"
            echo "case: " $OPTS
            python3 ../obstacle.py $OPTS
        done
    done
done
