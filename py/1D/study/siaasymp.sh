#!/bin/bash
set -e

# for SIA problem, measure asymptotic rates around exact solution
# for V-cycles with various smoothers

# to save results:
#   $ ./siaasymp.sh >& out.siaasymp

# quick view results:
#   $ grep -B 4 cycles out.siaasymp |less

# OTHER PARAMETERS TO CONSIDER:
#   -newtonits 1|2|3

PROBLEM=sia  # change to plap to compare

for SMOOTHER in "" "-down 1" "-down 2" "-jacobi -omega 0.6" "-jacobi -omega 0.5" "-jacobi -omega 0.4"; do
    for JJ in 4 6 8 10; do
        OPTS="-problem $PROBLEM -exactinitial -monitor -irtol 1.0e-8 -J $JJ $SMOOTHER"
        echo "case: " $OPTS
        python3 ../obstacle.py $OPTS
    done
done
