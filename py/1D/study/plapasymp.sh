#!/bin/bash
set -e

# for p-Laplacian pile problem, measure asymptotic rates around exact solution
# for V-cycles with various smoothers

# to save results:
#   $ ./plapasymp.sh >& out.plapasymp

# quick view results:
#   $ grep -B 4 cycles out.plapasymp |less

# OTHER PARAMETERS TO CONSIDER:
#   -newtonits 1|2|3

for SMOOTHER in "" "-down 1" "-down 2" "-jacobi -omega 0.6" "-jacobi -omega 0.5" "-jacobi -omega 0.4"; do
    for JJ in 6 8 10; do
        OPTS="-problem plap -plapcase pile -exactinitial -monitor -irtol 1.0e-8 -J $JJ $SMOOTHER"
        echo "case: " $OPTS
        python3 ../obstacle.py $OPTS
    done
done
