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

PROBLEM="-problem sia -siacase profile -exactinitial"  # out.siaasymp
#PROBLEM="-problem plap -plapcase pile -exactinitial"  # out.plapasymp
#PROBLEM="-problem sia -siacase bumpy -ni -nicycles 10"  # out.bumpy.siaasymp
#PROBLEM="-problem sia -siacase bumpy -ni -nicycles 10 -siaeps 500.0"  # out.bumpy.eps500.siaasymp

for SMOOTHER in "" "-down 1" "-down 2" "-jacobi -omega 0.6" "-jacobi -omega 0.5" "-jacobi -omega 0.4"; do
    for JJ in 4 6 8 10; do
        OPTS="$PROBLEM -monitor -irtol 1.0e-8 -J $JJ $SMOOTHER"
        echo "case: " $OPTS
        python3 ../obstacle.py $OPTS
    done
done
