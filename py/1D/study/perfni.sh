#!/bin/bash
set -e

# measure performance (WU) to get discretization error
# uses exact solution (convergence) cases from convperf.sh:
#   get error |u-uexact|_2 from paper/genfig/convergence.txt and *double*
#   that value to get target error value (i.e. -errtol) for that case

# to save results:
#   $ ./perfni.sh >& out.perfni

PROB=icelike
echo "*** problem $PROB ****"
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 2 -errtol 9.79940e-02
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 3 -errtol 5.35860e-02
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 4 -errtol 9.68700e-03
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 5 -errtol 1.96754e-02
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 6 -errtol 3.23440e-03
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 7 -errtol 4.51220e-03
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 8 -errtol 7.54840e-04
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 9 -errtol 1.15324e-03
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 10 -errtol 1.92032e-04
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 11 -errtol 2.86740e-04
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 12 -errtol 4.78000e-05
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 13 -errtol 7.17820e-05
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 14 -errtol 1.19630e-05
PROB=parabola
echo "*** problem $PROB ****"
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 2 -errtol 1.16462e-02
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 3 -errtol 4.09700e-03
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 4 -errtol 8.71000e-04
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 5 -errtol 2.36520e-04
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 6 -errtol 5.67680e-05
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 7 -errtol 1.44888e-05
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 8 -errtol 3.58820e-06
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 9 -errtol 9.03400e-07
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 10 -errtol 2.26160e-07
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 11 -errtol 5.76640e-08
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 12 -errtol 1.46498e-08
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 13 -errtol 4.11040e-09
python3 ../obstacle.py -ni -nicycles 2 -problem $PROB -jfine 14 -errtol 1.21676e-09
