#!/bin/bash
set -e
set +x

# run to generate convergence results shown in ../../doc/fas.pdf:
#   $ ./converge.sh &> converge.txt

# FIXME convergence in -mms case by brutal NGS sweeps:
#$ for JJ in 1 2 3 4 5 6; do ./fas1.py -downsweeps 10000 -ngsonly -mms -j $JJ; done

for LEV in 1 2 3 4 5 6 7 8 9 10 11 12; do
    ../fas1.py -mms -cycles 8 -j $LEV
done

