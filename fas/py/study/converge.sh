#!/bin/bash
set -e
set +x

# run to generate convergence results shown in ../../doc/fas.pdf:
#   $ ./converge.sh &> converge.txt

for LEV in 1 2 3 4 5 6 7 8 9 10 11 12; do
    ../fas1.py -mms -cycles 8 -j $LEV
done

