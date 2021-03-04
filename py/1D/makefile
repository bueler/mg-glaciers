# hybrid testing strategy:
#   1. test modules using pytest
#   2. test top-level codes using testit.sh

testmodules:
	@echo "**** TESTING MODULES ****"
	python3 -m pytest

runobstacle_1:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -jfine 3 -pgsonly -omega 1.5 -cyclemax 3" 1 1

runobstacle_2:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase parabola -parabolay -3.0 -jfine 1 -pgsonly -symmetric" 1 2

runobstacle_3:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -jfine 3 -monitorerr -diagnostics -o test3.png" 1 3

runobstacle_4:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase parabola -jfine 4 -jcoarse 3 -cyclemax 2 -monitorerr -mgview" 1 4

runobstacle_5:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -jfine 4 -monitorerr -errtol 5.0e-3 -ni" 1 5

runobstacle_6:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -jacobi -omega 0.85 -monitor -irtol 1.0e-2" 1 6

runobstacle_7:
	-@./../testit.sh obstacle.py "-problem sia -pgsonly -jfine 2 -monitor -irtol 1.0e-1" 1 7

runobstacle_8:
	-@./../testit.sh obstacle.py "-problem sia -exactinitial -pgsonly -jfine 3 -monitor -cyclemax 2" 1 8

test_obstacle: runobstacle_1 runobstacle_2 runobstacle_3 runobstacle_4 runobstacle_5 runobstacle_6 runobstacle_7 runobstacle_8

testbanner:
	@echo "**** TESTING obstacle.py ****"

test: testmodules testbanner test_obstacle

.PHONY: clean runobstacle_1 runobstacle_2 runobstacle_3 runobstacle_4 runobstacle_5 runobstacle_6 runobstacle_7 runobstacle_8 test_obstacle testmodules test testbanner

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -rf __pycache__ .pytest_cache/
