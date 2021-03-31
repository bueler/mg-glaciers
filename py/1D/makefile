all: mtest test

# hybrid testing strategy:
#   1. test modules using pytest
#   2. test top-level codes using testit.sh

mtest:
	@echo "**** TESTING MODULES ****"
	python3 -m pytest

runobstacle_1:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase traditional -poissonparabolay -3.0 -J 1 -sweepsonly -symmetric" 1 1 "poisson traditional [unconstrained] PGS sweeps symmetric"

runobstacle_2:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase traditional -J 4 -jcoarse 3 -cyclemax 2 -down 1 -up 0 -monitorerr -mgview" 1 2 "poisson traditional V(1,0) cycles monitor error and multigrid"

runobstacle_3:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase traditional -J 4" 1 3 "poisson traditional V(0,1) cycles"

runobstacle_4:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -J 3 -sweepsonly -omega 1.5 -cyclemax 3" 1 4 "poisson icelike PGS sweeps omega=1.5"

runobstacle_5:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -J 4 -ni -cyclemax 1 -down 1 -up 0 -monitorerr" 1 5 "poisson icelike F-cycle using V(1,0) cycle"

runobstacle_6:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -jacobi -omega 0.85 -monitor -irtol 1.0e-2" 1 6 "poisson icelike V(0,1) cycles Jacobi smoother"

runobstacle_7:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -J 3 -down 1 -up 1 -monitorerr -diagnostics -o test7.png" 1 7 "poisson icelike V(1,1) cycles generate diagnostics"

runobstacle_8:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase pde2 -J 3" 1 8 "poisson pde2 [unconstrained] V(1,0) cycles"

runobstacle_9:
	-@./../testit.sh obstacle.py "-problem sia -sweepsonly -symmetric -J 2 -monitor -irtol 1.0e-1" 1 9 "sia PNGS sweeps"

runobstacle_10:
	-@./../testit.sh obstacle.py "-problem sia -exactinitial -sweepsonly -jacobi -omega 0.5 -J 3 -monitor -cyclemax 2" 1 10 "sia PJacobi sweeps omega=0.5 exact initial"

runobstacle_11:
	-@./../testit.sh obstacle.py "-problem sia -sweepsonly -siashowsingular -J 4 -cyclemax 6 -monitor" 1 11 "sia PNGS sweeps show singular points"

runobstacle_12:
	-@./../testit.sh obstacle.py "-problem sia -J 4 -irtol 1.0e-1 -symmetric -monitor" 1 12 "sia V(0,1) cycles symmetric"

runobstacle_13:
	-@./../testit.sh obstacle.py "-problem sia -J 3 -down 1 -up 1 -monitor -ni -nicycles 2 -cyclemax 2" 1 13 "sia F-cycle using two V(1,1) cycles"

test_obstacle: runobstacle_1 runobstacle_2 runobstacle_3 runobstacle_4 runobstacle_5 runobstacle_6 runobstacle_7 runobstacle_8 runobstacle_9 runobstacle_10 runobstacle_11 runobstacle_12 runobstacle_13

testbanner:
	@echo "**** TESTING obstacle.py ****"

test: testbanner test_obstacle

.PHONY: clean runobstacle_1 runobstacle_2 runobstacle_3 runobstacle_4 runobstacle_5 runobstacle_6 runobstacle_7 runobstacle_8 runobstacle_9 runobstacle_10 runobstacle_11 runobstacle_12 runobstacle_13 test_obstacle testmodules test testbanner

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -rf __pycache__ .pytest_cache/
