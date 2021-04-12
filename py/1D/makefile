all: mtest test

# hybrid testing strategy:
#   1. test modules using pytest
#   2. test top-level codes using testit.sh

mtest:
	@echo "**** mtest: TESTING MODULES ****"
	python3 -m pytest

run_1:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase traditional -poissonparabolay -3.0 -J 1 -sweepsonly" 1 1 "poisson traditional PGS sweeps"

run_2:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase traditional -J 4 -jcoarse 3 -cyclemax 2 -down 1 -up 0 -monitorerr -mgview" 1 2 "poisson traditional V(1,0) cycles monitor error"

run_3:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase traditional -J 4" 1 3 "poisson traditional V(0,2) cycles"

run_4:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -J 3 -sweepsonly -omega 1.5 -cyclemax 3" 1 4 "poisson icelike PGS sweeps omega=1.5"

run_5:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -J 4 -ni -cyclemax 1 -down 1 -up 0 -monitorerr" 1 5 "poisson icelike F-cycle using V(1,0) cycle"

run_6:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -jacobi -omega 0.85 -monitor" 1 6 "poisson icelike V(0,2) cycles Jacobi smoother"

run_7:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase icelike -J 3 -down 1 -up 1 -monitorerr -diagnostics -o test7.png" 1 7 "poisson icelike V(1,1) cycles diagnostics"

run_8:
	-@./../testit.sh obstacle.py "-problem poisson -poissoncase pde2 -J 3" 1 8 "poisson pde2 [unconstrained] V(0,2) cycles"

run_9:
	-@./../testit.sh obstacle.py "-problem plap -plapcase bridge -up 2 -monitor" 1 9 "plap PNGS V(0,2) cycles"

run_10:
	-@./../testit.sh obstacle.py "-problem plap -plapcase bridge -down 1 -monitor -ni -nicycles 2 -jacobi -omega 0.5" 1 10 "plap PNJacobi F-cycle using 2 V(1,2) cycles"

run_11:
	-@./../testit.sh obstacle.py "-problem sia -sweepsonly -showsingular -J 4 -cyclemax 6 -monitor" 1 11 "sia PNGS sweeps show singular points"

run_12:
	-@./../testit.sh obstacle.py "-problem sia -up 1 -J 4 -irtol 1.0e-2 -monitor" 1 12 "sia V(0,1) cycles"

run_13:
	-@./../testit.sh obstacle.py "-problem sia -J 3 -down 1 -up 1 -monitor -ni -nicycles 2 -cyclemax 2" 1 13 "sia F-cycle using two V(1,1) cycles"

run_14:
	-@./../testit.sh obstacle.py "-problem sia -sweepsonly -J 2 -monitor -irtol 1.0e-2" 1 14 "sia PNGS sweeps"

run_15:
	-@./../testit.sh obstacle.py "-problem sia -exactinitial -sweepsonly -jacobi -omega 0.5 -J 3 -monitor -cyclemax 2" 1 15 "sia PJacobi sweeps omega=0.5 exact initial"

test_obstacle: run_1 run_2 run_3 run_4 run_5 run_6 run_7 run_8 run_9 run_10 run_11 run_12 run_13 run_14 run_15

testbanner:
	@echo "**** test: RUNS OF obstacle.py ****"

test: testbanner test_obstacle

.PHONY: clean test_obstacle testmodules test testbanner run_1 run_2 run_3 run_4 run_5 run_6 run_7 run_8 run_9 run_10 run_11 run_12 run_13 run_14 run_15

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -rf __pycache__ smoothers/__pycache__ .pytest_cache/
