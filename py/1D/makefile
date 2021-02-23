# hybrid testing strategy:
#   1. test modules using pytest
#   2. test top-level codes using testit.sh

testmodules:
	@echo "**** TESTING MODULES ****"
	python3 -m pytest

runobstacle_1:
	-@./../testit.sh obstacle.py "-problem parabola -jfine 3 -pgsonly -cyclemax 3" 1 1

runobstacle_2:
	-@./../testit.sh obstacle.py "-problem parabola -parabolay -3.0 -jfine 1 -pgsonly -symmetric" 1 2

runobstacle_3:
	-@./../testit.sh obstacle.py "-jfine 3 -monitorerr -diagnostics -o test3.png" 1 3

runobstacle_4:
	-@./../testit.sh obstacle.py "-problem parabola -jfine 4 -jcoarse 3 -cyclemax 2 -monitorerr -mgview" 1 4

#runobstacle_5:
#	-@./../testit.sh obstacle.py "-up 1 -mgview -jfine 2 -random -fscale 0.5 -monitor -irtol 1.0e-1" 1 5

test_obs1: runobstacle_1 runobstacle_2 runobstacle_3 runobstacle_4

testbanner:
	@echo "**** TESTING obstacle.py ****"

test: testmodules testbanner test_obs1

.PHONY: clean runobstacle_1 runobstacle_2 runobstacle_3 runobstacle_4 test_obs1 testmodules test testbanner

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -rf __pycache__ .pytest_cache/
