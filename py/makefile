# hybrid testing strategy:
#   1. test modules using pytest
#   2. test top-level codes using testit.sh

testmodules:
	@echo "**** TESTING MODULES ****"
	python3 -m pytest

runobstacle_1:
	-@./../testit.sh obstacle.py "-problem parabola -kfine 3 -pgsonly -cyclemax 3" 1 1

runobstacle_2:
	-@./../testit.sh obstacle.py "-problem parabola -parabolay -3.0 -kfine 1 -pgsonly -symmetric" 1 2

runobstacle_3:
	-@./../testit.sh obstacle.py "-kfine 3 -up 0 -monitorerr -diagnostics -o test3.png" 1 3

runobstacle_4:
	-@./../testit.sh obstacle.py "-problem parabola -kfine 4 -kcoarse 3 -cyclemax 2 -up 0 -monitorerr -mgview -slash" 1 4

runobstacle_5:
	-@./../testit.sh obstacle.py "-mgview -kfine 2 -random -fscale 0.5 -monitor -irtol 1.0e-1" 1 5

test_obs1: runobstacle_1 runobstacle_2 runobstacle_3 runobstacle_4 runobstacle_5

testbanner:
	@echo "**** TESTING obstacle.py ****"

test: testmodules testbanner test_obs1

.PHONY: clean runobstacle_1 runobstacle_2 runobstacle_3 runobstacle_4 runobstacle_5 test_obs1 testmodules test testbanner

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -rf __pycache__ .pytest_cache/

