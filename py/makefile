# testing

runobstacle1_1:
	-@./../testit.sh obstacle1.py "-pgs -j 3 -sweeps 3" 1 1

runobstacle1_2:
	-@./../testit.sh obstacle1.py "" 1 2

test_obstacle1: runobstacle1_1 runobstacle1_2

test: test_obstacle1

.PHONY: clean runobstacle1_1 runobstacle1_2 test_obstacle1 test

clean:
	@rm -f maketmp tmp difftmp
	@rm -rf __pycache__

