# a hybrid testing strategy: test modules using pytest and test driver using testit.sh

testmodules:
	@echo "**** TESTING MODULES ****"
	python3 -m pytest

runobs1_1:
	-@./../testit.sh obs1.py "-pgsonly -kfine 3 -down 3" 1 1

runobs1_2:
	-@./../testit.sh obs1.py "-pgsonly -down 10 -symmetric -problem low -kfine 1" 1 2

runobs1_3:
	-@./../testit.sh obs1.py "-monitorerr -problem low -kfine 3 -cycles 3 -symmetric -diagnostics -o test3.png" 1 3

runobs1_4:
	-@./../testit.sh obs1.py "-monitorerr -mgview -kfine 4 -kcoarse 3" 1 4

runobs1_5:
	-@./../testit.sh obs1.py "-mgview -cycles 1 -random" 1 5

runobs1_6:
	-@./../testit.sh obs1.py "-monitor -kfine 4 -cycles 4 -problem icelike" 1 6

test_obs1: runobs1_1 runobs1_2 runobs1_3 runobs1_4 runobs1_5 runobs1_6

testbanner:
	@echo "**** TESTING obs1.py ****"

test: testmodules testbanner test_obs1

.PHONY: clean runobs1_1 runobs1_2 runobs1_3 runobs1_4 runobs1_5 runobs1_6 test_obs1 testmodules test testbanner

clean:
	@rm -f maketmp tmp difftmp *test*.png
	@rm -rf __pycache__ .pytest_cache/

