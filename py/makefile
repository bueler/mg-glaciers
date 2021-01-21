# a hybrid testing strategy

testmodules:
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

test: testmodules test_obs1

.PHONY: clean runobs1_1 runobs1_2 runobs1_3 runobs1_4 runobs1_5 runobs1_6 test_obs1 testmodules test

clean:
	@rm -f maketmp tmp difftmp *test*.png
	@rm -rf __pycache__ .pytest_cache/

