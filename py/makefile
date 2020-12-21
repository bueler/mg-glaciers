# testing

runobs1_1:
	-@./../testit.sh obs1.py "-pgs -jfine 3 -downsweeps 3" 1 1

runobs1_2:
	-@./../testit.sh obs1.py "-mgview -cycles 1 -random" 1 2

runobs1_3:
	-@./../testit.sh obs1.py "-monitor -low -jfine 3 -cycles 3 -symmetric -diagnostics -o test3.png" 1 3

runobs1_4:
	-@./../testit.sh obs1.py "-monitor -mgview -jfine 4 -jcoarse 3" 1 4

test_obs1: runobs1_1 runobs1_2 runobs1_3 runobs1_4

test: test_obs1

.PHONY: clean runobs1_1 runobs1_2 runobs1_3 runobs1_4 test_obs1 test

clean:
	@rm -f maketmp tmp difftmp *test*.png
	@rm -rf __pycache__

