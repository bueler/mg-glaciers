# testing

runobs1_1:
	-@./../testit.sh obs1.py "-pgs -jfine 3 -downsweeps 3" 1 1

runobs1_2:
	-@./../testit.sh obs1.py "-mgview -cycles 1 -random" 1 2

runobs1_3:
	-@./../testit.sh obs1.py "-monitor -problem low -jfine 3 -cycles 3 -symmetric -diagnostics -o test3.png" 1 3

runobs1_4:
	-@./../testit.sh obs1.py "-monitor -mgview -jfine 4 -jcoarse 3" 1 4

runfas1_1:
	-@./../testit.sh fas1.py "-monitor -cycles 2 -coarsesweeps 20 -levels 2" 1 1

runfas1_2:
	-@./../testit.sh fas1.py "-monitor -cycles 5 -downsweeps 20 -ngsonly" 1 2

test_obs1: runobs1_1 runobs1_2 runobs1_3 runobs1_4

test_fas1: runfas1_1 runfas1_2

test: test_obs1 test_fas1

.PHONY: clean runobs1_1 runobs1_2 runobs1_3 runobs1_4 runfas1_1 runfas1_2 test_obs1 test_fas1 test

clean:
	@rm -f maketmp tmp difftmp *test*.png
	@rm -rf __pycache__

