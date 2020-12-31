# testing

runfas1_1:
	-@./../../testit.sh fas1.py "-monitor -cycles 2 -coarsesweeps 20 -levels 2" 1 1

runfas1_2:
	-@./../../testit.sh fas1.py "-monitor -cycles 5 -downsweeps 20 -ngsonly -mms" 1 2

test_fas1: runfas1_1 runfas1_2

test: test_fas1

.PHONY: clean runfas1_1 runfas1_2 test_fas1 test

clean:
	@rm -f maketmp tmp difftmp
	@rm -rf __pycache__

