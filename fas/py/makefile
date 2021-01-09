# testing

runfas1_1:
	-@./../../testit.sh fas1.py "-monitor -monitorcoarseupdate -cycles 2 -coarse 20 -levels 2" 1 1

runfas1_2:
	-@./../../testit.sh fas1.py "-monitor -cycles 5 -down 20 -ngsonly -mms" 1 2

runfas1_3:
	-@./../../testit.sh fas1.py "-monitor -cycles 3 -j 4 -down 2 -up 2 -mms -coarse 5" 1 3

test_fas1: runfas1_1 runfas1_2 runfas1_3

test: test_fas1

.PHONY: clean runfas1_1 runfas1_2 runfas1_3 test_fas1 test

clean:
	@rm -f maketmp tmp difftmp
	@rm -rf __pycache__

