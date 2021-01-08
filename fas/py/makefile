# testing

runfas1_1:
	-@./../../testit.sh fas1.py "-monitor -monitorcoarseupdate -cycles 2 -coarsesweeps 20 -levels 2" 1 1

runfas1_2:
	-@./../../testit.sh fas1.py "-monitor -cycles 5 -downsweeps 20 -ngsonly -mms" 1 2

runfas1_3:
	-@./../../testit.sh fas1.py "-monitor -cycles 3 -j 4 -downsweeps 2 -upsweeps 2 -mms -coarsesweeps 5" 1 3

test_fas1: runfas1_1 runfas1_2 runfas1_3

test: test_fas1

.PHONY: clean runfas1_1 runfas1_2 runfas1_3 test_fas1 test

clean:
	@rm -f maketmp tmp difftmp
	@rm -rf __pycache__

