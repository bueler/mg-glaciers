all:

run_1:
	-@./../testit.sh steady.py "-irtol 1.0e-1 -J 2 -alpha 600" 1 1 "steady coarse basic"

test_steady: run_1

test: test_steady

.PHONY: clean test_steady test run_1

clean:
	@rm -f maketmp tmp difftmp *.png *.pdf
	@rm -f *.pyc *.pvd *.pvtu *.vtu *.m
	@rm -rf __pycache__ .pytest_cache/
