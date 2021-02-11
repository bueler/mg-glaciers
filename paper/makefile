all: review.pdf

_figures := finehats coarsehats coarsesthats residualpoints cartoonplane prolongobstacle
figures := $(addprefix genfigs/, $(addsuffix .pdf, $(_figures)))

review.pdf: review.tex review.bib $(figures)
	pdflatex review
	bibtex review
	pdflatex review
	pdflatex review

$(figures): genfigs/hats1d.py genfigs/residualpoints.py genfigs/cartoonplane.py
	(cd genfigs/ && python3 hats1d.py && python3 residualpoints.py && python3 cartoonplane.py && python3 prolongobstacle.py)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done


.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
	@rm -f genfigs/*.pdf
	@rm -rf .pytest_cache/

