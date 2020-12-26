all: review.pdf

_figures := finehats coarsehats coarsesthats
#figures := $(addprefix figs/, $(_figures))
figures := $(addprefix figs/, $(addsuffix .pdf, $(_figures)))

review.pdf: review.tex review.bib $(figures)
	pdflatex review
	bibtex review
	pdflatex review
	pdflatex review

$(figures): figs/hats1d.py
	(cd figs/ && python3 hats1d.py)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done


.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
	@rm -f figs/*.pdf

