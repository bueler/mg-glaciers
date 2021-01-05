all: fas.pdf

_figures := hatcombination
figures := $(addprefix figs/, $(addsuffix .pdf, $(_figures)))

fas.pdf: fas.tex fas.bib $(figures)
	pdflatex fas
	bibtex fas
	pdflatex fas
	pdflatex fas

$(figures): figs/hatfig.py
	(cd figs/ && python3 hatfig.py)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
	@rm -f figs/*.pdf

