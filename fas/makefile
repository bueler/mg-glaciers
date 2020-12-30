all: fas.pdf

fas.pdf: fas.tex fas.bib $(figures)
	pdflatex fas
	bibtex fas
	pdflatex fas
	pdflatex fas

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz

