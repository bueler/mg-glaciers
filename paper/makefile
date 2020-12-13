all: review.pdf

review.pdf: review.tex review.bib
	pdflatex review
	bibtex review
	pdflatex review
	pdflatex review


.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl

