all: partI.pdf

_figuresI := finehats coarsehats coarsesthats residualpoints cartoonplane innerconeapprox prolongobstacle giscross poisson/convergence poisson/mcdl-cycles poisson/mcdl-wu poisson/mcdl-timeper poisson/perfni sia/convergence sia/asymprates sia/reshistory plap/convergence

figuresI := $(addprefix genfigs/, $(addsuffix .pdf, $(_figuresI)))

cropfiguresI: $(figuresI)
	for X in $(figuresI); do \
	     (pdfcrop $$X $$X); \
	done

partI.pdf: partI.tex partI.bib cropfiguresI
	pdflatex partI
	bibtex partI
	pdflatex partI
	pdflatex partI

writeout:
	(cd genfigs/poisson/ && ln -sf ../writeout.py)
	(cd genfigs/sia/ && ln -sf ../writeout.py)
	(cd genfigs/plap/ && ln -sf ../writeout.py)

# ampersand & is for a "grouped" target; requires make 4.3
genfigs/finehats.pdf genfigs/coarsehats.pdf genfigs/coarsesthats.pdf &: writeout genfigs/hats1d.py
	(cd genfigs/ && python3 hats1d.py)

genfigs/residualpoints.pdf: writeout genfigs/residualpoints.py
	(cd genfigs/ && python3 residualpoints.py)

genfigs/cartoonplane.pdf: writeout genfigs/cartoonplane.py
	(cd genfigs/ && python3 cartoonplane.py)

genfigs/prolongobstacle.pdf: writeout genfigs/prolongobstacle.py
	(cd genfigs/ && python3 prolongobstacle.py)

genfigs/giscross.pdf: writeout genfigs/giscross.py
	(cd genfigs/ && python3 giscross.py)

genfigs/poisson/convergence.pdf: writeout genfigs/poisson/convergence.py
	(cd genfigs/poisson/ && python3 convergence.py)

genfigs/poisson/mcdl-cycles.pdf genfigs/poisson/mcdl-wu.pdf genfigs/poisson/mcdl-timeper.pdf &: writeout genfigs/poisson/performance.py
	(cd genfigs/poisson/ && python3 performance.py)

genfigs/poisson/perfni.pdf: writeout genfigs/poisson/perfni.py
	(cd genfigs/poisson/ && python3 perfni.py)

genfigs/sia/convergence.pdf: writeout genfigs/sia/convergence.py
	(cd genfigs/sia/ && python3 convergence.py)

genfigs/sia/asymprates.pdf: writeout genfigs/sia/asymprates.py
	(cd genfigs/sia/ && python3 asymprates.py)

genfigs/sia/reshistory.pdf: writeout genfigs/sia/reshistory.py
	(cd genfigs/sia/ && python3 reshistory.py)

genfigs/plap/convergence.pdf: writeout genfigs/plap/convergence.py
	(cd genfigs/plap/ && python3 convergence.py)

.PHONY: clean writeout cropfiguresI

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
	@rm -rf genfigs/*.pdf genfigs/poisson/*.pdf genfigs/sia/*.pdf genfigs/plap/*.pdf
	@rm -rf genfigs/__pycache__/ genfigs/poisson/__pycache__/ genfigs/sia/__pycache__/ genfigs/plap/__pycache__/
	@rm -rf genfigs/poisson/writeout.py genfigs/sia/writeout.py genfigs/plap/writeout.py
	@rm -rf .pytest_cache/
