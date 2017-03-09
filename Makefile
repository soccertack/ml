PAPER = hw4-sol
TEX = $(wildcard *.tex)
BIB = references.bib
FIGS = $(wildcard figures/*.pdf figures/*.png graphs/*.pdf graphs/*.png)

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(FIGS)
	echo $(FIGS)
	pdflatex $(PAPER)
	bibtex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)
	rm -f *.aux *.bbl *.blg *.log *.out

clean:
	rm -f *.aux *.bbl *.blg *.log *.out $(PAPER).pdf

