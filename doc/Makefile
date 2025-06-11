# Simple Makefile for XeLaTeX-based thesis build

# Main document name (without .tex extension)
MAIN = licenta

# Compiler
LATEX = xelatex
BIBTEX = biber

# Files to clean
AUXFILES = *.aux *.log *.toc *.out *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Default target: build PDF
all: $(MAIN).pdf

# PDF build rule
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Clean rule
clean:
	rm -f $(AUXFILES) $(MAIN).pdf

# Clean only aux files (keep PDF)
cleanaux:
	rm -f $(AUXFILES)
