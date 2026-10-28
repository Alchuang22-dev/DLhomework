.PHONY: all figures clean
LATEXMK ?= latexmk
PYTHON ?= python3

all:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

figures:
	$(PYTHON) prepare_results.py

clean:
	$(LATEXMK) -c main.tex
