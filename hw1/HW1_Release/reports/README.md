# Homework 1 LaTeX report

The self-contained project is in `iclr_hw1/`; its main document is `main.tex`.
It uses the ICLR 2026 style, with English text and sections matching the assignment.
The official style files are unchanged. Author-visible mode is enabled, and the
coursework header replaces the conference-publication header in the document.

Template source: [ICLR 2026 author guide](https://iclr.cc/Conferences/2026/AuthorGuide)
and [ICLR Master Template](https://github.com/ICLR/Master-Template/tree/master/iclr2026).
The `.sty` and `.bst` files were copied from the locally bundled ICLR 2026 template.

## Compile

Open `iclr_hw1/` in Codex's LaTeX editor, or upload the contents of that directory
to Overleaf and choose pdfLaTeX. All report figures and data snapshots are included.
No Python environment is needed just to compile the saved report.

```sh
cd iclr_hw1
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This Mac already has TeX Live 2026 in `/Library/TeX/texbin`. If it is not on PATH:

```sh
PATH="/Library/TeX/texbin:$PATH" make
```

The generated file is `iclr_hw1/main.pdf`.

## Edit and refresh

- `main.tex`: title, student name/ID, package configuration, document structure.
- `sections/part1.tex`: placeholders for the author's independent handwritten scans.
- `sections/part2.tex`: completed NumPy implementation/check/training results.
- `sections/part3.tex`: completed Task 1/2 and a placeholder for Task 3.
- `sections/ai_use.tex`: factual assistance disclosure requiring author review.
- `data/`: snapshots of the official result JSON files.
- `generated/`: tables generated from those JSON files.
- `figures/`: vector PDF figures suitable for the report.

To refresh from `../../results/` after new experiments:

```sh
python prepare_results.py --refresh
```

To rebuild tables/figures from the included snapshots only:

```sh
python prepare_results.py
```

These commands need NumPy and Matplotlib. They never rerun training. Numerical
statements in the prose must be reviewed if refreshed experimental results change.
The bibliography currently cites the supplied course handout only; no unrelated
template/example references were carried into the report.

## Before submission

Replace the name/ID placeholders, insert the handwritten solutions, complete the
independent investigation, and review the AI Use Statement. This draft does not
claim those unfinished requirements have been completed or independently certify
compliance with the course's AI-use rules.
