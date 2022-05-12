LATEXC := pdflatex
BIBC := bibtex
SECTIONS := $(shell find sections/ -type f)
STYLES := $(shell find styles/ -type f)
BIBLIOGRAPHY := $(shell find bibliography/*.bib -type f)
PAPER := notes.pdf
DIR := $(shell git rev-parse --show-prefix)

.PHONY: main clean cleanall lm-clean view

main: ${PAPER}

${PAPER}: main.tex ${SECTIONS} ${STYLES} ${BIBLIOGRAPHY}
	$(eval MAIN=$(patsubst %.tex,%,$<))
	$(eval OUT=$(patsubst %.pdf,%,$@))
	$(LATEXC) --shell-escape -jobname=${OUT} ${MAIN}
	$(BIBC) ${OUT}
	$(LATEXC) --shell-escape -jobname=${OUT} ${MAIN}
	$(LATEXC) --shell-escape -jobname=${OUT} ${MAIN}


clean:
	@git clean -xdf . \
		-e ${PAPER} \
		-e .vscode/ \
		-e ${DIR}assets/plots/

cleanall: clean
	rm -f ${PAPER}

lm-clean:
	latexmk -c

view:
	xdg-open ${PAPER}
