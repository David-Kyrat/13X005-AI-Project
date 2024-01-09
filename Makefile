.PHONY: all run report

SRC := src
RES := res
MAIN := main.py


# Name of the report file without extension
REPORT_DIR := report
REPORT := report
TEX_ENGINE := pdflatex
TEX_ARGS = -halt-on-error $(REPORT).tex | grep '^!.*' -A200

all: run

run: 
	poetry run python $(SRC)/$(MAIN)

test:
	poetry run pytest -sv $(SRC)/*.py -k test  --disable-warnings  # warnings are very verbose and not always relevant 

# test f1 score of each model
test_model:
	poetry run pytest -sv $(SRC)/*.py -k test -k f1  --disable-warnings  # warnings are very verbose and not always relevant 

report:
	cd $(REPORT_DIR) && \
	pandoc $(REPORT).md -so $(REPORT).tex &&\
	($(TEX_ENGINE) $(TEX_ARGS)); bibtex $(REPORT) && ($(TEX_ENGINE) $(TEX_ARGS)) ;\
	cd ..

# pdflatex -halt-on-error report.tex && bibtex report && pdflatex -halt-on-error report.tex
# $(TEX_ENGINE) $(TEX_ARGS) && bibtex $(REPORT) && $(TEX_ENGINE) $(TEX_ARGS)  ;\

clean_report:
	@-cd $(REPORT_DIR) && \
	rm *.aux *.log *.out *.toc *.pdf *.bbl *.blg *.dvi *.ps *.lof *.lot *.gz *.fls *.fdb_latexmk *.synctex.gz *.synctex &> /dev/null ;\
	cd ..
