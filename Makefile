.PHONY: all run report check_dep f_poetry test

SRC := src
RES := res
MAIN := main.py
PKG_DIR := ai-project-group-3


# Name of the report file without extension
REPORT_DIR := report
REPORT := report
TEX_ENGINE := pdflatex
TEX_ARGS = -halt-on-error $(REPORT).tex | grep '^!.*' -A200
TEX_AUTOGEN = *.aux *.log *.out *.bbl *.blg *blx.bib *.fls *.fdb_latexmk *.synctex.gz *.synctex *.run.xml
PRES_DIR := presentation
SLIDES_WEB_DIR := $(PRES_DIR)/slides-svelte
SLIDES_WEB_RESOURCES := $(SLIDES_WEB_DIR)/static/presentation

all: run

run: check_dep 
	poetry run python $(SRC)/$(MAIN)

test: check_dep
	poetry run pytest -sv $(SRC)/*.py -k test  --disable-warnings  # warnings are very verbose and not always relevant 

# test f1 score of each model
test_model: check_dep
	poetry run pytest -sv $(SRC)/*.py -k test -k f1  --disable-warnings  # warnings are very verbose and not always relevant 

report:
	cd $(REPORT_DIR) && \
	pandoc $(REPORT).md -so $(REPORT).tex &&\
	($(TEX_ENGINE) $(TEX_ARGS)); bibtex $(REPORT) && ($(TEX_ENGINE) $(TEX_ARGS)) ;\
	cd ..

# install poetry and check if dependencies are installed
check_dep:
	@( [ -f ./poetry.lock ] && (command -v poetry > /dev/null 2>&1) ) ||\
	( echo "Dependencies not installed. Installing them..."; (./setup_poetry 1 || (exit 1) ))

# echo "Please make sure you've run 'pip install poetry'."; 

slides:
	cp -r $(PRES_DIR)/*.md $(SLIDES_WEB_RESOURCES)
	


# ZIP project for submission
package:
	mkdir $(PKG_DIR)
	mkdir $(PKG_DIR)/$(REPORT)
	cp -r $(RES) $(PKG_DIR)
	cp -r $(SRC) $(PKG_DIR)
	cp $(REPORT_DIR)/$(REPORT).pdf regressionLogistique.pdf $(PKG_DIR)/$(REPORT)
	cp Makefile pyproject.toml README.md $(PKG_DIR)
	rm $(PKG_DIR)/$(SRC)/*cache* $(PKG_DIR)/$(SRC)/.*cache* -rf
	- rm $(PKG_DIR)/poetry.lock
	zip -r $(PKG_DIR).zip $(PKG_DIR)
	rm $(PKG_DIR) -r  # removing temp dir

clean_report:
	@-rm  $(TEX_AUTOGEN) &> /dev/null ;\
	cd $(REPORT_DIR) && \
	rm  $(TEX_AUTOGEN) &> /dev/null ;\
	cd ..
