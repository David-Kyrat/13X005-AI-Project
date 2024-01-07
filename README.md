# 13X005 AI Project

End of semester AI Project on Logistic Regression and Naive Bayes.

Assignement Pdf:

[regressionLogistique.pdf](regressionLogistique.pdf)

## Todo

Link to trello used to manage todos:  [https://trello.com/b/xhglaB3g/13x005-ai-project](https://trello.com/b/xhglaB3g/13x005-ai-project)

**Preview:** 


<div align="center">
<a href="res/todo_img.png"><img src="res/todo_img.png" alt="Trello Board Screenshot" align="center" width="45%"></a>
</div>


## Building & Runing

Poetry was used to simplify the project & dependencies setup, 
i.e. avoid problems related to python/package versions as well as the _"It works on my machine"_ problem.

- **Dependencies:**
  If you don't already have the requiered dependencies (numpy, sklearn ...),  
  run `poetry install` to install them (if you don't have poetry, you can install it with `pip install poetry`).

- **Run:** To run it just use `make`.  
    or manually "poetry run python src/main.py"

### Editing the report

- You can directly edit the markdown version in [report/report.md](report/report.md) and use `make report` to convert it from markdown to latex and from latex pdf.
([pandoc](https://pandoc.org) and [pdflatex](https://www.latex-project.org/get/) are required for this to work. Pandoc should be installed by default on most linux distributions.)

- The report follows the LaTeX template defined in [preamble_ai_project.sty](./report/preamble_ai_project.sty), which looks like this:
[pdf-report](./report/report.pdf)

- The citations are in the file [report/references.bib](report/references.bib) and can be called with `\cite{citation-key}`.


## Authors

- [Gregory Sedykh](https://github.com/gregorysedykh)
- [Arkanthara](https://github.com/Arkanthara)
- [CTGN](https://github.com/CTGN)
- [Aviel20002](https://github.com/Aviel20002)
- [Noah Munz (Me)](https://github.com/David-Kyrat)
