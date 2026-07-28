.DEFAULT_GOAL := anonymous

LATEXMK ?= latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

ANONYMOUS_SOURCE := AnonymousSubmission2027.tex
ANONYMOUS_PDF := $(ANONYMOUS_SOURCE:.tex=.pdf)
ANONYMOUS_DEPS := paper_arxiv.bib aaai2027.sty aaai2027.bst \
	$(wildcard figures/*)

.PHONY: anonymous clean distclean

anonymous: $(ANONYMOUS_PDF)

$(ANONYMOUS_PDF): $(ANONYMOUS_SOURCE) $(ANONYMOUS_DEPS)
	$(LATEXMK) $(LATEXMK_FLAGS) $(ANONYMOUS_SOURCE)

clean:
	$(LATEXMK) -c $(ANONYMOUS_SOURCE)

distclean:
	$(LATEXMK) -C $(ANONYMOUS_SOURCE)
