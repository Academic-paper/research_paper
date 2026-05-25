# Dissertation build pipeline.
#
#   make figures   -> regenerate every matplotlib figure under figures/
#                     using the latest benchmark summaries in research/results
#   make pdf       -> compile thesis.tex into thesis.pdf via tectonic
#   make rebuild   -> figures + pdf
#   make clean     -> remove generated artefacts (PDF and aux files)
#
# Configurable: PYTHON points at a Python interpreter that has matplotlib,
# numpy and scipy installed. Defaults to the virtual env used by the
# benchmark harness.

PYTHON   ?= /tmp/p3sl_venv/bin/python
TECTONIC ?= tectonic

.PHONY: figures pdf rebuild clean

figures:
	$(PYTHON) make_figures.py

pdf:
	$(TECTONIC) thesis.tex

rebuild: figures pdf

clean:
	rm -f thesis.pdf thesis.aux thesis.log thesis.toc thesis.lof thesis.lot thesis.out
