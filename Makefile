export SHELL := /bin/bash

test:
	python -m unittest discover .

unittests:
	pytest kerax

coverage:
	pytest --cov=kerax --cov-config=.coveragerc kerax

