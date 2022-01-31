.PHONY: test lint coverage clean rir

all: rir

test:
	pytest -vv -Werror *.py

testpdb:
	pytest -vv --pdb --pdbcls=IPython.terminal.debugger:Pdb

lint:
	pycodestyle --ignore=E1,W1,E402,E501,W503 *.py \
	&& pylint *.py

coverage:
	coverage run -m pytest
	coverage html
	coverage report -m
	@echo "Check the report at: file://`pwd`/htmlcov/index.html"

clean:
	rm -rf output/jsonl output/mallet output/tokenized srp.bin langdetect.tsv

rir:
	python3 rir.py
