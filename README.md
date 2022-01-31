# KB RiR project to Collect a corpus of Dutch novels 1800-2000 and Investigate Canonicity

What determines canonicity? Is this purely subjective, or can we partly
attribute it to textual features?

![Image of books](./canon.jpg)

## project aims

- Collect a corpus of Dutch novels 1800-2000
- Investigate canonicity
- Release a [dataset](https://doi.org/10.5281/zenodo.5786254) of textual features and metadata
- Create an [online demo](https://kbresearch.nl/canonizer/)
- World domination

## requirements

Tested on Linux. Windows with WSL probably works too.

	$ pip3 install -r requirements.txt

The CLI version of zstd is required, see https://github.com/facebook/zstd
or

	$ sudo apt install zstd

Required external metadata is expected to be in `data/metadata`.
Extra external data is expected to be in `data/metadata/streng`.
Required data (TEI-files) is expected to be in `data/xml`.
Temporary generated data is written to `data/generated`.
Generated, reproducible output is written to `data/output/`, `fig/` and `tbl/`.

## reproducing results (needs to be run at least once before `make test`)

	$ make

## running tests, coverage, lint

	$ make test
	$ make coverage
	$ make lint

Warnings are treated as errors. Install latest versions of all packages
(including dependencies) to avoid them.
See `htmlcov/index.html` for the test coverage report.
