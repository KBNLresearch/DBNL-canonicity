"""KB Researcher in Residence project: Dutch novels corpus 1800-2000."""
# this project
from queries import (runbasisbibliotheekquery, rundbnlcanonquery,
		runwikidataqueries, writenames, runauthorqueries)
from metadata import readmetadata, rundbnlmetadataconversion, writemetadata
from textprocessing import (runlangdetect, runfreqextraction, runsrpextraction,
		runpairwisesimilarity, runoverallfrequencies, runsentimentanalysis)
from plots import createtablesfigures
from files import makedirs


def main(path='output'):
	"""Run pipeline to generate output. Slow steps are cached."""
	makedirs()
	runlangdetect()
	runfreqextraction()
	runsrpextraction()
	runpairwisesimilarity()
	runwikidataqueries()
	runbasisbibliotheekquery()
	rundbnlcanonquery()
	rundbnlmetadataconversion()
	_orig, df, _disjointset = readmetadata()
	writemetadata(df, path)
	writenames(df)
	runauthorqueries('data/dbrd.txt.zst', 'output/dbrdmatches.txt')
	runauthorqueries('data/wiki19.txt.zst', 'output/wpmatches.txt')
	runoverallfrequencies()
	runsentimentanalysis()
	createtablesfigures(df)


if __name__ == '__main__':
	main()
