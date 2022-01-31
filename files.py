"""Define file locations."""
import os
from glob import glob


def input_f(fname):
	"""return path for input (metadata) file"""
	return os.path.join('data', 'metadata', fname)


def generated_f(fname):
	"""return path for generated file"""
	return os.path.join('data', 'generated', fname)


def output_f(fname):
	"""return path for output file"""
	return os.path.join('output', fname)


def streng_f(fname):
	"""return path for file in Streng metadata"""
	return os.path.join('data', 'metadata', 'streng', fname)


def makedirs():
	"""Ensure directories exist."""
	if not os.path.exists(input_f('')):
		raise ValueError('directory not found: %r' % input_f(''))
	for path in (generated_f, output_f):
		os.makedirs(path(''), exist_ok=True)


DBNLMETADATA = input_f('DBNL_metadata_romans_1800-2000.xlsx')
DBNLVERSIONS = input_f('DBNL_titlecontains.xlsx')
TEIFILES = sorted(glob('data/xml/*.xml'))

JSONDIR = input_f('metadatapertitle')

OBMETADATA = input_f('PublicLibraryData.xlsx')
OBSUMMARY = generated_f('publiclibrariessummarized.tsv')

GGCMETADATA = input_f('ggc_metadata_20210727.xlsx')
GGCSUMMARY = generated_f('ggc_summarized.tsv')

OZPCLUSTERS = input_f('OZP-clusters.csv')
MANUALMATCHES = input_f('PPN-manualMatch.csv')
PPNLINKS = generated_f('dbnl-ppn.tsv')

REGISTER = input_f(
		'Registers_literatuurgeschiedenissen_1800-2005_values_only.xlsx')
