"""Create tables (LaTeX) and plots (PDF).

Anything used in publications should be produced in this module."""

# stdlib
import os

# pip
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from metadata import readexcel
from queries import readauthormatches


def createtablesfigures(metadata):
	"""Create tables (LaTeX) and plots (PDF). Anything used in publications
	should be produced in this function."""
	def save(fig, name, tightlayout=True, dpi=150):
		"""Save plot as PDF."""
		if tightlayout:  # workaround to prevent cropping of labels
			fig.tight_layout()
		fig.savefig('fig/%s.pdf' % name)
		# fig.savefig('fig/%s.svg' % name, backend='SVG')
		fig.savefig('fig/%s.png' % name, dpi=dpi)
		plt.close(fig)

	os.makedirs('tbl', exist_ok=True)
	cnt, idx = np.histogram(
			metadata['YearFirstPublished'],
			bins=range(1800, 2021, 10))
	cnt = pd.Series(cnt, index=idx[:-1])
	cnt.index.name = 'Decade'
	cnt.to_latex('tbl/hist.tex', header=['# texts'])

	os.makedirs('fig', exist_ok=True)
	# histogram of number of texts
	fig, ax = plt.subplots(figsize=(8, 3.5), sharex=True, sharey=True)
	ax = metadata['YearFirstPublished'].hist(bins=range(1800, 2021, 10))
	ax.set_ylabel('# texts')
	# FIXME: add ticklabels for all decades?
	save(fig, 'hist')

	# same plot with one or more canonicity indicators
	fig, ax = plt.subplots(figsize=(6, 3.5), sharex=True, sharey=True)
	ax = metadata.rename(columns={'YearFirstPublished': '# DBNL texts'}
			)['# DBNL texts'].hist(
			bins=range(1800, 2021, 10), legend=True)
	ax.set_ylabel('# texts')
	canondf = pd.read_csv(
			'data/nederlandsecanon2002.tsv',
			sep='\t').rename(columns={
				'Digitale_Bibliotheek_voor_de_Nederlandse_Letteren_author_ID':
					'pers_id'})
	canon_pers_id = canondf['pers_id'].drop_duplicates()
	canonicalsubset = metadata[metadata['DBNLpers_id'].isin(canon_pers_id)]
	canonicalsubset.rename(columns={
			'YearFirstPublished': '# texts by authors in Canon (2002)'}
			)['# texts by authors in Canon (2002)'].hist(
			bins=range(1800, 2021, 10), color='r', ax=ax, legend=True)
	ax.set_ylim(0, 250)
	save(fig, 'histcanon')

	fig, ax = plt.subplots(figsize=(6, 3.5), sharex=True, sharey=True)
	ax = metadata.rename(columns={'YearFirstPublished': '# DBNL texts'}
			)['# DBNL texts'].hist(
			bins=range(1800, 2021, 10), legend=True)
	ax.set_ylabel('# texts')
	canonicalsubset = metadata[metadata['DBNLSecRefsTitle'] > 0]
	canonicalsubset.rename(columns={
			'YearFirstPublished': '# texts with secondary references'}
			)['# texts with secondary references'].hist(
			bins=range(1800, 2021, 10), color='r', ax=ax, legend=True)
	ax.set_ylim(0, 250)
	save(fig, 'histcanonsecref')

	# 19th century novels: DBNL (sample) vs Streng (population of novels)
	fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
	metadata.rename(columns={'YearFirstPublished': '# texts'})['# texts'].hist(
			bins=range(1800, 1901, 10), ax=axes[0], legend=True)
	axes[0].set_ylabel('# texts')
	axes[0].set_title('DBNL texts')
	canonicalsubset.rename(columns={
			'YearFirstPublished': '# texts by authors in Canon (2002)'}
			)['# texts by authors in Canon (2002)'].hist(
			bins=range(1800, 1901, 10), color='r', ax=axes[0], legend=True)
	strengnovels = readexcel('data/streng/ZelfstTitelsNld.Romans.Excel.xlsx')
	strengnovels['TitelGegevensPerDruk::Jvu01'].hist(
			bins=range(1800, 1901, 10), ax=axes[1])
	axes[1].set_title('All Dutch novels (Streng, 2020)')
	save(fig, 'samplevspop19thcentury')

	# Histogram plots of canonicity indicators
	fig, ax = plt.subplots(figsize=(10, 6))
	matchcounts = readauthormatches('output/wpmatches.txt', metadata)
	sns.histplot(matchcounts, bins=20, ax=ax)
	ax.set_xlabel('number of matches on Wikipedia 2019')
	ax.set_ylabel('number of DBNL authors')
	save(fig, 'wpmatches')

	fig, ax = plt.subplots(figsize=(10, 6))
	matchcounts = readauthormatches('output/dbrdmatches.txt', metadata)
	sns.histplot(matchcounts, bins=20, ax=ax)
	ax.set_xlabel('number of matches on Hebban (DBRD)')
	ax.set_ylabel('number of DBNL authors')
	save(fig, 'dbrdmatches')

	fig, ax = plt.subplots(figsize=(10, 6))
	sns.histplot(metadata.holding, bins=20, ax=ax)
	ax.set_xlabel('number of copies in libraries')
	ax.set_ylabel('number of DBNL titles')
	save(fig, 'histholding')
	fig, ax = plt.subplots(figsize=(10, 6))
	sns.histplot(metadata.holding, bins=20, ax=ax)
	ax.set_xlabel('# times lent from libraries')
	ax.set_ylabel('number of DBNL titles')
	save(fig, 'histlending')

	# Bar plot of coverage of the canonicity indicators
	canonicity = [
			'AuthorInCanon2002',
			# 'TitleInCanon2002',
			'InBasisbibliotheek2008',
			'holding',
			'lending',
			'AuthorDBRDMatches',
			'AuthorNLWikipedia2019Matches',
			'GNTpages',
			'DBNLSecRefsAuthor',
			'DBNLSecRefsTitle',
			]
	canrename = {
			'InBasisbibliotheek2008': 'TitleInBasisbibliotheek2008',
			'AuthorNLWikipedia2019Matches': 'AuthorWikipediaMatches',
			'holding': 'TitleLibraryHoldings',
			'lending': 'TitleLibraryLending',
			'GNTpages': 'AuthorGNTpages',
			}
	ax = (metadata.loc[:, canonicity[::-1]].rename(columns=canrename) != 0
			).sum().plot.barh(figsize=(7, 3.5))
	ax.set_xlabel('# titles w/nonzero value')
	save(ax.figure, 'canonicitybarplot')

	# Create several scatterplots
	labels = {'AuthorNLWikipedia2019Matches': '# Author mentions on Wikipedia',
			'DBNLSecRefsAuthor': '# Author secondary references on DBNL',
			'DBNLSecRefsTitle': '# Novel secondary references on DBNL',
			'AuthorDBRDMatches': '# Author mentions on Hebban (book reviews)',
			'AuthorInCanon2002': 'In NL Canon (2002)',
			'GNTpages': 'No of pages mentioning author in GNT',
			'holding': 'number of copies in libraries',
			'lending': '# times lent from libraries',
			}
	plots = [('secrefvswiki', 'DBNLSecRefsAuthor',
				'AuthorNLWikipedia2019Matches'),
			('dbrdvswiki', 'AuthorDBRDMatches',
				'AuthorNLWikipedia2019Matches'),
			('secrefvsdbrd', 'AuthorDBRDMatches', 'DBNLSecRefsAuthor'),
			('secrefvsgnt', 'GNTpages', 'DBNLSecRefsAuthor'),
			('secrefsvsholding', 'holding', 'DBNLSecRefsTitle'),
			('secrefsvslending', 'lending', 'DBNLSecRefsTitle'),
			]
	for fname, x, y in plots:
		fg = sns.relplot(  # pylint: disable=invalid-name
				x=x, y=y, hue='AuthorInCanon2002', style='AuthorInCanon2002',
				data=metadata.replace(
					{'AuthorInCanon2002': {0: 'no', 1: 'yes'}}),
				height=5, alpha=0.5, markers=True,
				facet_kws=dict(legend_out=False))
		# https://stackoverflow.com/a/16904878/338811
		fg.set(xscale='symlog', yscale='symlog')
		for ax in fg.figure.axes:
			ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
			ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		fg.set_axis_labels(labels[x], labels[y])
		save(fg.figure, fname)
