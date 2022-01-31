"""Metadata."""
# stdlib
import os
import re
import sys
import time
import warnings
from itertools import chain
import json
import requests

# pip
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import DisjointSet

# this project
from queries import readauthormatches
from files import (input_f, generated_f, output_f, JSONDIR,
		DBNLMETADATA, DBNLVERSIONS,
		MANUALMATCHES, OBSUMMARY, OBMETADATA, GGCMETADATA, GGCSUMMARY,
		PPNLINKS, OZPCLUSTERS, REGISTER, TEIFILES)


def readexcel(fname, sheet_name=0, to_pqt=True):
	"""Read excel file; transparently caches to parquet for fast loading."""
	name, _ = os.path.splitext(os.path.basename(fname))
	pqtfname = generated_f(name + '.pqt')
	if (os.path.exists(pqtfname)
		and os.stat(pqtfname).st_mtime > os.stat(fname).st_mtime):
		return pd.read_parquet(pqtfname)
	# https://stackoverflow.com/a/66749978/338811
	with warnings.catch_warnings(record=True):
		warnings.simplefilter('always')
		df = pd.read_excel(fname, engine='openpyxl', sheet_name=sheet_name)
	if to_pqt:
		df.to_parquet(pqtfname, engine='pyarrow', index=False)
	return df  # NB! parquet stores NaN values as None.


def obtain_json(dbnl_id, cache=True):
	"""Obtain DBNL Metadata from API for given id

	:param dbnl_id: the identifier of the publication to obtain data on
	:param cache: whether to store the json metadata to file
	:returns: a json represntation of the DBNL Metadata for that id"""
	fname = os.path.join(JSONDIR, dbnl_id + '.json')
	if cache and os.path.exists(fname):
		with open(fname, 'r', encoding='utf8') as inp:
			json_content = json.load(inp)
	else:
		api = 'https://dbnl.org/extern/api/titel/read_one.php'
		response = requests.get(api, params=[('ti_id', dbnl_id)])
		json_content = response.json()
		if cache:
			with open(fname, 'w', encoding='utf8') as out:
				out.write(response.text)
	return json_content


def get_genrestring(dbnl_id, subgenre=False):
	"""Obtain list of genres in string format

	:param dbnl_id: the identifier of the publication to obtain genres for
	:param subgenre: whether to obtain subgenres or rather genres
	:returns: A comma-separated list of (sub)genres"""
	json_content = obtain_json(dbnl_id)
	try:
		genrelist = [genrename
				for (genre_id, genrename)
				in json_content[('subgenre' if subgenre else 'genre')]]
	except KeyError:   # (sub)genre not in json
		genrelist = []
	return ','.join(genrelist)


def readmetadata():
	"""Load metadata and return rows selected for the corpus.

	:returns: A tuple of two DataFrames and a DisjointSet
		`(orig, processed, disjointset)` where `orig` has (mostly) the original
		metadata columns, and `processed` has renamed columns and additional,
		computed columns. Both DataFrames contain only rows of novels that were
		selected for the corpus."""

	df = readexcel(DBNLMETADATA)  # main metadata table
	ver = readexcel(DBNLVERSIONS)  # information about duplicate titles
	# Extract e.g., "ca. 1920-1930" as integer 1920
	df['jaar'] = df['jaar'].apply(extractyear).astype('Int16')
	df['jaar_geboren'] = df['jaar_geboren'].apply(extractyear).astype('Int16')
	df['jaar_overlijden'] = df['jaar_overlijden'].apply(
			extractyear).astype('Int16')
	df['vrouw'] = df['vrouw'].astype('Int8')
	# Construct full name field
	df['naam'] = (df['voornaam'].fillna('')
			.str.cat(df['voorvoegsel'].fillna(''), sep=' ')
			.str.cat(df['achternaam'].fillna(''), sep=' ')
			.str.replace('  ', ' ')
			.str.strip())
	df.loc[df['voornaam'] == 'anoniem', 'naam'] = 'anonymous'

	df['genre'] = df.apply(
			lambda row: get_genrestring(row.ti_id, subgenre=False), axis=1)
	df['subgenre'] = df.apply(
			lambda row: get_genrestring(row.ti_id, subgenre=True), axis=1)
	df['is_translated'] = df.apply(
			lambda row: 'vertaling' in row.subgenre, axis=1)

	verz_index = df.subgenre.apply(lambda x: 'verzameld werk' in x)
	verz = set(df.loc[verz_index, 'ti_id'])

	# A Disjoint-Set datastructure to keep track of multiple versions of works.
	# NB: for a ti_id without multiple versions, there's no corresponding
	# row in `ver`.
	disjointset = DisjointSet(
			set(ver.t_ti_id) | set(ver.t2_ti_id) | set(df.ti_id))
	for a, b in zip(ver['t_ti_id'], ver['t2_ti_id']):
		if a in verz or b in verz:  # skip collections
			continue
		disjointset.merge(a, b)

	# Extract year; Int16 ensures we can have integers with missing values
	ver['t_jaar_int'] = ver.t_jaar.apply(extractyear).astype('Int16')
	ver['t2_jaar_int'] = ver.t2_jaar.apply(extractyear).astype('Int16')

	ggcsummary = summarizeggcmetadata(linkppns())
	# Add GGC metadata on which books were originally writtin in Dutch
	df['dutch'] = ggcsummary.set_index('dbnl_id')['dutch'].reindex(df['ti_id']
			).reset_index(drop=True)
	# Replace the year of the edition with the year of first publication.
	# Keep earliest year of publication across several metadata sources.
	a = ver.loc[:, ['t_ti_id', 't2_jaar_int']
			].rename(columns={'t_ti_id': 'ti_id'})
	b = ver.loc[:, ['t2_ti_id', 't2_jaar_int']
			].rename(columns={'t2_ti_id': 'ti_id'})
	c = df.loc[:, ['ti_id', 'jaar']
			].rename(columns={'jaar': 't2_jaar_int'})
	ggcyear = ggcsummary.loc[:, ['dbnl_id', 'year']].rename(
			columns={'dbnl_id': 'ti_id', 'year': 't2_jaar_int'})
	combined = pd.concat([a, b, c, ggcyear], ignore_index=True,
			).sort_values(by=['ti_id', 't2_jaar_int']
			).loc[:, ['ti_id', 't2_jaar_int']
			].dropna(axis=0
			).drop_duplicates('ti_id', keep='first')
	df['t2_jaar'] = combined.set_index('ti_id'
			)['t2_jaar_int'].reindex(df['ti_id']).reset_index(drop=True)

	df = droptitles(df, ver, verz)
	orig = df
	processed = droptexts(processmetadata(orig, disjointset))
	return orig, processed, disjointset


def sumlibrarydata(ppnlinks):
	"""Load metadata from public libraries and link to DBNL-metadata.

	Sum number of holding (collection) and lendings (transactions)
	per DBNL-title (through PPN-DBNL-id mapping ppnlinks)
	Write dataframe to csv summary_file and return

	:param ppnlinks: A DataFrame with columns [dbnl_id, ppn] that maps
		dbnl_ids to PPNs of different publications
	:returns: A DataFrame with columns [dbnl_id, holding, lending]"""
	if os.path.exists(OBSUMMARY):
		return pd.read_csv(OBSUMMARY, sep='\t')
	holdings = readexcel(OBMETADATA, sheet_name='Collectie', to_pqt=False
			).groupby('PPN')['Aantal'].sum()
	lending = readexcel(OBMETADATA, sheet_name='Transacties', to_pqt=False
			).groupby('PPN')['Aantal'].sum()
	summary = pd.DataFrame({
			'dbnl_id': ppnlinks.set_index('ppn').dbnl_id,
			'holding': holdings.reindex(ppnlinks.ppn, fill_value=0),
			'lending': lending.reindex(ppnlinks.ppn, fill_value=0)}
			).groupby('dbnl_id').sum()
	summary.to_csv(OBSUMMARY, sep='\t')
	return summary


def summarizeggcmetadata(ppnlinks):
	"""Load metadata from GGC and link to DBNL-metadata.
	Obtain year of first publication and whether publication is Dutch,
	plus annotations, per DBNL-title (through PPN-DBNL-id mapping ppnlinks)

	:param ppnlinks: A DataFrame with columns [dbnl_id; ppn] that maps
		dbnl_ids to PPNs of different publications
	:returns: A DataFrame with columns [dbnl_id, 'year', 'dutch', 'langannot']
	"""
	def getfields(dbnl_id):
		ppns = ppnlinks.loc[ppnlinks.dbnl_id == dbnl_id, 'ppn']
		jvu_van = ggcmetadata.loc[ggcmetadata.ppn.isin(ppns), 'jvu_van']
		min_jvu = np.nan
		if len(jvu_van):
			jvu_van = jvu_van.str.replace('X', '5')
			min_jvu = min(pd.to_numeric(jvu_van))
		lang = ggcmetadata.loc[ggcmetadata.ppn.isin(ppns)]
		anno = ''
		pub = lang['taal_publ_1'].dropna().unique()
		dutch = np.nan  # No information available
		if len(pub):
			original = lang['taal_orig_1'].dropna().unique()
			if len(original) == 0:  # no translations
				dutch = int('ned' in pub)  # originally Dutch
				if len(pub) > 1 or pub[0] != 'ned':
					anno += 'origineel_' + '_'.join(pub)
					if len(lang.loc[~lang['taal_publ_2'].isna()]) > 0:
						anno += '_tweetaligUitgegeven_' + '_'.join(
								lang['taal_publ_2'].dropna().unique())
			else:
				dutch = int('ned' in original)  # translated from Dutch
				anno += '_'.join(original) + '_vertaaldNaar_' + '_'.join(pub)
				if len(lang.loc[~lang['taal_orig_2'].isna()]) > 0:
					anno += '_tweetaligOrigineel_' + '_'.join(
							lang['taal_orig_2'].dropna().unique())
		return dbnl_id, min_jvu, dutch, anno

	if os.path.exists(GGCSUMMARY):
		return pd.read_csv(GGCSUMMARY, sep='\t', dtype={
				'year': 'Int16', 'dutch': 'Int16'})
	ggcmetadata = readexcel(GGCMETADATA)
	result = pd.DataFrame(
			[getfields(dbnl_id) for dbnl_id in ppnlinks.dbnl_id.unique()],
			columns=['dbnl_id', 'year', 'dutch', 'langannot']
			)
	result = result.astype(
			dtype={'year': 'Int16', 'dutch': 'Int16'})
	result.to_csv(GGCSUMMARY, sep='\t', index=False)
	return result


def linkppns():
	"""Obtain mapping to PPNs (publication identifiers) for DBNL titles,
	based on OZP clusters and manual matches.

	:returns: A DataFrame with columns [dbnl_id, ppn] that maps dbnl_ids
		to PPNs of different publications
	"""
	def get_ozp_versions(ppn):
		"""Get the PPNs of all publications that belong to the same
		cluster as the given ppn."""
		try:
			cid = ozpclustersbyppn.at[ppn, 'cluster_id']
		except KeyError:
			return set()
		return set(ozpclustersbycluster.loc[cid, 'ppn'])

	def get_versions(dbnl_row):
		"""Return a list of PPNs that are versions of the title that the
		row describes"""
		version_ppns = set()
		ti_id = dbnl_row.ti_id
		for ppn in [dbnl_row.ppn, dbnl_row.ppn_o]:
			if ppn is not None:
				version_ppns |= get_ozp_versions(ppn)
		version_ppns |= set(manual_matches.loc[
				manual_matches.index  # pylint: disable=E1101
				== dbnl_row.ti_id, 'PPN'])
		return {(ti_id, ppn) for ppn in version_ppns}

	if os.path.exists(PPNLINKS):
		return pd.read_csv(PPNLINKS, sep='\t')
	start = time.perf_counter()
	print('Linking PPNs;', end='', file=sys.stderr)
	dbnlmetadata = readexcel(DBNLMETADATA).loc[
			:, ['ti_id', 'ppn', 'ppn_o']].drop_duplicates(subset=['ti_id'])
	# csv file with PPNs for DBNL ids (DBNL_id; PPN; notes)
	manual_matches = pd.read_csv(
			MANUALMATCHES, sep=';', index_col=0, usecols=[0, 1])
	# csv file that lists OZP clusters (cluster_id; ppn)
	ozpclusters = pd.read_csv(OZPCLUSTERS, sep=';')
	ozpclustersbycluster = ozpclusters.set_index('cluster_id')
	ozpclustersbyppn = ozpclusters.set_index('ppn')  # pylint: disable=E1101
	versions = pd.DataFrame(
			chain.from_iterable(dbnlmetadata.apply(get_versions, axis=1)),
			columns=['dbnl_id', 'ppn']).drop_duplicates()
	# write obtained mapping (dbnl_id; ppn)
	versions.to_csv(PPNLINKS, sep='\t', index=False)
	print('took %.3gs' % (time.perf_counter() - start), file=sys.stderr)
	return versions


def droptitles(df, ver, verz):
	"""Use various metadata criteria to exclude titles.

	Criteria for inclusion in the corpus:

		- fictional prose text of a certain length (novels or novellas).
		- Dutch-language texts originally written in Dutch (no translations).
		- written by a single author.

	:param df: DataFrame of an excel sheet with metadata.
	:param ver: DataFrame of an excel sheet with information about
		duplicate titles."""
	print(len(df), 'before dropping texts', file=sys.stderr)

	# drop books with multiple authors
	df = df[df['ti_id'].isin(
			(df.groupby('ti_id')['pers_id'].nunique() == 1).index)]
	print(len(df), 'texts with one author', file=sys.stderr)

	# drop books with multiple novels
	df = df[~df['ti_id'].isin(verz)]
	print(len(df), 'texts with one novel', file=sys.stderr)

	# drop translated titles
	translatedtitles = df.loc[
			df.dutch == 0,
			'ti_id'].unique()
	df = df[~df.ti_id.isin(translatedtitles)]
	print(len(df), 'after dropping translated titles', file=sys.stderr)

	# known non-Dutch authors
	foreignauthors = {'twai001', 'vern002', '_ver002', 'dele035', 'swif001',
			'carr021', 'beec028', 'alco001', 'lage022', '_rom006'}
	df = df[~df.pers_id.isin(foreignauthors)]
	print(len(df), 'after dropping known foreign authors', file=sys.stderr)

	# keep only one instance for other duplicate titles
	# (can be due to multiple genres)
	# NB: we arbitrarily keep the first instance.
	df = df.drop_duplicates('ti_id')
	print(len(df), 'unique titles in metadata', file=sys.stderr)

	# drop texts whose first edition is from before 1800
	too_old = ver[ver.t_ti_id.isin(df.ti_id) & (ver.t2_jaar_int < 1800)]
	df = df[~df.ti_id.isin(too_old.t_ti_id)]
	print(len(df), 'first edition >= 1800', file=sys.stderr)
	# df = df[~df.ti_id.isin(too_old.t2_ti_id)]
	# print(len(df), 'first edition t2_ti_id >= 1800', file=sys.stderr)

	# drop older editions s.t. we only have the latest edition of each text
	# (NB: it is important to leave out collections for this step)
	df = df[~df.ti_id.isin(
			ver[ver.t_ti_id.isin(df.ti_id)
				& (ver.t2_jaar_int >= 1800)
				].t2_ti_id)]
	print(len(df), 'latest editions', file=sys.stderr)

	# NB: duplicate texts will remain, due to missing metadata;
	# For example, 'coup002boek01' and ('coup002boek04', 'coup002boek05')
	# are the same texts, but not listed as such.
	# On the other hand, 'coup002boek04' and ('coup002boek06', 'coup002boek07')
	# are (correctly) listed as the same texts.
	# See `runpairwisesimilarity()`.

	# IMPORTANT: reset the index, we want the index to be a plain continuous
	# range of integers s.t. "df[newcolumn] = series" does not give surprising
	# results (given that series also has such an index)
	df = df.reset_index(drop=True)
	return df


def droptexts(df):
	"""Drop texts based on content analysis of TEI files."""
	# remove texts for which we (currently) do not have the full text:
	fnames = {os.path.basename(fname) for fname in TEIFILES}
	df = df[df['Filename'].isin(fnames)]
	print(len(df), 'TEI available', file=sys.stderr)

	# drop texts detected as non-Dutch
	lang = pd.read_csv(output_f('langdetect.tsv'), sep='\t')
	dutch = lang.loc[
			lang['is_reliable']  # pylint: disable=E1136
				& (lang['language'] == 'nl'),  # pylint: disable=E1136
			'Filename']
	df = df[df['Filename'].isin(dutch)]
	print(len(df), 'language detection', file=sys.stderr)

	# Heuristically detected duplicates based on Stable Random Projections
	similarity = pd.read_csv(output_f('similarity.tsv'), sep='\t')
	# based on distance threshold of 0.12 in Schmidt (2018).
	# NB: not detecting part-whole relationship of volumes
	similarity = similarity[similarity.cosine > 0.88]
	df = df[~df.Filename.isin(similarity[
			similarity.file1.isin(df.Filename)].file2)]
	print(len(df), 'duplicate detection', file=sys.stderr)

	# IMPORTANT: reset the index, we want the index to be a plain continuous
	# range of integers s.t. "df[newcolumn] = series" does not give surprising
	# results (given that series also has such an index)
	return df.reset_index(drop=True)


def processmetadata(df, disjointset):
	"""Create a processed version of the metadata.

	:param df: a DataFrame `orig` as returned by readmetadata(),
		or a deduplicated concatenation of several such DataFrames.
	:param disjointset: datastructure with IDs of equivalent texts.
	:returns: a DataFrame with rows corresponding to texts selected for the
		corpus; columns are metadata selected or computed for the corpus."""
	# Sort by year of publication, break ties by last name, ti_id.
	df = df.sort_values(['t2_jaar', 'achternaam', 'ti_id'])
	# reset index to continuous range of integers
	df = df.reset_index(drop=True)
	# Columns in the original metadata
	originalcolumns = {
			'ti_id': 'DBNLti_id',
			'pers_id': 'DBNLpers_id',
			't2_jaar': 'YearFirstPublished',
			'jaar': 'YearEditionPublished',
			'druk': 'Edition',
			'vrouw': 'Woman',
			'jaar_geboren': 'Born',
			'jaar_overlijden': 'Died',
			'geb_plaats': 'AuthorOrigin',
			'geb_land_code': 'DBNLgeb_land_code',
			'genre': 'DBNLgenre',
			'subgenre': 'DBNLsubgenre',
			'naam': 'Author',
			'titel': 'Title',
			# 'voornaam', 'voorvoegsel', 'achternaam',
			# 'categorie',
			'geplaatst': 'Filename',
		}
	df = df.rename(columns=originalcolumns)

	# Add our computed columns
	computed = addcomputedcolumns(df, disjointset)

	df['Filename'] = df['Filename'].str.cat(['.xml'] * len(df))
	df = df.loc[:, list(originalcolumns.values()) + computed]
	return df


def addcomputedcolumns(df, disjointset):
	"""Add columns related to canonicity to the DataFrame.

	:param df: a DataFrame `orig` as returned by readmetadata(),
		or a deduplicated concatenation of several such DataFrames.
	:param disjointset: datastructure with IDs of equivalent texts.
	:returns: A list of column names that were added to `df` (in-place)."""
	columns = []

	name = 'ti_id_set'
	columns.append(name)
	df[name] = [','.join(sorted(disjointset.subset(ti_id)))
			for ti_id in df['DBNLti_id']]

	wppersid = pd.read_csv(
			generated_f('wp_pers_id.tsv'),
			sep='\t', index_col='DBNLpers_id')
	wppersid = wppersid[~wppersid.index.duplicated(keep='first')]
	name = 'WPAuthor'
	columns.append(name)
	df[name] = wppersid['itemLabel'].reindex(
			df['DBNLpers_id'], fill_value=None).reset_index(drop=True)

	canondf = pd.read_csv(generated_f('nederlandsecanon2002.tsv'), sep='\t')
	dbnl_id_col = 'Digitale_Bibliotheek_voor_de_Nederlandse_Letteren_author_ID'
	# Drop authors that were not alive after 1800
	# NB: we don't filter by >= 1800, because that would remove authors with
	# missing dates.
	canondf = canondf[~(canondf.date_of_death < '1800')]
	# remove duplicate rows
	canon_pers_id = canondf[dbnl_id_col].drop_duplicates()
	# 108 authors
	name = 'AuthorInCanon2002'
	columns.append(name)
	df[name] = df['DBNLpers_id'].isin(canon_pers_id).astype('Int8')

	# 125 titles
	dbnlcanon = pd.read_csv(
			generated_f('dbnlcanon.tsv'), sep='\t', header=None)[0]
	dbnlcanon = {a for ti_id in dbnlcanon
			for a in (disjointset.subset(ti_id)
				if ti_id in disjointset else {ti_id})}
	name = 'TitleInCanon2002'
	columns.append(name)
	df[name] = df['DBNLti_id'].isin(dbnlcanon).astype('Int8')

	# 1000 titles in "Basisbibliotheek"
	basisbib = pd.read_csv(generated_f('basisbibliotheek.tsv'),
			sep='\t', header=None)[0]
	basisbib = {a for ti_id in basisbib
			for a in (disjointset.subset(ti_id)
				if ti_id in disjointset else {ti_id})}
	name = 'InBasisbibliotheek2008'
	columns.append(name)
	df[name] = df['DBNLti_id'].isin(basisbib).astype('Int8')

	# counting matches in 110kDBRD (Hebban reviews)
	name = 'AuthorDBRDMatches'
	columns.append(name)
	df[name] = readauthormatches(output_f('dbrdmatches.txt'), df)

	# counting matches in Wikipedia 2019 dump
	name = 'AuthorNLWikipedia2019Matches'
	columns.append(name)
	df[name] = readauthormatches(output_f('wpmatches.txt'), df)
	# TODO: query Wikipedia for number of edits, links, references
	# for the page of an author or book.

	# counting DBNL secondary literature references
	secrefs = pd.read_csv(input_f(
			'dbnl_secondaryreferences_titeltabel.csv.gz'))
	sec_count = secrefs.ti_id.value_counts()
	name = 'DBNLSecRefsAuthor'
	columns.append(name)
	df[name] = sec_count.reindex(
			df['DBNLpers_id'], fill_value=0
			).reset_index(drop=True).astype('Int64')
	name = 'DBNLSecRefsTitle'
	columns.append(name)
	# consider secondary references of all versions
	df[name] = [
			sum(sec_count.get(a, 0)
				for a in disjointset.subset(ti_id))
			if ti_id in disjointset else 0
			for ti_id in df['DBNLti_id']]
	# TODO: look at types of secondary reference;
	# a review or close reading in a literary journal is evidence of
	# canonicity, while other types of references might better be ignored.
	# TODO: use year field of secondary reference;
	# e.g., references across a long time span are stronger evidence of
	# canonicity than a set of reviews right after publication.

	# library holdings and lending data
	lib = sumlibrarydata(linkppns()).set_index('dbnl_id')
	for name in lib.columns:
		columns.append(name)
		df[name] = lib[name].reindex(df['DBNLti_id'], fill_value=0
				).reset_index(drop=True).astype('Int64')

	# mentions of authors in GNT literary textbooks
	# https://doi.org/10.22148/16.046
	register = pd.read_excel(REGISTER)
	persons = register[register['Type'] == 'Person']
	splitnames = persons.Name.str.split(', ')
	last = splitnames.str[0].str.strip()
	first = splitnames.str[1].str.strip()
	names = first.str.cat(last, sep=' ')
	occurrences = pd.Series(dict(zip(names, persons['Occurrences'])))
	occurrences = occurrences[occurrences.index.notnull()]
	name = 'GNTpages'
	columns.append(name)
	# Consider both name variants
	a = occurrences.reindex(df['Author'], fill_value=0
			).reset_index(drop=True)
	b = occurrences.reindex(df['WPAuthor'], fill_value=0
			).reset_index(drop=True)
	df[name] = pd.DataFrame([a, b]).T.max(axis=1)

	return columns


def extractyear(yearstr):
	"""Extract year and return integer or None.

	>>> extractyear('15de/16de eeuw')
	1400
	>>> extractyear('ca. 1900')
	1900
	>>> [extractyear(a) for a in ('', '?', 'xx', float('nan'))]
	[None, None, None, None]
	"""
	# Check for NaN
	if (yearstr == '?' or yearstr is None
			or yearstr != yearstr):  # pylint: disable=R0124
		return None
	elif isinstance(yearstr, (int, float)):
		return int(yearstr)
	m = re.search('([0-9]{3,4})', yearstr)
	if m is not None:
		return int(m.group(1))
	m = re.match(
			r'(?:begin\ |einde?\ |ca\.\ )?'
			r'([0-9]{1,2})(?:de|ste)'
			r'(?:[/-][0-9]{1,2}(?:de|ste))?'
			r'\ eeuw',
			yearstr,
			flags=re.VERBOSE)
	if m is not None:
		return int(m.group(1)) * 100 - 100
	return None


def rundbnlmetadataconversion(
		outfile=generated_f('dbnl_metadata_x_titeltabel.pqt')):
	"""Convert CSV metadata to parquet file w/selected, typed columns."""
	# NB: the parquet file is a temporary, generated file used for efficiency;
	# the format may change across Pandas versions. The original CSV remains
	# the reference version.
	infile = input_f('dbnl_metadata_x_titeltabel.csv.gz')
	if (os.path.exists(outfile)
			and os.stat(infile).st_mtime < os.stat(outfile).st_mtime):
		return
	print('Reading %r; this will take a minute and lots of RAM.' % infile,
			file=sys.stderr)
	metadata = pd.read_csv(
			infile,
			sep=';', engine='c', na_values=['NULL'], keep_default_na=False,
			usecols=['ti_id', 'type', 'value', '_jaar', 'genre_bk',
					'genre2_bk', 'subgenre', 'categorie'],
			dtype={
				'ti_id': 'object',
				'type': 'category',
				'value': 'object',
				'_jaar': 'int16',
				'genre_bk': 'category',
				'genre2_bk': 'category',
				'subgenre': 'category',
				'categorie': 'category',
			})
	# this table is a join of multiple tables and therefore contains
	# multiple rows for titles.
	# there will be exact duplicate rows because we don't select all columns.
	metadatanodup = metadata.drop_duplicates()
	metadatanodup.to_parquet(outfile)
	print('Wrote %r' % outfile, file=sys.stderr)


def writemetadata(df, path):
	"""Write metadata in tsv and excel format to given path."""
	df.to_csv('%s/metadata.tsv' % path, sep='\t', index=False)
	with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
		'%s/metadata.xlsx' % path, engine='xlsxwriter') as writer:
		# disable ugly header style ...
		pd.io.formats.excel.ExcelFormatter.header_style = None
		df.to_excel(writer, index=False, sheet_name='Sheet1')
		worksheet = writer.sheets['Sheet1']
		# widths in "character units".
		widths = [15, 10, 7, 7, 10, 7, 7, 7, 20, 7, 7,
				10, 30, 40, 20, 10, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
		if len(widths) != len(df.columns):
			raise ValueError('Column widths incomplete; '
					'%d widths, %d columns' % (len(widths), len(df.columns)))
		for n, width in enumerate(widths):
			worksheet.set_column(n, n, width)
