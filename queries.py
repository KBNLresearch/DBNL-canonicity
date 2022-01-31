"""Linked Open Data and other queries."""
import os
import re
import sys
import requests
import pandas as pd

# this project
from files import generated_f


def runwikidataqueries():
	"""Run WikiData queries and store results."""
	# Query for authors in "Canon Nederlandse Letterkunde 2002"
	# FIXME: add documentation for numerical identifiers
	# Paste queries into https://query.wikidata.org/
	canonquery = """SELECT ?item ?itemLabel ?itemDescription ?occupation
		?occupationLabel ?pseudonym ?date_of_death
		?Digitale_Bibliotheek_voor_de_Nederlandse_Letteren_author_ID
	WHERE {
		?item wdt:P31 wd:Q5;
		wdt:P361 wd:Q2773681.
		SERVICE wikibase:label {
			bd:serviceParam wikibase:language "nl,en". }
		OPTIONAL { ?item wdt:P106 ?occupation. }
		OPTIONAL { ?item wdt:P742 ?pseudonym. }
		OPTIONAL { ?item wdt:P570 ?date_of_death. }
		OPTIONAL { ?item wdt:P723
			?Digitale_Bibliotheek_voor_de_Nederlandse_Letteren_author_ID. }
	}
	"""
	authorquery = """
	SELECT ?DBNLpers_id ?item ?itemLabel ?sitelink ?genderLabel
	WHERE {
		?item wdt:P31 wd:Q5;
			wdt:P723 ?DBNLpers_id.
		SERVICE wikibase:label {
			bd:serviceParam wikibase:language "nl,en". }
		OPTIONAL { ?item wdt:P21 ?gender. }
		OPTIONAL { ?sitelink schema:about ?item;
			schema:isPartOf <https://nl.wikipedia.org/>. }
	}
	LIMIT 100000"""
	queries = {canonquery: generated_f('nederlandsecanon2002.tsv'),
			authorquery: generated_f('wp_pers_id.tsv')}
	for query, fname in queries.items():
		if not os.path.exists(fname):
			runwikidataquery(query, fname)


def runwikidataquery(query, outfile):
	"""Run a WikiData query and store results in TSV-file `outfile`."""
	url = 'https://query.wikidata.org/sparql'
	print('Querying %r for:\n%s' % (url, query), file=sys.stderr)
	req = requests.get(url, params={'format': 'json', 'query': query})
	data = req.json()
	columns = data['head']['vars']
	rows = [[binding[col]['value'] if col in binding else None
			for col in columns]
				for binding in data['results']['bindings']]
	wikidata_result = pd.DataFrame(rows, columns=columns)
	wikidata_result = wikidata_result.drop_duplicates('item')
	wikidata_result = wikidata_result.sort_values(wikidata_result.columns[0])
	wikidata_result.to_csv(outfile, sep='\t', index=False)
	print('Wrote results to %r' % outfile, file=sys.stderr)


def runbasisbibliotheekquery(outfile=generated_f('basisbibliotheek.tsv')):
	"""Scrape DBNL website for Basisbibliotheek IDs."""
	if os.path.exists(outfile):
		return
	url = 'https://www.dbnl.org/basisbibliotheek/index.php?o=ca'
	req = requests.get(url)
	ids = re.findall(r'http://www.dbnl.org/tekst/(\w+)_01', req.text)
	with open(outfile, 'w', encoding='utf8') as out:
		out.writelines(a + '\n' for a in ids)


def rundbnlcanonquery(outfile=generated_f('dbnlcanon.tsv')):
	"""Scrape DBNL website for DBNL canon IDs."""
	if os.path.exists(outfile):
		return
	url = ('https://www.dbnl.org/letterkunde/enquete/'
			'enquete_dbnlmnl_21062002.htm')
	start = ('<p class="h3"><b>de Nederlandse literaire canon in honderd '
			'(en enige) werken</b></p>')
	end = ('Elsschots romans <i>Lijmen</i> en<i> Het been</i> werden '
			'dertien keer gezamenlijk genoemd.')
	req = requests.get(url)
	# FIXME: is this page up to date?
	ids = re.findall(r'/tekst/(\w+)_01/',
			req.text[req.text.index(start):req.text.index(end)])
	with open(outfile, 'w', encoding='utf8') as out:
		out.writelines(a + '\n' for a in ids)


def writenames(metadata, outfile=generated_f('dbnl_authors.txt')):
	"""Write author names for queries in Makefile."""
	# These names give too many spurious matches (i.e., false positives):
	filternames = {'Mario', 'Cornelia', 'Wilma', 'Cara', 'Anonymous'}
	authors = sorted(metadata['Author'].append(
			metadata['WPAuthor'], ignore_index=True).dropna().unique())
	authors = [a for a in authors if a not in filternames]
	cur = None
	if os.path.exists(outfile):
		with open(outfile, encoding='utf8') as inp:
			cur = inp.read().splitlines()
	# Avoid writing same file again, because Makefile uses modification time.
	if authors != cur:
		with open(outfile, 'w', encoding='utf8') as out:
			out.writelines(name + '\n' for name in authors)


def runauthorqueries(fname, outfname, queries='output/dbnl_authors.txt'):
	"""Run author queries on zst file `fname` and write to `outfname`."""
	# 'pv' only provides progress info, can be removed
	# write to temporary file so that query can be interrupted without
	# affecting previous version.
	if not os.path.exists(fname):
		print('%r not found.' % fname, file=sys.stderr)
	elif (os.path.exists(outfname)
			and os.stat(outfname).st_mtime > os.stat(fname).st_mtime
			and os.stat(outfname).st_mtime > os.stat(queries).st_mtime):
		print('%r is up to date' % outfname, file=sys.stderr)
	else:
		print('Running author queries on %r' % fname, file=sys.stderr)
		os.system(
				'zstdcat %s '
				'| pv '
				'| LC_ALL=C grep '
				'--fixed-strings '
				'--word-regexp '
				'--ignore-case '
				'--no-filename '
				'--only-matching '
				'--file %s >tmp '
				'&& mv tmp %s' % (fname, queries, outfname))


def readauthormatches(fname, metadata):
	"""Read results of `runauthorqueries` into a Series."""
	# counting matches in Wikipedia 2019 dump
	if os.path.exists(fname):
		with open(fname, encoding='utf8') as inp:
			matches = inp.read().splitlines()
		# FIXME: what to do about case mismatches? fold case?
		counts = pd.Series(matches).value_counts()
	else:
		print('WARNING: %r not found; '
				'run rir.py again to get correct results.' % fname,
				file=sys.stderr)
		counts = pd.Series(dtype=object)
	# for each name variant, use the one with the largest # matches
	return pd.concat([
			counts.reindex(
				metadata['Author'], fill_value=0).reset_index(drop=True),
			counts.reindex(
				metadata['WPAuthor'], fill_value=0).reset_index(drop=True),
			], axis=1).max(axis=1).astype('Int64')
