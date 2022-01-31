"""Text processing."""
# std lib
import os
import re
import sys
import json
import gzip
import html
import itertools
import subprocess
from glob import glob
from collections import Counter

# pip
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
import scipy.sparse
import joblib
from nltk import ngrams
import cld3
from tqdm import tqdm
import SRP
from diff_match_patch import diff  # pylint: disable=no-name-in-module
from pattern.nl import sentiment  # pylint: disable=E0611,E0401

# this project
from files import TEIFILES

PBPATTERN = re.compile(r'<pb n="([^"]*)".*?/>')  # A TEI page-beginning tag
# Any TEI tag introducing a paragraph: paragraph, line of poem, quote, heading
PARPATTERN = re.compile(r'<(?:p|l|q|head)\b.*?>')
# trim whitespace, but re-insert linebreaks after </p> and <lb/>
ENDPARPATTERN = re.compile(r'</(?:p|l|q|head)>')
PARATEXT = re.compile(r'<cf>.*?</cf>')
TAGPATTERN = re.compile(r'^.*?<body>|-<pb.*?>|<.*?>', flags=re.DOTALL)
# Ensure that an em-dash ("gedachtenstreepje") at the end of a page
# is surrounded by spaces, such that it is not interpreted as
# hyphen ("afbreekstreepje").
DASHPATTERN = re.compile(r' -(<pb.*?>)')
SPACEPATTERN = re.compile(r'[\n\xa0]+')
SPACEPATTERN2 = re.compile(r'(?:\n *)+')

LINEPATTERN = re.compile(r'\S.*?\n')  # a line in a plain/tokenized text file
TOKENPATTERN = re.compile(r'(\S+)\s')  # pattern for token in tokenized text
ALPCHUNKPATTERN = re.compile(  # match chunks in output of Alpino tokenizer
		r'key\|\d+\n'  # chunk header
		r'(.*?\n)'  # match text of chunk
		r'(?=key\||$)',  # lookahead to next chunk or end of string
		flags=re.DOTALL)


def striptei(tei):
	"""Convert a string with (a fragment of) TEI to plain text.

	Avoids lxml overhead. Use at your own risk; may summon the ancient ones:
	https://stackoverflow.com/a/1732454/338811

	Hyphenated words at end of page are dehyphenated.
	Linebreaks are removed, except for paragraph ends and <lb/> elements."""
	unescaped = html.unescape(tei)
	# Ensure that an em-dash ("gedachtenstreepje) at the end of a page
	# is surrounded by spaces, such that it is not interpreted as
	# hypen ("afbreekstreepje").
	fixdash = DASHPATTERN.sub(r' - \1', unescaped)
	# trim whitespace, but re-insert linebreaks at paragraph ends
	fixlinebreaks = ENDPARPATTERN.sub('\\g<0>\n',
			SPACEPATTERN.sub(' ', fixdash))
	result = SPACEPATTERN2.sub('\n', TAGPATTERN.sub('', fixlinebreaks))
	return result


def readtei(fname):
	"""Read TEI, strip XML, trim whitespace, and return plain text string."""
	with open(fname, encoding='utf8') as inp:
		tei = inp.read()
	trimpara = PARATEXT.sub('', tei)  # Remove paratext not by author
	return striptei(trimpara)


def readteichunked(fname, threshold=1000):
	"""Version of readtei() that splits TEI into chunks of paragraphs.

	:param threshold: the threshold in characters for chunks.
	:yields: tuples `(teifragment, textfragment)`"""
	with open(fname, encoding='utf8') as inp:
		tei = inp.read()
	tei = PARATEXT.sub('', tei)  # Remove paratext not by author
	teibody = tei.index('<body>')
	prev = teibody
	for match in PARPATTERN.finditer(tei, pos=teibody):
		if match.start() - prev > threshold:
			chunk = tei[prev:match.start()]
			yield chunk, striptei(chunk)
			prev = match.start()
	chunk = tei[prev:]
	yield chunk, striptei(chunk)


def readtext(fname):
	"""Read text in TEI format; return tokens per page and sentences with IDs.

	:param fname: filename of a TEI-XML file.
	:returns: a tuple of lists `(tokensperpage, sentswithids)`:
		:`tokensperpage`: a list of tuples `(pageno, tokens)`
			where `pageno` is a str or int, and `tokens` is a list of strings.
		:`sentswithids`: a list of strings of the form
			`parno-sentno-pageno|tokenized sent .`
			NB: `pageno` can be an arbitrary string containing spaces and `-`,
			but not `"`; fortunately, `|` is also not found in the corpus.
	"""

	def pagenoperchar(tei):
		"""Produce mapping of TEI string index to pageno (actually a str)."""
		prev = 0
		pageno = 'NA'  # page ID for unnamed page
		for match in PBPATTERN.finditer(tei, pos=teibody):
			yield from [pageno] * (match.start() - prev)
			pageno = match.group(1)
			prev = match.start()
		yield from [pageno] * (len(tei) - prev)

	def parperchar(tei):
		"""Produce mapping of TEI string index to paragraph number."""
		prev = n = 0
		paragraphs = np.zeros(len(tei), dtype=np.int32) - 1
		for n, match in enumerate(PARPATTERN.finditer(tei, pos=teibody)):
			paragraphs[prev:match.start()] = n
			prev = match.start()
		paragraphs[prev:] = n + 1
		return paragraphs

	def alignchunks(fname):
		"""Apply tokenization and diff on chunks.

		- split TEI into chunks with one or more <p> elements
		- for each chunk: strip TEI, run alpinotokenize
		- run two diff passes on each chunk: TEI stripping, tokenization
		- collect chunks, append diff results into single list

		:returns: a tuple (text, tokenized, tokalignment, textalignment) where
			`text` is the Alpino tokenizer input with one paragraph per line;
			`tokenized` is Alpino tokenizer output with one sentence per line;
			and `textalignment` and `tokalignment` are lists with diff()
			results for `text -> tokenized` and `tei -> text`, respectively.
		"""
		tokenizedchunks, textchunks = [], []
		tokalignment, textalignment = [], []
		for teichunk, textchunk in readteichunked(fname):
			textchunks.append(textchunk)
			textalignment.extend(diff(teichunk, textchunk,
					timelimit=0, checklines=False, cleanup_semantic=False))
		tokinput = ''.join('key|%d\n%s\n' % (n, textchunk)
				for n, textchunk in enumerate(textchunks))
		tokoutput = alpinotokenize(tokinput)
		tokenizedchunks = ALPCHUNKPATTERN.findall(tokoutput)
		for textchunk, tokenizedchunk in zip(textchunks, tokenizedchunks):
			tokalignment.extend(diff(textchunk, tokenizedchunk,
					timelimit=0, checklines=False, cleanup_semantic=False))
		tokenized = ''.join(tokenizedchunks)
		text = ''.join(textchunks)
		return text, tokenized, textalignment, tokalignment

	def alignindices(target, alignment, initidx=0):
		"""Produce a mapping of string indices based on diff output.

		:param target: the result of a string transformation `source -> target`
		:param alignment: output from diff() function.
		:param initidx: the starting index of the source string
		:returns: a mapping s.t. `target[idx] == source[mapping[idx]]`
		"""
		mapping = np.zeros(len(target), dtype=np.int32) - 1
		mapping[0], n, m = 0, initidx, 0
		for op, length in alignment:
			if op == '=':
				mapping[m:m + length] = np.arange(n, n + length, dtype=np.int32)
				n += length
				m += length
			elif op == '+':
				mapping[m:m + length] = mapping[max(0, m - 1)]
				m += length
			elif op == '-':
				n += length
		return mapping

	def gettokensperpage(tokenized, tokenized2tei, pagenos):
		"""Produce list of `(pageno, tokens)` tuples."""
		tokensperpage = []
		curpage = []
		curpageno = 'NA'
		for match in TOKENPATTERN.finditer(tokenized):
			token = match.group(1)
			idx = tokenized2tei[match.start()]
			pageno = pagenos[idx]
			if pageno != curpageno:
				if curpage:
					tokensperpage.append((curpageno, curpage))
					curpage = []
				curpageno = pageno
			curpage.append(token)
		tokensperpage.append((curpageno, curpage))
		return tokensperpage

	def getsentswithids(tokenized, tokenized2tei, pagenos, paragraphs):
		"""Produce list of sentence IDs and tokens separated by `|`."""
		sentswithids = []
		sentno = prevparno = 1
		for match in LINEPATTERN.finditer(tokenized):
			line = match.group()
			idx = tokenized2tei[match.start()]
			pageno = pagenos[idx]
			parno = paragraphs[idx]
			if parno != prevparno:
				sentno = 1
			sentswithids.append('%d-%d-%s|%s' % (parno, sentno, pageno, line))
			sentno += 1
			prevparno = parno
		return sentswithids

	with open(fname, encoding='utf8') as inp:
		tei = inp.read()
	teibody = tei.index('<body>')

	pagenos = np.array(list(pagenoperchar(tei)))
	if len(pagenos) != len(tei):
		raise ValueError('length mismatch: %d %d' % (len(pagenos), len(tei)))

	paragraphs = parperchar(tei)
	if (paragraphs == -1).any():
		raise ValueError('paragraphs incomplete')

	text, tokenized, textalignment, tokalignment = alignchunks(fname)
	tokenized2text = alignindices(tokenized, tokalignment, initidx=0)
	text2tei = alignindices(text, textalignment, initidx=teibody)
	# FIXME: there must be a simple vectorized numpy method for this
	tokenized2tei = np.array(
			[text2tei[n] for n in tokenized2text],
			dtype=np.int32)
	if (tokenized2tei == -1).any():
		raise ValueError('unaligned index in tokenized2tei: %r'
				% (tokenized2tei == -1).nonzero())

	tokensperpage = gettokensperpage(tokenized, tokenized2tei, pagenos)
	if len(tokensperpage) == 1:
		# Apparently this file has no pagebreaks; fall back to creating chunks
		# with a fixed number of words.
		chunklen = 350  # number of words on a typical page
		tokens = tokensperpage[0][1]
		tokensperpage = [(n, tokens[m:m + chunklen])
				for n, m in enumerate(range(0, len(tokens), chunklen))]

	sentswithids = getsentswithids(
			tokenized, tokenized2tei, pagenos, paragraphs)
	if not sentswithids:
		raise ValueError('no text: %r' % fname)

	# if fname == 'data/xml/jowa001claa01_01.xml': raise
	return tokensperpage, sentswithids


def alpinotokenize(text):
	"""Apply Alpino's tokenizer to the string `text`; returns a string.

	Requires a working installation of Alpino in $ALPINO_HOME
	See http://www.let.rug.nl/vannoord/alp/Alpino/versions/binary/

	:param text: a string
	:returns: a string of space-separated tokens, one sentence per line.

	>>> alpinotokenize('Hallo,\\nwereld! Het werkt.'.replace('\\n', ' '))
	'Hallo , wereld !\\nHet werkt .\\n'
	"""
	cmd = subprocess.run(
			executable=os.path.join(
				os.getenv('ALPINO_HOME'), 'Tokenization/tokenize.sh'),
			args=[],
			input=text,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			encoding='utf8', text=True, check=True)
	# FIXME: more error checking? io deadlock possible?
	return cmd.stdout


def runlangdetect(outfile='output/langdetect.tsv'):
	"""Run language detection on TEI files.

	Writes results to `output/langdetect.tsv`.

	>>> import cld3
	>>> for a in (
	...			'Onze Vader die in de hemelen zijt, Uw naam worde geheiligd.',
	...			"Us Heit, dy't yn de himelen is jins namme wurde hillige.",
	...			'Our Father, which art in Heaven, Hallowed be thy Name.'):
	...		print(cld3.get_language(a).language, end=' ')
	nl fy en

	NB: cld3 seems to always return a language, even for nonsense input
	(cld2 does't have this issue).

	>>> for a in ('oeuaoeuoeuaoekaoetnh', 'abcdefghijklmnopqrstuvwxyz',
	...			'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'):
	...		print(cld3.get_language(a))
	LanguagePrediction(language='st', probability=0.8312870264053345,
			is_reliable=True, proportion=1.0)
	LanguagePrediction(language='hu', probability=0.8115301728248596,
			is_reliable=True, proportion=1.0)
	LanguagePrediction(language='ja', probability=0.7837570905685425,
			is_reliable=True, proportion=1.0)
	"""
	if not TEIFILES:
		raise ValueError('no TEI files.')
	# check if we need to re-run language detection
	if (os.path.exists(outfile)
			and list(pd.read_csv(outfile, sep='\t')['Filename'])
				== sorted(os.path.basename(fname) for fname in TEIFILES)
			and os.stat(outfile).st_mtime > max(
				os.stat(fname).st_mtime for fname in TEIFILES)):
		print('no need to run language detection again.', file=sys.stderr)
		return
	results = []
	index = []
	columns = ['language', 'is_reliable', 'probability', 'proportion']
	for fname in tqdm(TEIFILES, 'language detection'):
		index.append(os.path.basename(fname))
		pages, _ = readtext(fname)  # FIXME: use cached output
		text = '\n'.join(' '.join(text) for _pageno, text in pages)
		result = cld3.get_language(text)
		if result:
			results.append([
					result.language,
					result.is_reliable,
					result.probability,
					result.proportion])
		else:
			results.append([None for _ in columns])
	df = pd.DataFrame(results, index=index, columns=columns)
	df.index.name = 'Filename'
	df.to_csv('output/langdetect.tsv', sep='\t')


def strbigrams(tokens):
	"""Return list of space-separated bigrams in sequence.

	:param tokens: an iterable of tokens as strings.
	:returns: a list of bigrams as strings.

	>>> strbigrams(['Hallo', ',', 'wereld', '!'])
	['Hallo ,', ', wereld', 'wereld !']"""
	return [' '.join(gram) for gram in ngrams(tokens, 2)]


def sortcounts(counts):
	"""Sort bag of words by counts (descending) and words (alphabetically).

	A Python `dict` and `Counter` preserves insertion order;
	to avoid the possibility of being able to reconstruct the original text,
	we sort by the counts, breaking ties by sorting words alphabetically.

	:param counts: a mapping of strings to counts.
	:returns: a dictionary of strings to counts, in sorted order.

	>>> sortcounts({'Hallo': 1, ',': 1, 'wereld': 1, '!': 1, 'Het': 1,
	...         'werkt': 1, '.': 1})
	{'!': 1, ',': 1, '.': 1, 'Hallo': 1, 'Het': 1, 'wereld': 1, 'werkt': 1}
	"""
	return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def extractfreqs(pages):
	r"""Return page-level unigram and bigram frequencies of a single text.

	:param pages: a list of pages as extracted by `readtext()`
	:returns: a list of tuples `(pageno, unigrams, bigrams)`

	>>> extractfreqs([(1, 'Hallo , wereld !\nHet werkt .\n'.split())])
	[(1,
		{'!': 1, ',': 1, '.': 1, 'Hallo': 1, 'Het': 1, 'wereld': 1, 'werkt': 1},
		{'! Het': 1, ', wereld': 1, 'Hallo ,': 1, 'Het werkt': 1,
			'wereld !': 1, 'werkt .': 1})]
	"""
	return [(pageno,
			sortcounts(Counter(page)),
			sortcounts(Counter(strbigrams(page))))
			for pageno, page in pages]


def writefreqs(fname, counts, label=None):
	"""Serialize frequencies to given filename.

	:param fname: output format is selected based on filename extension:
		:'jsonl': each line is a JSON structure corresponding to a page.
		:'json': the output is a single, pretty-printed JSON structure.
		:'mallet_tsv': a tab-separated file where each line has a document ID,
			class label, and a sequence of tokens consisting of the elements in
			the bag of words in sorted order, suitable for an LDA topic model.
	:param counts: counts returned by `extractfreqs()`.
	:param label: metadata label to use in Mallet file.
	"""
	name, ext = os.path.splitext(fname)
	myopen = open
	if ext == '.gz':
		myopen = gzip.open
		name, ext = os.path.splitext(name)
	with myopen(fname, 'wt', encoding='utf8') as out:
		if ext == '.jsonl':
			out.writelines(json.dumps(page) + '\n' for page in counts)
		elif ext == '.json':
			json.dump(counts, out, indent=2)
		elif ext == '.mallet_tsv':
			ti_id = os.path.basename(name).rsplit('_', 1)[0]
			for pageno, unigrams, _bigrams in counts:
				out.write('%s_%s\t%s\t%s\n' % (ti_id, pageno, label, ' '.join(
							word for word, count in unigrams.items()
						for _ in range(count))))
		else:
			raise ValueError('unrecognized format: %r' % ext)


def runfreqextraction():
	"""Extract frequencies for the corpus."""
	def getoutfiles(fname):
		name, _ext = os.path.splitext(os.path.basename(fname))
		return ('output/jsonl/%s.jsonl.gz' % name,
				'output/mallet/%s.mallet_tsv.gz' % name,
				'output/tokenized/%s.tok.gz' % name)

	os.makedirs('output/jsonl', exist_ok=True)
	os.makedirs('output/mallet', exist_ok=True)
	os.makedirs('output/tokenized', exist_ok=True)
	if all(os.path.exists(outfile)
			and os.stat(outfile).st_mtime > os.stat(fname).st_mtime
			for fname in TEIFILES
				for outfile in getoutfiles(fname)):
		print('no need to extract frequencies again.', file=sys.stderr)
		return
	for fname in tqdm(TEIFILES, 'frequency extraction'):
		with open(fname, encoding='utf8') as inp:
			year = re.search(r'<edition>.*\b(\d{4})\b.*</edition>', inp.read())
		pages, sentswithids = readtext(fname)
		counts = extractfreqs(pages)
		outfiles = getoutfiles(fname)
		writefreqs(outfiles[0], counts)
		writefreqs(outfiles[1], counts, label=year)
		with gzip.open(outfiles[2], 'wt', encoding='utf8') as out:
			out.writelines(sentswithids)


def runsrpextraction(outfile='output/srp.bin'):
	"""Extract Stable Random Projection vectors (Schmidt, CA 2018)."""
	fnames = {os.path.basename(a): a for a in TEIFILES}
	if (os.path.exists(outfile) and (
			SRP.Vector_file(outfile).to_matrix()['names'] == list(fnames))):
		# FIXME: should also check if text preprocessing has been changed
		print('no need to extract SRP again.', file=sys.stderr)
		return
	hasher = SRP.SRP(1280)
	with SRP.Vector_file(filename=outfile, dims=1280, mode='w') as out:
		for fname in tqdm(fnames, 'SRP extraction'):
			# FIXME: use extracted word frequencies? use original formatted text?
			text = readtei(fnames[fname])
			vector = hasher.stable_transform(text)
			out.add_row(fname, vector)


def runpairwisesimilarity(threshold=0.75):
	"""Use SRP vectors and cosine similarity to detect duplicates.

	:param threshold: similarity threshold; anything higher is stored as
		potential duplicate."""
	with SRP.Vector_file('output/srp.bin') as inp:
		data = inp.to_matrix()
	index, vectors = data['names'], data['matrix']
	sim = pd.DataFrame(pairwise.pairwise_kernels(vectors, metric='cosine'),
			index=index, columns=index, dtype=float).stack().reset_index()
	sim.columns = ['file1', 'file2', 'cosine']
	sim = sim[(sim.cosine > threshold) & (sim.file1 < sim.file2)]
	sim.to_csv('output/similarity.tsv', sep='\t', index=False)


def countscasefold(counts):
	"""Return a lowercased version of counts."""
	lowercasedcounts = Counter()
	for a, b in counts.items():
		lowercasedcounts[a.lower()] += b
	return lowercasedcounts


def runoverallfrequencies(mfw=None, mfbigrams=None):
	"""Sum frequencies per page to obtain frequencies per text.

	`mfw` and `mfbigrams` can be used to limit the vocabulary;
	`None` means no limit."""
	if os.path.exists('output/unigram_counts.npz') and os.path.exists(
			'output/bigram_counts.npz'):
		return
	overallunigrams = Counter()
	overallbigrams = Counter()
	metadata = pd.read_csv(
			'output/metadata.tsv', sep='\t', index_col='DBNLti_id')
	mapping = {fname.split('/')[-1].rsplit('_', 1)[0]: fname
			for fname in glob('output/jsonl/*.gz')}
	for name in tqdm(metadata.index, 'freq. per text'):
		fname = mapping[name]
		with gzip.open(fname, 'rt', encoding='utf8') as inp:
			for line in inp:
				_, unigramcounts, bigramcounts = json.loads(line)
				unigramcounts = countscasefold(unigramcounts)
				bigramcounts = countscasefold(bigramcounts)
				overallunigrams.update(unigramcounts)
				overallbigrams.update(bigramcounts)
	print('%d unigram types; %d bigram types.' % (
			len(overallunigrams), len(overallbigrams)), file=sys.stderr)
	selectedunigrams = {a: n for n, (a, b)
			in enumerate(overallunigrams.most_common(mfw))}
			# if re.search(r'\w', a) is not None
	selectedbigrams = {a: n for n, (a, b)
			in enumerate(overallbigrams.most_common(mfbigrams))}
	spunigrams = scipy.sparse.lil_matrix(
			(len(metadata.index), len(selectedunigrams)), dtype=np.int32)
	spbigrams = scipy.sparse.lil_matrix(
			(len(metadata.index), len(selectedbigrams)), dtype=np.int32)
	for name in tqdm(metadata.index, 'ngrams per text'):
		fname = mapping[name]
		i = metadata.index.get_loc(name)
		unigrams = Counter()
		bigrams = Counter()
		with gzip.open(fname, 'rt', encoding='utf8') as inp:
			for line in inp:
				_, unigramcounts, bigramcounts = json.loads(line)
				for a in unigramcounts.keys() & selectedunigrams.keys():
					unigrams[a] += unigramcounts[a]
				for a in bigramcounts.keys() & selectedbigrams.keys():
					bigrams[a] += bigramcounts[a]
		for n, b in sorted((selectedunigrams[a], b)
				for a, b in unigrams.items()):
			spunigrams[i, n] = b
		for n, b in sorted((selectedbigrams[a], b)
				for a, b in bigrams.items()):
			spbigrams[i, n] = b
	scipy.sparse.save_npz(
			'output/unigram_counts.npz', spunigrams.tocsr(), compressed=True)
	scipy.sparse.save_npz(
			'output/bigram_counts.npz', spbigrams.tocsr(), compressed=True)
	joblib.dump((metadata.index, selectedunigrams, selectedbigrams),
			'output/ngrams_idx_col.pkl')


def runsentimentanalysis():
	"""Apply Pattern sentiment analysis to each page."""
	os.makedirs('output/sentiment', exist_ok=True)
	if all(os.path.exists('output/sentiment/%s.csv'
				% os.path.basename(fname).split('.')[0])
			and os.stat('output/sentiment/%s.csv'
				% os.path.basename(fname).split('.')[0]
				).st_mtime > os.stat(fname).st_mtime
			for fname in TEIFILES):
		print('no need to run sentiment analysis again.', file=sys.stderr)
		return
	for fname in tqdm(glob('output/tokenized/*.tok.gz'), 'sentiment'):
		result = []
		name = os.path.basename(fname).split('.')[0]
		with gzip.open(fname, 'rt', encoding='utf8') as inp:
			for pageno, lines in itertools.groupby(
					inp, lambda x: x.split('|', 1)[0].split('-', 2)[2]):
				text = '\n'.join(line.split('|', 1)[1] for line in lines)
				pol, subj = sentiment(text)
				result.append([pageno, pol, subj])
		pd.DataFrame(result, columns=['pageno', 'polarity', 'subjectivity']
				).to_csv('output/sentiment/%s.csv' % name, index=False)
