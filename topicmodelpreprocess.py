"""Read a set of parsed texts and preprocesses them into Mallet's file format.
The texts are specified in the variable ``TEXTS`` below.
The texts should be parsed by Alpino and converted into the export format
using disco-dop. Each text is divided into chunks of 1000 tokens.
Function words and names are filtered."""
import re
import glob
import random
import pandas as pd
from tqdm import tqdm
from discodop.util import genericdecompressor

# Specify one or more patterns of filenames to use as the corpus.
# Directories and filenames may contain '*', which will be expanded.
TEXTS = ['*.export.zst', ]

# Extract chunks of n tokens from texts;
# each chunk will be a separate 'document'.
CHUNKSIZE = 1000

# A set of words that will be ignored.
STOPWORDS = frozenset(
		'andere deze over zei zal ge niet als daar moet had wel te toch bij '
		'niets dan nog maar dat doch geen worden die een dit der en altijd '
		'haar ze mijn kunnen zonder naar er doen omdat we iemand wezen men '
		'met ja toen om tegen of kon voor iets hier geweest veel op wie zelf '
		'wil wij zo zijn ons het heeft van eens tot heb hem wat was door hun '
		'ook me dus ben zij uw aan hij je werd meer alles reeds af is al ik '
		'uit want in hoe na zou waren nu de kan mij zich hebben u '
		'zoo zyn altyd echter zeer nou weer weêr hadden nie zelfs maer '
		'éen heel moest laten den alleen zeî zóó'.split())

# filter punctuation + names
FILTERPOS = '|'.join(re.escape(a) for a in ('LET()', 'eigen'))
FUNCWORDPOS = {
		'let',  # punctuation
		'vg',   # conjuction
		'lid',  # determiner
		'vnw',  # pronouns
		'tsw',  # interjection
		'vz'    # preposition
		}


def chunks(tokens, chunksize):
	"""Split a list into chunks of ``chunksize`` tokens each."""
	for n in range(0, len(tokens), chunksize):
		yield tokens[n:n + chunksize]


def readtokens_export(filename):
	"""Read file in Negra export format and filter stopwords / names."""
	WORD, LEMMA, TAG, MORPH = 0, 1, 2, 3  # pylint: disable=invalid-name
	with genericdecompressor('zstd', filename) as inp:
		df = pd.read_csv(
				inp, sep='\t', comment='#', quoting=3, header=None,
				usecols=[WORD, LEMMA, TAG, MORPH])
	mask = (df[MORPH].str.contains(FILTERPOS, na=True)
			| df[TAG].isin(FUNCWORDPOS)
			| df[WORD].str.lower().isin(STOPWORDS)
			| df[LEMMA].str.lower().isin(STOPWORDS))
	return df[~mask][WORD]


def dumpcorpus_export(filenames):
	"""Write corpus in Mallet's single file format."""
	with open('chunkedcorpus.txt', 'w', encoding='utf8') as out:
		for filename in tqdm(filenames):
			for n, chunk in enumerate(chunks(
					readtokens_export(filename),
					CHUNKSIZE)):
				out.write('x %s_%03d %s\n' % (filename.split('.')[0], n,
						' '.join(chunk
								).lower()))


def main():
	"""Main."""
	filenames = [a for pattern in TEXTS
				for a in glob.glob(pattern)]
	# shuffle filenames so that similar files are not together due to filenames
	random.shuffle(filenames)
	dumpcorpus_export(filenames)


if __name__ == '__main__':
	main()
