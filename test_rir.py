"""pytest unit tests."""
# pylint: disable=redefined-outer-name
import os
import pytest
import pandas as pd
import rir
import metadata
import queries
import textprocessing
import plots


@pytest.fixture
def dbnlmetadata():
	"""Load metadata."""
	return metadata.readmetadata()


def test_readmetadata():
	df, _metadata, disjointset = metadata.readmetadata()
	processed = metadata.processmetadata(df, disjointset)
	assert not df['t2_jaar'].isna().any()
	assert len(processed) != 0
	assert not processed['YearFirstPublished'].isna().any()


def test_writemetadata(dbnlmetadata, tmp_path):
	_orig, dbnlmetadata, _disjointset = dbnlmetadata
	metadata.writemetadata(dbnlmetadata, tmp_path)
	df = pd.read_excel('%s/metadata.xlsx' % tmp_path)
	assert len(df)


def test_droptexts(dbnlmetadata):
	# FIXME: this is messy ...
	orig, dbnlmetadata, disjointset = dbnlmetadata
	processed = metadata.processmetadata(orig, disjointset)
	df = metadata.droptexts(processed)
	assert len(df) != 0
	assert len(df) < len(processed)


def test_addcanonicity(dbnlmetadata):
	_orig, dbnlmetadata, _disjointset = dbnlmetadata
	assert dbnlmetadata['AuthorInCanon2002'].any()
	assert dbnlmetadata['TitleInCanon2002'].any()
	assert (dbnlmetadata['AuthorDBRDMatches'] > 0).any()
	assert (dbnlmetadata['AuthorNLWikipedia2019Matches'] > 0).any()
	assert (dbnlmetadata['DBNLSecRefsAuthor'] > 0).any()
	assert (dbnlmetadata['DBNLSecRefsTitle'] > 0).any()


def test_rundbnlcanonquery(path='data/generated'):
	queries.rundbnlcanonquery(outfile='%s/dbnlcanon.tsv' % path)
	df = pd.read_csv('%s/dbnlcanon.tsv' % path)
	assert len(df) > 0


def test_regex():
	text = '<pb n="binnenkant omgevouwen voorplat-voorkant schutblad" ed=""/>'
	expected = 'binnenkant omgevouwen voorplat-voorkant schutblad'
	assert textprocessing.PBPATTERN.match(text).group(1) == expected


def test_readtext_simple(tmp_path):
	tei = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE TEI.2 PUBLIC "-//DBNL//DTD TEI.2 XML//NL">
<TEI.2>
    <teiHeader>
        <fileDesc>
           <titleStmt>
               <title type="main">Test</title>
           </titleStmt>
        </fileDesc>
   </teiHeader>
   <text>
<body>
<pb n="1"/>
<p>Hallo,
wereld! Het werkt.</p>
<p>Deze zin wordt
<pb n="2"/>
bruut onderbroken. Deze zin wordt dat niet.</p>
<p>Deze zin wordt al hele-<pb n="3"/>maal bruut onderbroken.</p>
<p>Dit is nog een streepje -<pb n="4"/>een gedachtestreepje.</p>
</body></text>
</TEI.2>"""
	expected = """\
1-1-1|Hallo , wereld !
1-2-1|Het werkt .
2-1-1|Deze zin wordt bruut onderbroken .
2-2-2|Deze zin wordt dat niet .
3-1-2|Deze zin wordt al helemaal bruut onderbroken .
4-1-3|Dit is nog een streepje - een gedachtestreepje .
"""
	fname = '%s/test.xml' % tmp_path
	with open(fname, 'w', encoding='utf8') as out:
		out.write(tei)
	tokensperpage, sentswithids = textprocessing.readtext(fname)
	assert len(tokensperpage) == 4
	assert len(sentswithids) == 6
	assert tokensperpage[0] == ('1', ['Hallo', ',', 'wereld', '!', 'Het',
			'werkt', '.', 'Deze', 'zin', 'wordt'])
	assert tokensperpage[1] == ('2', ['bruut', 'onderbroken', '.', 'Deze',
			'zin', 'wordt', 'dat', 'niet', '.', 'Deze', 'zin', 'wordt', 'al',
			'helemaal'])
	assert tokensperpage[2] == ('3', ['bruut', 'onderbroken', '.', 'Dit', 'is',
			'nog', 'een', 'streepje', '-'])
	assert tokensperpage[3] == ('4', ['een', 'gedachtestreepje', '.'])
	assert ''.join(sentswithids) == expected


def test_readtext_jowa():
	fname = 'data/xml/jowa001claa01_01.xml'
	pages, sents = textprocessing.readtext(fname)
	assert len(pages) == 155
	pageno, tokens = pages[3]
	assert pageno == '1'
	assert len(tokens) == 167
	assert tokens[:10] == ['Hoofdstuk', 'I.', 'Op', 'school', '.', "'t",
			'Was', 'Maandagmorgen', '.', '-']
	assert tokens[-10:] == ['zag', 'men', 'de', 'meisjes', 'in', 'allerijl',
			'de', 'hoeden', 'van', 'den']
	assert sents[0] == '1-1-I|CLAARTJE\n'
	assert sents[531] == '232-1-55|‘ Hemel Claar , lig jij er nog in ?\n'
	assert sents[856] == '373-2-85|‘ Wat een prachtig gezicht !\n'
	assert sents[997] == '440-2-99|Wat was n.l. het geval geweest ?\n'
	assert sents[1041] == '456-2-104|Tot over iets meer dan vier weken !\n'
	assert sents[1462] == '638-3-146|Is het heusch waar ?\n'
	assert sents[-1] == '650-1-148|einde .\n'
	assert len(sents) == 1493


def test_readtei():
	fname = 'data/xml/jowa001claa01_01.xml'
	text = textprocessing.readtei(fname)
	# note: to get an ascii repr use text.encode('unicode-escape').decode()
	assert text[:100] == ('    CLAARTJE\nCLAARTJE\nEEN VERHAAL VOOR MEISJES\n'
			'DOOR\nA. JOWAL\n:: MET ILLUSTRATIES VAN ::\nGUST. VAN DE')
	assert text[10000:10100] == ('righeid vergeten.’\n‘Hè, ja, maar ik moet U '
			'eerst nog één ding vragen: mag ik vanmiddag uit blijven t')
	assert text[-100:] == (' een beroemd en gevierd schilderes zou zijn, door '
			'ieder bemind om haar liefheid en goedheid.\neinde.\n')


def test_extractfreqs():
	fname = 'data/xml/jowa001claa01_01.xml'
	pages, _ = textprocessing.readtext(fname)
	counts = textprocessing.extractfreqs(pages)
	pageno, unigramcounts, bigramcounts = counts[3]
	assert pageno == '1'
	assert len(unigramcounts) == 119
	assert len(bigramcounts) == 162
	assert unigramcounts == {'de': 14, ',': 8, '.': 6, 'van': 5, 'en': 4,
			'het': 3, '!': 2, '-': 2, 'Op': 2, 'der': 2, 'er': 2, 'in': 2,
			'mooie': 2, 'op': 2, 'te': 2, 'veel': 2, 'was': 2, 'worden': 2,
			'‘': 2, '’': 2, "'t": 1, ';': 1, 'Anders': 1, 'Dat': 1, 'De': 1,
			'Hoofdstuk': 1, 'I.': 1, 'Maandagmorgen': 1, 'Maar': 1, 'Nog': 1,
			'Rembrandtschool': 1, 'Was': 1, 'Zaterdag': 1, 'Ze': 1, 'aan': 1,
			'al': 1, 'allerijl': 1, 'arme': 1, 'beuken': 1, 'bij': 1,
			'bijna': 1, 'bladerlooze': 1, 'bloemen': 1, 'bloempjes': 1,
			'dachten': 1, 'dadelijk': 1, 'dan': 1, 'dankbaarheid': 1, 'den': 1,
			'dien': 1, 'door': 1, 'een': 1, 'eerst': 1, 'enkele': 1,
			'frissche': 1, 'gegroeid': 1, 'gelet': 1, 'genieten': 1,
			'gewichtigers': 1, 'glazen': 1, 'groote': 1, 'hand': 1,
			'heerlijk': 1, 'heldere': 1, 'hier': 1, 'hoeden': 1, 'hoog': 1,
			'hooge': 1, 'hun': 1, 'hé': 1, 'iets': 1, 'kinderen': 1, 'klas': 1,
			'kleuren': 1, 'koesterende': 1, 'komen': 1, 'krokusjes': 1,
			'liedje': 1, 'linden': 1, 'meisjes': 1, 'men': 1, 'merken': 1,
			'met': 1, 'morgen': 1, 'niet': 1, 'nog': 1, 'och': 1, 'om': 1,
			'ons': 1, 'oogenblikken': 1, 'opengaan': 1, 'opwekkend': 1,
			'perken': 1, 'prachtige': 1, 'recht': 1, 'richtten': 1,
			'scheen': 1, 'school': 1, 'sinds': 1, 'speelplaats': 1,
			'takken': 1, 'tuindeuren': 1, 'tulpen': 1, 'vijfde': 1,
			'vinden': 1, 'vogels': 1, 'volop': 1, 'voor': 1, 'voorjaarszon': 1,
			'vroolijk': 1, 'weer': 1, 'wel': 1, 'zag': 1, 'zal': 1, 'zich': 1,
			'zongen': 1, 'zonnestralen': 1, 'zou': 1, 'zullen': 1}
	assert bigramcounts == {'van de': 3, ', en': 2, '. -': 2, '! Ze': 1,
			'! ’': 1, "'t Was": 1, ', dien': 1, ', mooie': 1, ', och': 1,
			', om': 1, ', ‘': 1, ', ’': 1, '- Op': 1, '- ‘': 1, ". 't": 1,
			'. Anders': 1, '. Dat': 1, '. De': 1, '; er': 1, 'Anders zag': 1,
			'Dat was': 1, 'De krokusjes': 1, 'Hoofdstuk I.': 1, 'I. Op': 1,
			'Maandagmorgen .': 1, 'Maar ,': 1, 'Nog enkele': 1, 'Op de': 1,
			'Op school': 1, 'Rembrandtschool scheen': 1,
			'Was Maandagmorgen': 1, 'Zaterdag !': 1, 'Ze zullen': 1,
			'aan de': 1, 'al dadelijk': 1, 'allerijl de': 1,
			'arme bloempjes': 1, 'beuken en': 1, 'bij het': 1,
			'bijna bladerlooze': 1, 'bladerlooze takken': 1, 'bloemen ,': 1,
			'bloempjes ;': 1, 'dachten de': 1, 'dadelijk te': 1,
			'dan komen': 1, 'dankbaarheid voor': 1, 'de Rembrandtschool': 1,
			'de arme': 1, 'de bloemen': 1, 'de hand': 1, 'de heldere': 1,
			'de hoeden': 1, 'de kinderen': 1, 'de koesterende': 1,
			'de meisjes': 1, 'de nog': 1, 'de perken': 1, 'de speelplaats': 1,
			'de vijfde': 1, 'de vogels': 1, 'der groote': 1, 'der hooge': 1,
			'dien morgen': 1, 'door de': 1, 'een liedje': 1, 'eerst recht': 1,
			'en de': 1, 'en linden': 1, 'en tulpen': 1, 'en zal': 1,
			'enkele oogenblikken': 1, 'er niet': 1, 'er was': 1,
			'frissche kleuren': 1, 'gegroeid vinden': 1,
			'gelet worden': 1, 'genieten van': 1, 'gewichtigers aan': 1,
			'glazen tuindeuren': 1, 'groote glazen': 1, 'hand .': 1,
			'heerlijk opwekkend': 1, 'heldere voorjaarszon': 1, 'het hier': 1,
			'het opengaan': 1, 'het prachtige': 1, 'hier eerst': 1,
			'hoeden van': 1, 'hoog op': 1, 'hooge beuken': 1, 'hun mooie': 1,
			'hé ,': 1, 'iets veel': 1, 'in allerijl': 1, 'in de': 1,
			'kinderen van': 1, 'klas ,': 1, 'kleuren richtten': 1,
			'koesterende zonnestralen': 1, 'komen de': 1, 'krokusjes en': 1,
			'liedje van': 1, 'linden .': 1, 'meisjes in': 1, 'men de': 1,
			'merken bij': 1, 'met hun': 1, 'mooie frissche': 1,
			'mooie weer': 1, 'morgen zou': 1, 'niet veel': 1, 'nog bijna': 1,
			'och hé': 1, 'om volop': 1, 'ons wel': 1, 'oogenblikken ,': 1,
			'op ,': 1, 'op de': 1, 'opengaan der': 1, 'opwekkend door': 1,
			'perken met': 1, 'prachtige ,': 1, 'recht vroolijk': 1,
			'richtten zich': 1, 'scheen de': 1, 'school .': 1,
			'sinds Zaterdag': 1, 'speelplaats van': 1, 'takken der': 1,
			'te genieten': 1, 'te merken': 1, 'tuindeuren .': 1,
			'tulpen in': 1, 'van dankbaarheid': 1, 'van den': 1,
			'veel gelet': 1, 'veel gewichtigers': 1, 'vijfde klas': 1,
			'vinden sinds': 1, 'vogels zongen': 1, 'volop te': 1,
			'voor het': 1, 'voorjaarszon heerlijk': 1, 'vroolijk worden': 1,
			'was al': 1, 'was iets': 1, 'weer .': 1, 'wel gegroeid': 1,
			'worden !': 1, 'worden op': 1, 'zag men': 1, 'zal het': 1,
			'zich hoog': 1, 'zongen een': 1, 'zonnestralen ,': 1, 'zou er': 1,
			'zullen ons': 1, '‘ Nog': 1, '‘ dan': 1, '’ Maar': 1,
			'’ dachten': 1}


def test_createtablesfigures(dbnlmetadata):
	_orig, dbnlmetadata, _disjointset = dbnlmetadata
	# if function does not raise an exception, assume output is OK.
	plots.createtablesfigures(dbnlmetadata)
	assert os.path.exists('fig/hist.pdf')


def test_main(tmp_path):
	rir.main(tmp_path)
	assert os.path.exists('%s/metadata.tsv' % tmp_path)
	assert os.path.exists('%s/metadata.xlsx' % tmp_path)
	# etc...
