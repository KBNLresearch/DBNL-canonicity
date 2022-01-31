"""Web interface for applying predictive models to given sample of text."""
# stdlib
from __future__ import print_function, absolute_import
import io
import sys
import logging
from base64 import b64encode
# data science
import matplotlib
matplotlib.use('SVG')
import numpy as np
import scipy.sparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.transform import factor_cmap, factor_mark
from bokeh.embed import components
from sklearn import feature_extraction
from sklearn.metrics.pairwise import linear_kernel
import joblib
import nltk
from marisa_trie import RecordTrie
# Flask & co
from flask import (Flask, make_response, request, render_template,
		send_from_directory)
from flask_caching import Cache

DEBUG = False  # when True: enable debugging interface
APP = Flask(__name__)
STANDALONE = __name__ == '__main__'
CACHE = Cache(config={
		'CACHE_TYPE': 'SimpleCache',
		'CACHE_DEFAULT_TIMEOUT': 3600,  # cache responses server side for 1hr
		'CACHE_THRESHOLD': 128,  # max no of items to cache
		})
CACHE.init_app(APP)

def abbr(a):
	return a[:97] + '...' if len(a) > 100 else a


@APP.route('/')
@APP.route('/index')
@CACHE.cached()
def index():
	"""Start page where a text can be entered."""
	titles = sorted((a,
			abbr('%s (%d) %s' % (
				MD.at[a, 'Author'],
				MD.at[a, 'YearFirstPublished'],
				MD.at[a, 'Title'])))
			for a in MD.index)
	return render_template(
			'index.html',
			titles=titles)


@APP.route('/results', methods=('GET', ))
@CACHE.cached(query_string=True)
def results():
	"""Results page for text in corpus."""
	if 'id' in request.args:
		ti_id = request.args['id']
		i = ROWNAMES.get_loc(ti_id)
		feat = BIGRAMS[i:i + 1, :BIGRAMLIMIT].toarray()
	else:
		return 'Pick a title from the drop down menu.'
	result = getpredictions(feat, ti_id)
	resp = make_response(render_template('predictions.html', **result))
	resp.cache_control.max_age = 604800
	return resp


@APP.route('/classify', methods=('POST', ))
def classify():
	"""Results page for user entered text."""
	ti_id = None
	if 'text' in request.form and request.form['text']:
		text = request.form['text']
		tokens = tokenize(text.lower())
		feat = extractfeatures(tokens)
	else:
		return 'No text entered.'
	result = getpredictions(feat, ti_id)
	return render_template('predictions.html', **result)


@APP.route('/plot', methods=('GET', ))
@CACHE.cached(query_string=True)
def plot():
	"""Show average frequency over time of a given feature."""
	def fetch(feature):
		if feature in BIGRAMTRIE:
			return pd.DataFrame({
					'feat': BIGRAMS[:, BIGRAMTRIE[feature][0][0]
						].toarray()[:, 0],
					'tot': BIGRAMSSUM},
					index=MD.index)
		elif feature in UNIGRAMTRIE:
			return pd.DataFrame({
					'feat': UNIGRAMS[:, UNIGRAMTRIE[feature][0][0]
						].toarray()[:, 0],
					'tot': UNIGRAMSSUM},
					index=MD.index)

	if 'feature' not in request.args:
		return 'feature is required'
	feature = request.args['feature'].lower()
	try:
		smoothing = int(request.args.get('smoothing', 5))
	except ValueError:
		smoothing = 5
	df = fetch(feature)
	if df is None:
		df = fetch(tokenize(feature))
	if df is None:
		return ('%r is not in the corpus.'
				% feature)
	df['freq'] = (df.feat / df.tot * 100)
	top10 = MD.merge(
			df.rename(columns={'feat': 'count'}),
			left_index=True, right_index=True
			).nlargest(10, 'freq')
	c = df[MD.DBNLSecRefsTitle > 0].groupby(MD.YearFirstPublished).sum()
	nc = df[MD.DBNLSecRefsTitle == 0].groupby(MD.YearFirstPublished).sum()
	yearfreqs = pd.concat({
			# 'all': freqs.groupby(MD.YearFirstPublished).mean(),
			'Canonical': 100 * c['feat'] / c['tot'],
			'Non-canonical': 100 * nc['feat'] / nc['tot'],
			}, axis=1, sort=True)
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.set_xlabel('Year')
	ax.set_ylabel('Frequency (%)')
	smoothingdescr = ('(smoothing: %d)' % smoothing
			) if smoothing else ''
	ax.set_title('%r in DBNL novels %s' % (feature, smoothingdescr))
	csvfile = b64encode(yearfreqs.to_csv(None).encode('utf8')).decode('ascii')
	if smoothing:
		yearfreqs = yearfreqs.interpolate().rolling(
				2 * smoothing + 1, center=True).mean()
	lineplot = b64fig(yearfreqs.plot(ax=ax))
	plt.close(fig)
	resp = make_response(render_template(
			'plot.html', lineplot=lineplot, top10=top10, smoothing=smoothing,
			csvfile=csvfile, feature=feature))
	resp.cache_control.max_age = 604800
	return resp


@APP.route('/favicon.ico')
def favicon():
        """Serve the favicon."""
        return send_from_directory(
				'static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')


def tokenize(text):
	return ' '.join(nltk.word_tokenize(text, language='dutch'))


def extractfeatures(tokens):
	return VEC.transform([tokens]).toarray()


def getpredictions(feat, selected_ti_id):
	freq = MODEL[0].transform(feat).toarray()[0]
	score = MODEL.predict_proba(feat)[0, :].max()
	pred = MODEL.predict(feat)[0]
	features = pd.DataFrame({
				'count': feat[0],
				'weight': MODEL[1].coef_[0],  # * BIGRAMS.std(axis=0),
				'comb': freq * MODEL[1].coef_[0]},
			index=BIGRAMCOLUMNS)
	# features.loc['<INTERCEPT>'] = [
	# 		1, MODEL[1].intercept_[0], MODEL[1].intercept_[0]]
	features = features[features['count'] > 0]
	featlow = features[features['comb'] < 0].nsmallest(100, 'comb').round(3)
	feathigh = features[features['comb'] > 0].nlargest(100, 'comb').round(3)

	# can use linear kernel for cosine similarity due to l2 normalization
	sim = pd.Series(
			linear_kernel(
				BIGRAMSNORMALIZED, MODEL[0].transform(feat))[:, 0],
			index=MD.index, name='similarity').nlargest(10)
	sim = pd.concat([MD.loc[sim.index, :], sim], axis=1)

	histplot = makehistplot(features, MODEL[1].intercept_[0])
	script, div = scatterplot(selected_ti_id, feat, sim)

	return dict(score=score, pred=int(pred),
			featlow=featlow, feathigh=feathigh,
			sim=sim, histplot=histplot,
			script=script, div=div, ti_id=selected_ti_id)


def makehistplot(features, intercept):
	"""Create histogram of feature contributions

	x-axis: count (logarithmic), y-axis: contribution (freq * weight)."""
	# This is not a proper histogram, because the y-axis does not show
	# counts, but the sum of the actual feature contributions.
	# A count histogram is misleading in this case because a large number of
	# features with a very small positive contribution would give the
	# impression that the predicted label should be positive as well.
	fig, ax = plt.subplots(figsize=(5, 2))
	pos = features.loc[features['comb'] > 0, :]
	neg = features.loc[features['comb'] < 0, :]
	neg['comb'] += intercept / len(neg)  # account for intercept
	totalpos = features.loc[features['comb'] > 0, 'comb'].sum()
	totalneg = features.loc[features['comb'] < 0, 'comb'].sum() + MODEL[1
			].intercept_[0]
	# Logarithmic histogram https://stackoverflow.com/a/54529366
	hist, bins = np.histogram(features['count'], bins=20)
	logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
	ax.hist(pos['count'], bins=logbins, weights=pos['comb'],
			histtype='stepfilled', label='pos: %+.3f' % totalpos)
	ax.hist(neg['count'], bins=logbins, weights=neg['comb'],
			histtype='stepfilled', label='neg: %.3f' % totalneg)
	ax.set_xscale('log')
	ax.set_xlabel('Feature count')
	ax.set_ylabel('Contribution')
	a, b = ax.get_ylim()
	ax.set_ylim((-max(abs(a), abs(b)), max(abs(a), abs(b))))
	ax.legend(loc='best')
	ax.xaxis.set_major_formatter(ScalarFormatter())
	for item in [ax.title, ax.xaxis.label, ax.yaxis.label
			] + ax.get_xticklabels() + ax.get_yticklabels():
		item.set_fontsize(8)
	fig.subplots_adjust(bottom=0.2, top=0.95, right=0.95)
	result = b64fig(ax)
	plt.close(fig)
	return result


def makehistplot_prev(features, intercept):
	"""Create histogram of feature contributions.

	x-axis: contribution, y-axis: contribution."""
	# This is not a proper histogram, because the y-axis does not show
	# counts, but the sum of the actual feature contributions.
	# Don't know how to properly call this visualization ...
	# A histogram is misleading in this case because a large number of features
	# with a very small positive contribution would give the impression that
	# the predicted label should be positive as well.
	fig, ax = plt.subplots(figsize=(5, 2))
	x = max(abs(features['comb'].min()), abs(features['comb'].max()))
	rng = (-x, x)
	neg = features.loc[features['comb'] < 0, 'comb'].sort_values()
	negweight = neg.copy()
	# add the intercept term to the smallest negative weight
	negweight.iat[-1] += intercept
	pos = features.loc[features['comb'] > 0, 'comb']
	ax.hist(neg, bins=20, weights=negweight.abs(),
			range=rng, histtype='stepfilled', label='non-canonical')
	ax.hist(pos, bins=20, weights=pos,
			range=rng, histtype='stepfilled', label='canonical')
	ax.set_xlabel('Feature contribution (binned)')
	ax.set_ylabel('Contribution ')
	ax.set_xlim((-x * 1.05, x * 1.05))
	ax.legend(loc='best')
	for item in [ax.title, ax.xaxis.label, ax.yaxis.label
			] + ax.get_xticklabels() + ax.get_yticklabels():
		item.set_fontsize(8)
	result = b64fig(ax)
	plt.close(fig)
	return result


def scatterplot(selected_ti_id, feat, sim):
	"""Create a Bokeh scatter plot with categorical/qualitative colors.

	t-SNE if selected_ti_id is part of the corpus (better than PCA);
	PCA if user provided text (since t-SNE in sklearn does not support adding
	new datapoints)."""
	# TODOs:
	# - avoid sending all metadata for each PCA plot;
	#   only the highlight column is different for each request,
	#   and the row with user entered text.
	# - 3D PCA plot; apparently not supported by Bokeh.
	cols = ['PC 1', 'PC 2', 'Author', 'Title', 'YearFirstPublished',
			'DBNLgenre', 'DBNLsubgenre', 'Filename']
	if selected_ti_id is None:
		red = PCARED.loc[:, cols].copy()
	else:
		red = TSNERED.loc[:, cols].copy()
	red['highlight'] = [ti_id if ti_id == selected_ti_id
			else 'most similar' if ti_id in sim.index else 'rest'
			for ti_id in red.index]
	# some gymnastics to get the legend in the preferred order ...
	red = red.sort_values(
			by='highlight',
			key=lambda x: (x != selected_ti_id) + 2 * (x == 'rest'))
	if selected_ti_id is None:
		res = dict.fromkeys(red.columns, '')
		res['PC 1'], res['PC 2'] = REDUCER.transform(
				MODEL[0].transform(feat))[0, :]
		res['highlight'] = res['Title'] = 'your text'
		red = pd.DataFrame([res], index=['your text']).append(red)
	plot = bp.figure(plot_width=900, plot_height=700,
			title='%s of bigram frequencies in DBNL corpus '
				'(N=%d novels, log-transformed frequencies)'  % (
					'PCA' if selected_ti_id is None else 't-SNE',
					len(red)),
			tools='pan,wheel_zoom,box_zoom,reset,hover,save',
			x_axis_type=None, y_axis_type=None, min_border=1)
	plot.scatter(x='PC 1', y='PC 2', size=6,
			color=factor_cmap('highlight',
				palette=sns.color_palette('deep', as_cmap=True),
				factors=red.highlight.unique()[::-1]),
			marker=factor_mark('highlight', ['square', 'triangle', 'circle'],
				red.highlight.unique()),
			legend_field='highlight', fill_alpha=0.2, # line_color='white',
			source=bp.ColumnDataSource(red))
	hover = plot.select(dict(type=HoverTool))
	hover.tooltips={'label': '@Author @YearFirstPublished @Title | @DBNLgenre '
			'@DBNLsubgenre @Filename'}
	script, div = components(plot)
	return script, div


def b64fig(ax):
	"""Return plot as base64 encoded SVG string for use in data URI."""
	# ax.figure.tight_layout()
	figbytes = io.BytesIO()
	ax.figure.savefig(figbytes, format='svg')
	return b64encode(figbytes.getvalue()).decode('ascii')


logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
log.info('loading.')
if STANDALONE:
	from getopt import gnu_getopt, GetoptError
	try:
		opts, _args = gnu_getopt(sys.argv[1:], '',
				['port=', 'ip=', 'debug'])
		opts = dict(opts)
	except GetoptError as err:
		print('error: %r' % err, file=sys.stderr)
		sys.exit(2)
	DEBUG = '--debug' in opts
# pre-load data/models
MD = pd.read_csv('data/metadata.tsv', sep='\t', index_col='DBNLti_id')
UNIGRAMS = scipy.sparse.load_npz('data/unigram_counts.npz')
BIGRAMS = scipy.sparse.load_npz('data/bigram_counts.npz')
UNIGRAMSSUM = UNIGRAMS.sum(axis=1).A1
BIGRAMSSUM = BIGRAMS.sum(axis=1).A1
# Up to 100k ngrams, used for training classifier
ROWNAMES, _UNIGRAMCOLUMNS, BIGRAMCOLUMNS = joblib.load('data/ngrams_idx_col.pkl')
ROWNAMES = pd.Index(ROWNAMES)
BIGRAMCOLUMNS = pd.Index(BIGRAMCOLUMNS)
# Create trie for full index of ngrams
# ROWNAMES, UNIGRAMCOLUMNS, BIGRAMCOLUMNS = joblib.load('../output/ngrams_idx_col.pkl')
# UNIGRAMTRIE = RecordTrie('<i', ((a, (n, )) for n, a in enumerate(UNIGRAMCOLUMNS)))
# BIGRAMTRIE = RecordTrie('<i', ((a, (n, )) for n, a in enumerate(BIGRAMCOLUMNS)))
# UNIGRAMTRIE.save('data/unigrams.marisa')
# BIGRAMTRIE.save('data/bigrams.marisa')
UNIGRAMTRIE, BIGRAMTRIE = RecordTrie('<i'), RecordTrie('<i')
UNIGRAMTRIE.load('data/unigrams.marisa')
BIGRAMTRIE.load('data/bigrams.marisa')
# number of most frequent bigrams used by classifier
BIGRAMLIMIT = len(BIGRAMCOLUMNS)
log.info('read ngrams')

MODEL = joblib.load('data/classifier.pkl')
log.info('read classifier.pkl.')
VEC = feature_extraction.text.CountVectorizer(
		input='content', ngram_range=(2, 2), token_pattern=r'\b\S+\b',
		lowercase=True, vocabulary=BIGRAMCOLUMNS)
BIGRAMSNORMALIZED = MODEL[0].transform(BIGRAMS[:, :BIGRAMLIMIT])
log.info('transformed bigrams')
# from sklearn import decomposition, manifold
# REDUCER = decomposition.TruncatedSVD(n_components=2, random_state=1)
# REDUCER.fit(BIGRAMSNORMALIZED)
# joblib.dump(REDUCER, 'data/pca.pkl')
# REDUCER1 = decomposition.TruncatedSVD(n_components=50, random_state=1)
# REDUCER2 = manifold.TSNE(n_components=2, perplexity=30,
# 		learning_rate=len(ROWNAMES) / 48, metric='cosine', random_state=0,
# 		square_distances=True)
# TSNERED = pd.DataFrame(
# 		REDUCER2.fit_transform(REDUCER1.fit_transform(BIGRAMSNORMALIZED)),
# 		index=MD.index, columns=['PC 1', 'PC 2'])
# TSNERED.to_parquet('data/tsne.pqt')
TSNERED = pd.read_parquet('data/tsne.pqt')
TSNERED = pd.concat([TSNERED, MD], axis=1)
REDUCER = joblib.load('data/pca.pkl')
PCARED = pd.DataFrame(REDUCER.transform(BIGRAMSNORMALIZED),
		index=MD.index, columns=['PC 1', 'PC 2'])
PCARED = pd.concat([PCARED, MD], axis=1)
log.info('read pca.pkl & tsne.pqt.')
log.info('done.')
if STANDALONE:
	# from werkzeug.middleware.profiler import ProfilerMiddleware
	# APP.wsgi_app = ProfilerMiddleware(
	# 		APP.wsgi_app, restrictions=(10, ), sort_by=('time', ))
	APP.run(
			host=opts.get('--ip', '0.0.0.0'),
			port=int(opts.get('--port', 5004)),
			use_reloader=False, threaded=True,
			debug=DEBUG)
