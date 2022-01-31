"""Train predictive models."""
import warnings
import numpy as np
import pandas as pd
import joblib
import scipy.sparse
from scipy.stats import chi2_contingency
from sklearn import (pipeline, preprocessing, feature_extraction, svm,
		linear_model, model_selection, metrics)

NORM = 'l2'
USE_IDF = False
SMOOTH_IDF = False
SUBLINEAR_TF = True
C = 1

warnings.filterwarnings("error")


class OrderedGroupKFold:
	"""K-fold iterator that deterministically assigns items to folds, while
	ensuring that each given label appears only in a single fold.

	Each k-th label ends up in the k-th fold, where k is the index of where the
	label is first encountered in ``labels``. The number of distinct labels has
	to be at least equal to the number of folds.

	This code is based on GroupKFold from scikit-learn; it can be seen as a
	combination of StratifiedKFold and GroupKFold.

	Parameters
	----------
	groups : array-like with shape (n_samples, )
		Contains a group for each sample.
		The folds are built so that the same label does not appear in two
		different folds.

	n_splits : int, default=3
		Number of folds. Must be at least 2.

	Examples
	--------
	>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
	>>> y = np.array([1, 2, 3, 4])
	>>> groups = np.array([0, 0, 2, 2])
	>>> group_kfold = OrderedGroupKFold(n_splits=2)
	>>> group_kfold.get_n_splits()
	2
	>>> print(group_kfold)
	OrderedGroupKFold(n_splits=2)
	>>> for train_index, test_index in group_kfold.split(X, y, groups):
	...     print("TRAIN:", train_index, "TEST:", test_index)
	...     X_train, X_test = X[train_index], X[test_index]
	...     y_train, y_test = y[train_index], y[test_index]
	...     print(X_train, X_test, y_train, y_test)
	...
	TRAIN: [2 3] TEST: [0 1]
	[[5 6]
	 [7 8]] [[1 2]
	 [3 4]] [3 4] [1 2]
	TRAIN: [0 1] TEST: [2 3]
	[[1 2]
	 [3 4]] [[5 6]
	 [7 8]] [1 2] [3 4]
	"""
	def __init__(self, n_splits=3):
		self.n_splits = n_splits

	def _iter_test_indices(self, X, y, groups):
		if groups is None:
			raise ValueError("The 'groups' parameter should not be None.")

		unique_groups, unique_indices, unique_inverse = np.unique(
				groups, return_index=True, return_inverse=True)
		n_groups = len(unique_groups)

		if self.n_splits > n_groups:
			raise ValueError("Cannot have number of splits n_splits=%d greater"
					" than the number of groups: %d."
					% (self.n_splits, n_groups))

		# indices of groups in order of first occurrence
		ranking = np.argsort(unique_indices)

		# Weight labels by their number of occurences
		n_samples_per_label = np.bincount(unique_inverse)

		# Total weight of each fold
		n_samples_per_fold = np.zeros(self.n_splits, dtype=np.intp)

		# Mapping from label index to fold index
		group_to_fold = np.zeros(n_groups, dtype=np.intp)

		for n in range(n_groups):
			# Assign this label to the fold that currently has the least
			# number of samples
			fold = np.argmin(n_samples_per_fold)
			n_samples_per_fold[fold] += n_samples_per_label[ranking[n]]
			group_to_fold[ranking[n]] = fold

		self.idxs = group_to_fold[unique_inverse]
		for f in range(self.n_splits):
			yield (self.idxs == f).nonzero()[0]

	def _iter_test_masks(self, X=None, y=None, groups=None):
		"""Generates boolean masks corresponding to test sets.
		By default, delegates to _iter_test_indices(X, y, groups)
		"""
		n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
		for test_index in self._iter_test_indices(X, y, groups):
			test_mask = np.zeros(n_samples, dtype=bool)
			test_mask[test_index] = True
			yield test_mask

	def __repr__(self):
		return '%s(n_splits=%d)' % (self.__class__.__name__, self.n_splits)

	def get_n_splits(self, X=None, y=None, groups=None):
		"""Returns the number of splitting iterations in the cross-validator"""
		return self.n_splits

	def split(self, X, y=None, groups=None):
		"""Generate indices to split data into training and test set.
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			Training data, where n_samples is the number of samples
			and n_features is the number of features.
		y : array-like of shape (n_samples,), default=None
			The target variable for supervised learning problems.
		groups : array-like of shape (n_samples,), default=None
			Group labels for the samples used while splitting the dataset into
			train/test set.
		Yields
		------
		train : ndarray
			The training set indices for that split.
		test : ndarray
			The testing set indices for that split.
		"""
		n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
		if self.n_splits > n_samples:
			raise ValueError("Cannot have number of splits n_splits=%d greater"
					" than the number of samples: n_samples=%d."
					% (self.n_splits, n_samples))

		indices = np.arange(n_samples)
		for test_index in self._iter_test_masks(X, y, groups):
			train_index = indices[np.logical_not(test_index)]
			test_index = indices[test_index]
			yield train_index, test_index


def evalreport(y_true, y_pred, target):
	"""Given two Series objects aligned with 'target', compute metrics for each
	fold and report mean/stderr."""
	result = pd.DataFrame(index=['mean', 'std err'])
	ids = [target[target.fold == n].index for n in target.fold.unique()]
	acc = pd.Series([
		metrics.accuracy_score(y_true[a].astype(bool), y_pred[a].astype(bool))
		for a in ids])
	result['Accuracy'] = 100 * pd.Series([
			acc.mean(), acc.sem()], index=result.index)
	return result.T


def train_cosinedelta(md):
	df = pd.read_csv('output/mfw.csv', index_col=0)
	df.index = [a.rsplit('_', 1)[0] for a in df.index]
	df = df.loc[md.index, :]

	model = pipeline.Pipeline([
		('pre', pipeline.Pipeline([
			('norm', preprocessing.Normalizer('l1')),
			('scaler', preprocessing.StandardScaler())])),
		('logreg', linear_model.LogisticRegressionCV(
			random_state=0, max_iter=5000, class_weight='balanced'))])
	X = df.values
	y_true = md.DBNLSecRefsTitle != 0
	cv = model_selection.PredefinedSplit(test_fold=md['fold'])
	y_pred = model_selection.cross_val_predict(model, X, y_true, cv=cv)
	result = pd.Series(y_pred, index=md.index)
	print('classifier trained on 1000 MFW cosine delta vectors')
	print(evalreport(y_true, result, md).round(decimals=3), '\n')
	print(metrics.classification_report(
			y_true.astype(bool),
			result.astype(bool),
			digits=3, target_names=['Non-canonical', 'Canonical']))
	model.fit(X, y_true)
	joblib.dump(model, 'classifier_cosinedelta.pkl')
	feat = pd.DataFrame({
			'weight': model[1].coef_[0]},
			index=df.columns).sort_values('weight')
	print('top non-canonical features:')
	print(feat.head(15))
	print('\ntop canonical features:')
	print(feat.tail(15).iloc[::-1])


def train_bigrams(md):
	bigrams = scipy.sparse.load_npz('output/bigram_counts.npz')
	_rownames, _unigramcolumns, bigramcolumns = joblib.load(
			'output/ngrams_idx_col.pkl')
	bigrams = bigrams[:, :100_000]
	bigramcolumns = bigramcolumns[:100_000]

	model = pipeline.Pipeline([
		('tfidf', feature_extraction.text.TfidfTransformer(
			norm=NORM, use_idf=USE_IDF, smooth_idf=SMOOTH_IDF,
			sublinear_tf=SUBLINEAR_TF)),
		# ('svm', svm.LinearSVC(C=C, random_state=1, max_iter=5000,
		# 	class_weight='balanced'))])
		# ('logreg', linear_model.LogisticRegressionCV(
		# 	Cs=3, random_state=0, max_iter=5000, class_weight='balanced',
		# 	solver='sag'))
		('logreg', linear_model.LogisticRegression(
			random_state=0, max_iter=5000, class_weight='balanced',
			solver='sag'))
		])
	X = bigrams
	y_true = md.DBNLSecRefsTitle != 0
	cv = model_selection.PredefinedSplit(test_fold=md['fold'])
	# y_pred = model_selection.cross_val_predict(model, X, y_true, cv=cv)
	y_score = model_selection.cross_val_predict(model, X, y_true, cv=cv,
			method='predict_proba')[:, 1]
	y_pred = y_score > 0.5
	result = pd.Series(y_pred, index=md.index)
	print('classifier trained on 100,000 bigrams')
	print(evalreport(y_true, result, md).round(decimals=3), '\n')
	print(metrics.classification_report(
			y_true.astype(bool),
			result.astype(bool),
			digits=3, target_names=['Non-canonical', 'Canonical']))
	model.fit(X, y_true)
	joblib.dump(model, 'demo/classifier.pkl')
	avgfreq = model[0].transform(bigrams).mean(axis=0).A1
	feat = pd.DataFrame({
			'weight': model[1].coef_[0],
			'avgfreq': avgfreq},
			index=bigramcolumns)
	feat['comb'] = feat['avgfreq'] * feat['weight']
	feat = feat.sort_values('comb')
	print('top non-canonical features:')
	print(feat.head(15))
	print('\ntop canonical features:')
	print(feat.tail(15).iloc[::-1])

	md['score'] = y_score
	md = md.rename(columns={'YearFirstPublished': 'Year'}
			).sort_values(by='score')
	cols = ['score', 'Year', 'Author', 'Title', 'DBNLsubgenre', 'DBNLSecRefsTitle']
	print('\nmisclassifications: predicted as non-canonical, but is canonical')
	print(md.loc[y_true & ~y_pred, cols].head(10))
	print(md.loc[y_true & ~y_pred, cols].tail(5).iloc[::-1, :])
	print('misclassifications: predicted as canonical, but is non-canonical')
	print(md.loc[~y_true & y_pred, cols].head(5))
	print(md.loc[~y_true & y_pred, cols].tail(10).iloc[::-1, :])
	print('correct canonical classifications):')
	print(md.loc[y_true & y_pred, cols].head(5))
	print(md.loc[y_true & y_pred, cols].tail(10).iloc[::-1, :])

	# Significance test to see if gender influences prediction performance
	obs = np.array([
			[((y_true == y_pred) & md.Woman).sum(),
				((y_true != y_pred) & md.Woman).sum()],
			[((y_true == y_pred) & ~md.Woman).sum(),
				((y_true != y_pred) & ~md.Woman).sum()],
			])
	print(obs)
	print(chi2_contingency(obs))
	print('only genre==jeugdliteratuur')
	subset = md.DBNLgenre.str.contains('jeugdliteratuur')
	print(metrics.classification_report(
			y_true[subset].astype(bool),
			result[subset].astype(bool),
			digits=3, target_names=['Non-canonical', 'Canonical']))


def train_bigrams_reg(md):
	df = pd.read_parquet('output/bigrams.pqt')
	df.index = [a.rsplit('_', 1)[0] for a in df.index]
	df = df.loc[md.index, :]

	model = pipeline.Pipeline([
		('tfidf', feature_extraction.text.TfidfTransformer(
			norm=NORM, use_idf=USE_IDF, smooth_idf=SMOOTH_IDF,
			sublinear_tf=SUBLINEAR_TF)),
		('svm', svm.LinearSVR(C=C, random_state=1, max_iter=5000))])
	X = df.values
	y_true = md.DBNLSecRefsTitle
	cv = model_selection.PredefinedSplit(test_fold=md['fold'])
	y_pred = model_selection.cross_val_predict(model, X, y_true, cv=cv)
	result = pd.Series(y_pred, index=md.index)
	print('regression trained on 10,000 bigrams')
	print(metrics.r2_score(y_true, result))
	# model.fit(X, y_true)
	# joblib.dump(model, 'regression_bigrams.pkl')
	avgfreq = model[0].transform(df.values).toarray().mean(axis=0)
	feat = pd.DataFrame({
			'weight': model[1].coef_[0],
			'avgfreq': avgfreq},
			index=df.columns)
	feat['comb'] = feat['avgfreq'] * feat['weight']
	feat = feat.sort_values('comb')
	print('top non-canonical features:')
	print(feat.head(15))
	print('\ntop canonical features:')
	print(feat.tail(15).iloc[::-1])


def train_factorizationmachines(md):
	import polylearn  # https://github.com/scikit-learn-contrib/polylearn
	# Result: very slow and no improvement in scores.

	# df = pd.read_parquet('output/bigrams.pqt')
	# df.index = [a.rsplit('_', 1)[0] for a in df.index]
	# df = df.loc[md.index, :]

	# df1 = pd.read_csv('output/mfw.csv', index_col=0)
	# df1.index = [a.rsplit('_', 1)[0] for a in df1.index]
	# df1 = df1.loc[md.index, :]
	# df = pd.concat([df, df1], axis=1)

	df = pd.read_csv('output/mfw.csv', index_col=0).iloc[:, :100]
	df.index = [a.rsplit('_', 1)[0] for a in df.index]
	df = df.loc[md.index, :]
	df['YearFirstPublished'] = md['YearFirstPublished']
	# df['YearAuthorBorn'] = md['Born']

	model = pipeline.Pipeline([
		('tfidf', feature_extraction.text.TfidfTransformer(
			norm=NORM, use_idf=USE_IDF, smooth_idf=SMOOTH_IDF,
			sublinear_tf=SUBLINEAR_TF)),
		('fm', polylearn.FactorizationMachineClassifier(
			n_components=10, random_state=0, fit_linear=False,
			warm_start=True, verbose=True))])
	X = df.values
	y_true = md.DBNLSecRefsTitle != 0
	cv = model_selection.PredefinedSplit(test_fold=md['fold'])
	y_pred = model_selection.cross_val_predict(model, X, y_true, cv=cv)
	y_score = np.zeros(len(y_true))
	# y_score = model_selection.cross_val_predict(model, X, y_true, cv=cv,
	# 		method='decision_function')
	# y_pred = y_score > 0
	result = pd.Series(y_pred, index=md.index)
	print('factorization machines classifier '
			'trained on 10,000 bigrams and 1000 unigrams')
	print(evalreport(y_true, result, md).round(decimals=3), '\n')
	print(metrics.classification_report(
			y_true.astype(bool),
			result.astype(bool),
			digits=3, target_names=['Non-canonical', 'Canonical']))
	model.fit(X, y_true)
	joblib.dump(model, 'classifier_fm_unibigrams.pkl')

	md['score'] = y_score
	md = md.rename(columns={'YearFirstPublished': 'Year'}
			).sort_values(by='score')
	cols = ['score', 'Year', 'Author', 'Title', 'DBNLsubgenre', 'DBNLSecRefsTitle']
	print('\nmisclassifications: predicted as non-canonical, but is canonical')
	print(md.loc[y_true & ~y_pred, cols])
	print('misclassifications: predicted as canonical, but is non-canonical')
	print(md.loc[~y_true & y_pred, cols])
	print('correct canonical classifications):')
	print(md.loc[y_true & y_pred, cols])


def scattertext_plot(md):
	import scattertext as st
	df = pd.read_parquet('output/bigrams.pqt')
	df.index = [a.rsplit('_', 1)[0] for a in df.index]
	df = df.loc[md.index, :]
	df = df.iloc[:, :4000]

	corpus = st.CorpusFromScikit(
		X=scipy.sparse.csr_matrix(df.values),
		y=(md.DBNLSecRefsTitle != 0).values,
		feature_vocabulary={a: n for n, a in enumerate(df.columns)},
		category_names=['non-canonical', 'canonical'],
		raw_texts=[],
	).build()

	clf = joblib.load('demo/classifier.pkl')[1]
	term_scores = clf.coef_[0][:4000]

	html = st.produce_frequency_explorer(
		corpus,
		'canonical',
		scores=term_scores,
		use_term_significance=False,
		terms_to_include=df.columns,
		# st.AutoTermSelector.get_selected_terms(corpus, term_scores, 4000),
		metadata=df.index)

	with open('output/stplotbi.html', 'w') as out:
		out.write(html)


# def exp():
# 	# train on 19th cent., eval on 20th,
# 	# and vice versa


def main():
	md = pd.read_csv('output/metadata.tsv', sep='\t', index_col='DBNLti_id')
	cv = OrderedGroupKFold(n_splits=5)
	newsort = md.sort_values(by=['DBNLSecRefsTitle', 'YearFirstPublished'])
	_ = list(cv._iter_test_indices('x', 'y', newsort.DBNLpers_id))
	md['fold'] = pd.Series(cv.idxs, index=newsort.index)

	# train_cosinedelta(md)
	train_bigrams(md)
	# train_bigrams_reg(md)
	# train_factorizationmachines(md)
	# scattertext_plot(md)


if __name__ == '__main__':
	main()
