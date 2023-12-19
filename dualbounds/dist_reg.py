import copy
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import stats
from sklearn.linear_model import RidgeCV, LogisticRegression, LogisticRegressionCV
from . import utilities
from .utilities import parse_dist, BatchedCategorical

def create_folds(n, nfolds):
	splits = np.linspace(0, n, nfolds+1).astype(int)
	starts = splits[0:-1]
	ends = splits[1:]
	return starts, ends

def _cross_fit_predictions(
	W,
	X,
	Y,
	S=None,
	nfolds=2,
	train_on_selections=False,
	model=None,
	model_cls=None, 
	**model_kwargs
):
	"""
	Parameters
	----------
	model : DistReg
		instantiation of DistReg class. This will be copied.
	model_cls : 
		Alterantively, give the class name and have it constructed.
	model_kwargs : dict
		kwargs to construct model; used only if model_cls is specified.
	S : array
		n-length array of selection indicators. Optional, may not be provided.
	train_on_selections : bool
		If True, trains model only on data where S[i] == 1.

	Returns
	-------
	pred0s : list or array
		List of test predictions from model_cls on each fold assuming W = 0.
		Also assumes S = 1 if S is provided.
	pred1s : list or array
		List of test predictions from model_cls on each fold assuming W = 1.
		Also assumes S = 1 if S is provided.
	"""
	# concatenate S to features
	n = len(X)
	if S is None:
		S = np.ones(n)
	X = np.concatenate([S.reshape(n, 1), X], axis=1)

	# create folds
	starts, ends = create_folds(n=n, nfolds=nfolds)
	# loop through folds
	pred0s = []; pred1s = [] # results for W = 0, W = 1
	for start, end in zip(starts, ends):
		# Pick out data from the other folds
		not_in_fold = [i for i in np.arange(n) if i < start or i >= end]
		if train_on_selections:
			opt2 = [i for i in not_in_fold if S[i] == 1]
			if len(opt2) == 0:
				warnings.warn(f"S=0 for all folds except {start}-{end}.")
			else:
				not_in_fold = opt2

		# Fit model
		if model is None:
			reg_model = model_cls(**model_kwargs)
		else:
			reg_model = copy.copy(model)
		
		reg_model.fit(
			W=W[not_in_fold], X=X[not_in_fold], Y=Y[not_in_fold]
		)

		# predict and append on this fold
		subX = X[start:end].copy(); subX[:, 0] = 1 # set selection = 1 for predictions
		pred0, pred1 = reg_model.predict(subX)
		pred0s.append(pred0); pred1s.append(pred1)
	# concatenate if arrays; else return
	if isinstance(pred0s[0], np.ndarray):
		pred0s = np.concatenate(pred0s, axis=0)
		pred1s = np.concatenate(pred1s, axis=0)
	return pred0s, pred1s

class RidgeDistReg:
	def __init__(
		self,
		eps_dist='gaussian',
		**model_kwargs,
		#heterosked=False,
	):
		"""
		Parameters
		----------
		eps_dist : str
			Str specifying the (parametric) distribution of the residuals.
		model_kwargs : dict
			kwargs for sklearn.linear_models.RidgeCV() constructor.
		"""
		self.eps_dist = eps_dist
		self.model_kwargs = model_kwargs
		#self.heterosked = heterosked

	def feature_transform(self, W, X):
		"""
		In the future, can add splines/basis functions.
		"""
		return np.concatenate([W.reshape(-1, 1), X],axis=1)

	def fit(self, W, X, Y):
		"""
		"""
		# fit ridge
		features = self.feature_transform(W, X)
		self.model = RidgeCV(**self.model_kwargs)
		self.model.fit(features, Y)

		# fit variance
		self.hatsigma = np.sqrt(
			np.power(self.model.predict(features) - Y, 2).mean()
		)
	
	def predict(self, X, W=None):
		"""
		If W is None, returns (y0_dists, y1_dists)
		Else, returns (y_dists) 
		"""
		if W is not None:
			features = self.feature_transform(W, X=X)
			mu = self.model.predict(features)
			# return scipy dists
			return parse_dist(
				self.eps_dist, loc=mu, scale=self.hatsigma, 
			)
		else:
			n = len(X)
			W0 = np.zeros(n); W1 = np.ones(n)
			return self.predict(X, W=W0), self.predict(X, W=W1)

class MonotoneLogisticReg:
	def __init__(self):
		pass

	def fit(self, X, Y, lmda=1):
		n, p = X.shape
		sig1 = X[:, 0].std()
		zeros = np.zeros(n)
		beta = cp.Variable(p)
		X1beta = X @ beta
		term1 = cp.multiply(Y, X1beta)
		term2 = cp.log_sum_exp(cp.vstack([zeros, X1beta]), axis=0)
		term3 = lmda * cp.sum(cp.power(beta, 2))
		obj = cp.Maximize(cp.sum(term1 - term2) - term3)
		problem = cp.Problem(objective=obj, constraints=[beta[0] >= 0.1 / sig1])
		try:
			problem.solve(solver='ECOS', max_iters=100)
		except cp.error.SolverError:
			problem.solve(solver='ECOS', max_iters=500)
		self.beta = beta.value

	def predict_proba(self, X):
		mu = X @ self.beta
		p1s = np.exp(mu) / (1 + np.exp(mu))
		return np.stack([1 - p1s, p1s], axis=1)

class LogisticCV:
	def __init__(self, monotonicity, **model_kwargs):
		self.monotonicity = monotonicity
		self.model_kwargs = model_kwargs

	def feature_transform(self, W, X):
		""" Concatenates W, X and adds intercept currently """
		return np.concatenate([W.reshape(-1, 1), X, np.ones((len(X), 1))], axis=1)

	def fit(self, W, X, Y):
		# fit sklearn model
		self.model = MonotoneLogisticReg(**self.model_kwargs)
		#self.model = LogisticRegression(**self.model_kwargs)
		self.model.fit(
			self.feature_transform(W, X),
			Y
		)

	def predict(self, X, W=None):
		"""
		If W is None, returns (P(Y = 1 | W = 0, X), P(Y = 1 | W = 1, X))
		Else, returns P(Y = 1 | W , X) 
		"""
		if W is not None:
			# return predict P(Y = 1 | X, W)
			return self.model.predict_proba(
				self.feature_transform(W, X)
			)[:, 1]
		else:
			# make predictions for W = 0 and W = 1
			n = len(X)
			W0 = np.zeros(n); W1 = np.ones(n)
			p0s, p1s = self.predict(X, W=W0), self.predict(X, W=W1)
			# enforce monotonicity
			if self.monotonicity:
				flags = p0s > p1s
				avg = (p0s[flags] + p1s[flags]) / 2
				p0s[flags] = avg
				p1s[flags] = np.minimum(1, avg + 1e-5)
			return p0s, p1s