import copy
import numpy as np
import cvxpy as cp
from scipy import stats
import sklearn.linear_model as lm
from . import utilities
from .utilities import parse_dist, BatchedCategorical

TOL = 1e-6

def parse_model_type(model_type, discrete):
	# return non-strings
	if not isinstance(model_type, str):
		return model_type
	# parse
	model_type = model_type.lower()
	if not discrete:
		if model_type == 'ridge':
			return lm.RidgeCV
		elif model_type == 'lasso':
			return lm.LassoCV
		elif model_type in ['elastic', 'elasticnet']:
			return lm.ElasticNetCV
		elif model_type in ['rf', 'randomforest']:
			import sklearn.ensemble
			return sklearn.ensemble.RandomForestRegressor
		elif model_type == 'knn':
			import sklearn.neighbors
			return sklearn.neighbors.KNeighborsRegressor
		else:
			raise ValueError(f"Unrecognized model_type={model_type}")
	else:
		if model_type in ['logistic', 'ridge', 'lasso', 'elastic', 'elasticnet']:
			return lm.LogisticRegressionCV
		elif model_type in ['monotone_logistic', 'logistic_monotone']:
			return MonotoneLogisticReg
		elif model_type in ['rf', 'randomforest']:
			import sklearn.ensemble
			return sklearn.ensemble.RandomForestClassifier
		elif model_type == 'knn':
			import sklearn.neighbors
			return sklearn.neighbors.KNeighborsClassifier
		else:
			raise ValueError(f"Unrecognized model_type={model_type}")

def create_folds(n, nfolds):
	splits = np.linspace(0, n, nfolds+1).astype(int)
	starts = splits[0:-1]
	ends = splits[1:]
	return starts, ends

def cross_fit_predictions(
	W,
	X,
	y,
	S=None,
	nfolds=2,
	train_on_selections=False,
	model=None,
	model_cls=None, 
	probs_only=False,
	**model_kwargs
):
	"""
	Performs cross-fitting on a model class inheriting from ``dist_reg.DistReg.``

	Parameters
	----------
	model : DistReg
		instantiation of ``dist_reg.DistReg`` class. This will
		be copied. E.g., 
		``model=dist_reg.CtsDistReg(model_type='ridge', eps_dist="laplace").``
	model_cls : 
		Alterantively, give the class name and have it constructed.
		E.g, ``model_cls=dist_reg.CtsDistReg``.
	model_kwargs : dict
		kwargs to construct model; used only if model_cls is specified.
		E.g., ``model_kwargs=dict(eps_dist=laplace)``.
	S : array
		Optional n-length array of selection indicators.
	train_on_selections : bool
		If True, trains model only on data where S[i] == 1.
	probs_only : bool
		For binary data, returns P(Y = 1 | X, W) instead of a distribution.
		Defaults to False.

	Returns
	-------
	y0_dists : list
		list of batched scipy distributions whose shapes sum to n.
		the ith distribution is the out-of-sample estimate of
		the conditional law of Yi(0) | X[i]
	y1_dists : list
		list of batched scipy distributions whose shapes sum to n.
		the ith distribution is the out-of-sample estimate of
		the conditional law of Yi(1) | X[i]
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
	fit_models = []
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
			reg_model = copy.deepcopy(model)
		
		reg_model.fit(
			W=W[not_in_fold], X=X[not_in_fold], y=y[not_in_fold]
		)

		# predict and append on this fold
		subX = X[start:end].copy(); subX[:, 0] = 1 # set selection = 1 for predictions
		if not probs_only:
			pred0, pred1 = reg_model.predict(subX)
		else:
			pred0, pred1 = reg_model.predict_proba(subX)
		pred0s.append(pred0); pred1s.append(pred1)
		fit_models.append(reg_model)
	# concatenate if arrays; else return
	if isinstance(pred0s[0], np.ndarray):
		pred0s = np.concatenate(pred0s, axis=0)
		pred1s = np.concatenate(pred1s, axis=0)
	return pred0s, pred1s, fit_models

class DistReg:
	"""
	A generic class for distributional regressions, meant for subclassing.

	Parameters
	----------
	how_transform : str
		Str specifying how to transform the features before fitting
		the underlying model. One of several options:

		- 'identity': does not transform the features
		- 'interactions' : interaction terms btwn. the treatment/covariates

		The default is ``interactions``.
	"""

	def __init__(self, how_transform):
		self.how_transform = how_transform


	def fit(self, W, X, y):
		"""
		Fits model on the data.

		Parameters
		----------
		W : np.array
			n-length array of binary treatment indicators.
		X : np.array
			(n, p)-shaped array of covariates.
		y : np.array
			n-length array of outcome measurements.
		"""
		raise NotImplementedError()

	def feature_transform(self, W, X):
		"""
		Transforms the features before feeding them to the base model.

		Parameters
		----------
		X : np.array
			(n, p)-shaped array of covariates.
		W : np.array
			n-length array of treatment indicators

		Returns
		-------
		features : np.array
			(n, d)-shaped array of features.
		"""
		if self.how_transform in ['none', 'identity']:
			return np.concatenate([W.reshape(-1, 1), X],axis=1)
		elif self.how_transform in [
			'int', 'interaction', 'interactions'
		]:
			return np.concatenate([
				W.reshape(-1, 1), W.reshape(-1, 1) * X, X], axis=1
			)
		else:
			raise ValueError(f"Unrecognized transformation {self.how_transform}")

	def predict(self, X, W=None):
		"""
		Parameters
		----------
		X : np.array
			(n, p)-shaped array of covariates.
		W : np.array
			Optional n-length array of binary treatment indicators.

		Returns
		-------
		y_dists : stats.rv_continuous / stats.rv_discrete
			batched scipy distribution of shape (n,) where the ith
			distribution is the conditional law of Y[i] | X[i], W[i].
			Only returned if W is provided.
		(y0_dists, y1_dists) : tuple 
			If W is not provided, returns a tuple of batched scipy 
			dists of shape (n,). The ith distribution in yk_dists is
			the conditional law of Yi(k) | Xi, for k in {0,1}.
		"""
		raise NotImplementedError()

class CtsDistReg(DistReg):
	"""
	Distributional regression for continuous outcomes.

	Parameters
	----------
	model_type : str or sklearn class
		Str specifying a sklearn model class to use; options include
		'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'. One can
		also directly pass an sklearn class constructor, e.g., 
		``model_type=sklearn.ensemble.KNeighborsRegressor``. 
	how_transform : str
		Str specifying how to transform the features before fitting
		the base sklearn model.
	eps_dist : str
		Str specifying the (parametric) distribution of the residuals.
		One of ['gaussian', 'laplace', 'expon', 'tdist']. Defaults 
		to ``gaussian``.
	heterosked : bool
		If True, estimates Var(Y | X) as a function of X by using
		the specified model to predict both E[Y^2 | X] and E[Y | X].
		If False, assumes Var(Y | X) is constant. Defaults to False.
	model_kwargs : dict
		kwargs for sklearn base model constructor. E.g., for ``knn``,
		model_kwargs could include ``n_neighbors``.
	"""
	def __init__(
		self,
		model_type='ridge',
		eps_dist='gaussian',
		how_transform='interactions',
		heterosked=False,
		**model_kwargs,
	):
		self.eps_dist = eps_dist
		self.model_type = parse_model_type(model_type, discrete=False)
		self.model_kwargs = model_kwargs
		self.how_transform = str(how_transform).lower()
		self.heterosked = heterosked

	def fit(self, W, X, y):
		# fit ridge
		features = self.feature_transform(W=W, X=X)
		self.model = self.model_type(**self.model_kwargs)
		self.model.fit(features, y)

		# fit variance
		if self.heterosked:
			self.m2_model = self.model_type(**self.model_kwargs)
			self.m2_model.fit(features, y**2)

		# estimate of E[Var(Y | X)] 
		self.hatsigma = np.sqrt(
			np.power(self.model.predict(features) - y, 2).mean()
		)
	
	def predict(self, X, W=None):
		if W is not None:
			features = self.feature_transform(W, X=X)
			mu = self.model.predict(features)
			# heteroskedasticity
			if self.heterosked:
				scale = self.m2_model.predict(features) - mu**2
				# ensure positivity
				scale = np.maximum(scale, 0.1 * self.hatsigma)
			else:
				scale = self.hatsigma
			# return scipy dists
			return parse_dist(
				self.eps_dist, mu=mu, sd=scale, 
			)
		else:
			n = len(X)
			W0 = np.zeros(n); W1 = np.ones(n)
			return self.predict(X, W=W0), self.predict(X, W=W1)

class BinaryDistReg(DistReg):
	"""
	A wrapper of sklearn.LogisticRegression which inherits from ``DistReg``

	Parameters
	----------
	how_transform : str
		Str specifying how to transform the features before fitting
		a ``LogisticCV`` model. See the base ``DistReg`` class for details.
	monotonicity : bool
		If true, ensures that the coefficient corresponding to the treatment
		is nonnegative. This is important when fitting Lee Bounds that assume
		monotonicity. Defaults to False.
	model_kwargs : dict
		kwargs to sklearn 
	"""
	def __init__(
		self, 
		model_type='logistic', 
		monotonicity=False,
		how_transform='interactions',
		**model_kwargs
	):
		self.model_type = parse_model_type(model_type, discrete=True)
		self.monotonicity = monotonicity
		self.model_kwargs = model_kwargs
		self.how_transform = str(how_transform).lower()

	def fit(self, W, X, y):
		# check y is binary
		if set(np.unique(y).tolist()) != set([0,1]):
			raise ValueError(f"y must be binary; instead np.unique(y)={np.unique(y)}")
		# fit 
		features = self.feature_transform(W=W, X=X)
		self.model = self.model_type(**self.model_kwargs)
		self.model.fit(features, y)

	def predict_proba(
		self, X, W=None
	):
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
			p0s, p1s = self.predict_proba(X, W=W0), self.predict_proba(X, W=W1)
			p0s = np.minimum(1-TOL, np.maximum(TOL, p0s))
			p1s = np.minimum(1-TOL, np.maximum(TOL, p1s))
			# enforce monotonicity
			if self.monotonicity:
				flags = p0s > p1s
				avg = (p0s[flags] + p1s[flags]) / 2
				p0s[flags] = np.maximum(TOL, avg-TOL)
				p1s[flags] = np.minimum(1-TOL, avg+TOL)
			return p0s, p1s

	def predict(self, X, W=None):
		"""
		If W is None, returns (y0_dists, y1_dists)
		Else, returns (y_dists) 
		"""
		if W is not None:
			# predict
			features = self.feature_transform(W, X=X)
			probs = self.model.predict_proba(features)
			# return BatchedCategorical object
			vals = np.zeros((len(X), 2))
			vals[:, -1] += 1
			return BatchedCategorical(
				vals=vals, probs=probs
			)
		else:
			n = len(X)
			W0 = np.zeros(n); W1 = np.ones(n)
			return self.predict(X, W=W0), self.predict(X, W=W1)

class MonotoneLogisticReg:
	"""
	A logistic regression solver which ensures that beta[0] >= 0.
	Useful for computing Lee bounds which assume monotonicity.
	"""
	def __init__(self):
		pass

	def fit(self, X, y, lmda=1):
		n, p = X.shape
		sig1 = X[:, 0].std()
		zeros = np.zeros(n)
		beta = cp.Variable(p)
		X1beta = X @ beta
		term1 = cp.multiply(y, X1beta)
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