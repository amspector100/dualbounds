import copy
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import stats
import sklearn.linear_model as lm
from . import utilities
from .utilities import parse_dist, BatchedCategorical, vrange
# typing
import sklearn.base
from typing import Optional, Union

TOL = 1e-6
DEFAULT_ALPHAS = np.log(np.logspace(0.0001, 100, base=np.e, num=10))
DEFAULT_ALPHAS = np.concatenate([DEFAULT_ALPHAS, np.array([1000, 10000])])


def parse_model_type(model_type, discrete):
	# return non-strings
	if not isinstance(model_type, str):
		return model_type
	# parse
	model_type = model_type.lower()
	if not discrete:
		if model_type == 'ridge':
			return lm.RidgeCV
		elif model_type == 'ols':
			return lm.LinearRegression
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

def _enforce_monotonicity(
	y0_dists: Union[BatchedCategorical, np.array], 
	y1_dists: Union[BatchedCategorical, np.array],
	margin=0.005,
):
	"""
	For a binary variable, enforces the monotonicity constraint
	that P(Y_i(0) = 1) <= P(Y_i(1) = 1).

	Parameters
	----------
	y0_dists : BatchedCategorical | np.array
		Either an array whose ith element is P(Y_i(0) = 1)
		or a BatchedCategorical distribution.
	y1_dists : BatchedCategorical | np.array
		Either an array whose ith element is P(Y_i(1) = 1)
		or a BatchedCategorical distribution.
	margin : float
		Minimum distance between P(Y_i(0) = 1) and P(Y_i(1) = 1).

	Returns
	-------
	y0_dists : BatchedCategorical | np.array
		Either an array whose ith element is P(Y_i(0) = 1)
		or a BatchedCategorical distribution.
	y1_dists : BatchedCategorical | np.array
		Either an array whose ith element is P(Y_i(1) = 1)
		or a BatchedCategorical distribution.

	Notes
	-----
	The output format is the same as the input format.
	"""
	# Convert to np array
	if isinstance(y0_dists, BatchedCategorical):
		p0s = y0_dists.probs[:, 1].copy()
	else:
		p0s = y0_dists.copy()

	if isinstance(y1_dists, BatchedCategorical):
		p1s = y1_dists.probs[:, 1].copy()
	else:
		p1s = y1_dists.copy()

	# Clip
	p0s = np.minimum(1-TOL, np.maximum(TOL, p0s))
	p1s = np.minimum(1-TOL, np.maximum(TOL, p1s))
	# enforce monotonicity
	flags = p0s > p1s - margin
	avg = (p0s[flags] + p1s[flags]) / 2
	p0s[flags] = np.maximum(TOL, avg-margin)
	p1s[flags] = np.minimum(1-TOL, avg+margin)
	# ensure correct format
	if isinstance(y0_dists, BatchedCategorical):
		p0s = BatchedCategorical.from_binary_probs(p0s)
	if isinstance(y1_dists, BatchedCategorical):
		p1s = BatchedCategorical.from_binary_probs(p1s)
	# Return
	return p0s, p1s

def _evaluate_model_predictions(y, haty, tol=1e-9):
	# Preliminaries
	n = len(y)
	resids = y - haty
	ybar_loo = (n * y.mean() - y) / (n-1)
	resids_null = y - ybar_loo
	# RMSE calculations
	rmse = np.sqrt(np.mean(resids**2))
	rmse_null = np.sqrt(np.mean(resids_null**2))
	oos_r2 = 1 - rmse**2 / rmse_null**2
	# Accuracy/likelihood/MAE calculations
	if set(list(np.unique(y))) == set([0,1]):
		y = y.astype(int)
		# accuracy
		haty_round = np.round(np.clip(haty, 0, 1)).astype(int)
		accuracy = (haty_round == y).astype(float)
		accuracy[haty == 0.5] = 0.5
		accuracy = np.mean(accuracy)
		# null model accuracy
		null_accuracy = (np.round(ybar_loo).astype(int) == y).astype(float)
		null_accuracy[ybar_loo == 0.5] = 0.5
		null_accuracy = np.mean(null_accuracy)
		# likelihood
		haty = np.clip(haty, tol, 1-tol)
		lls = np.log(haty) * y + np.log(1-haty) * (1 - y)
		ybar_loo = np.clip(ybar_loo, tol, 1-tol)
		null_lls = np.log(ybar_loo) * y + np.log(1-ybar_loo) * (1-y)
		# take geometric mean
		likelihood_geom = np.exp(lls.mean())
		null_likelihood_geom = np.exp(null_lls.mean())
		return pd.DataFrame(
			np.array([
				[oos_r2, 0],
				[accuracy, null_accuracy],
				[likelihood_geom, null_likelihood_geom]
			]),
			index=['Out-of-sample R^2', 'Accuracy', 'Likelihood (geom. mean)'],
			columns=['Model', 'No covariates'],
		)
	else:   
		mae = np.mean(np.abs(resids))
		mae_null = np.mean(np.abs(resids_null))
		return pd.DataFrame(
			np.array([
				[oos_r2, 0],
				[rmse, rmse_null],
				[mae, mae_null]
			]),
			index=['Out-of-sample R^2', 'RMSE', 'MAE'],
			columns=['Model', 'No covariates'],
		)

def ridge_loo_resids(
	features, y, ridge_cv_model
):
	# Ensure we add an intercept which is unregularized
	n, p = features.shape
	if ridge_cv_model.fit_intercept:
		# This is not *exact* if we fit an intercept
		# because it assumes the intercept doesn't change
		# but it's realistically pretty close
		X_offset = features.mean(axis=0)
		y_offset = y.mean(axis=0)
		y = y - y_offset; features = features - X_offset
	# Regularization strength
	try:
		alpha = ridge_cv_model.alpha_
	except:
		alpha = ridge_cv_model.alpha
	# Predictions
	Q = features.T @ features + alpha * np.eye(p)
	Qinv = np.linalg.inv(Q)
	hatbeta = Qinv @ features.T @ y
	preds = features @ hatbeta
	# Leave one out residuals
	scales = np.sum((features @ Qinv) * features, axis=1)
	loo_resids = (y - preds) / (1 - scales)
	return loo_resids

class DistReg:
	"""
	A generic class for distributional regressions, meant for subclassing.

	Parameters
	----------
	how_transform : str
		Str specifying how to transform the features before fitting
		the underlying model. One of several options:

		- 'identity': does not transform the features
		- 'intercept': adds an intercept
		- 'interactions' : adds treatment-covariate interactions

		The default is ``interactions``.

	Notes
	-----
	To inherit from this class, simply define the ``fit``
	and ``predict`` functions, ensuring that they match the 
	signature in the docs for this class.
	"""

	def __init__(self, how_transform):
		self.how_transform = how_transform


	def fit(
		self,
		W: np.array,
		X: np.array,
		y: np.array,
		Z: Optional[np.array]=None,
		sample_weight: Optional[np.array]=None,
	):
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
		Z : np.array
			Optional n-length array of instrument values
			for the instrumental variables setting.
		sample_weight : np.array
			Optional n-length array of weights to use when
			fitting the underlying model.

		Notes
		-----
		Not all model_types can be trained using sample_weight.
		For some models (e.g. KNNRegressors), including
		sample_weight will yield an error.
		"""
		raise NotImplementedError()

	def feature_transform(self, W: np.array, X: np.array, Z: Optional[np.array]=None):
		"""
		Transforms the features before feeding them to the base model.

		Parameters
		----------
		X : np.array
			(n, p)-shaped array of covariates.
		W : np.array
			n-length array of treatment/exposure indicators.
		Z : np.array
			Optional n-length array of binary instruments
			for the instrumental variables setting.

		Returns
		-------
		features : np.array
			(n, d)-shaped array of features.
		"""
		# Transformations based on W
		if self.how_transform in ['none', 'identity']:
			features = np.concatenate([W.reshape(-1, 1), X],axis=1)
		elif self.how_transform in ['intercept']:
			features = np.concatenate([W.reshape(-1, 1), np.ones((len(X), 1)), X],axis=1)
		elif self.how_transform in [
			'int', 'interaction', 'interactions'
		]:
			features = np.concatenate([
				W.reshape(-1, 1), W.reshape(-1, 1) * X, X], axis=1
			)
		else:
			raise ValueError(f"Unrecognized transformation {self.how_transform}")

		# Add Z. Note that we cannot add interactions with W
		if Z is None:
			return features
		else:
			Z = Z.reshape(-1, 1)
			if self.how_transform in ['int', 'interaction', 'interactions']:
				return np.concatenate([features, Z, Z * X], axis=1)
			else:
				return np.concatenate([features, Z], axis=1)

	def features_to_WX(self, features: np.array):
		"""
		Inverse of feature_transform.
		"""
		# Raise error
		if self._iv:
			raise NotImplementedError("features_to_XW not implemented for IV regressions.")
		# Reverse transform
		if self.how_transform in ['none', 'identity']:
			return features[:, 0], features[:, 1:]
		elif self.how_transform in ['intercept']:
			return features[:, 0], features[:, 2:]
		else:
			p = int((features.shape[1] - 1) / 2)
			return features[:, 0], features[:, 1:(p+1)]

	def predict(
		self, 
		X: np.array,
		W: np.array,
		Z: np.array,
	):
		"""
		Predicts the conditional law of the outcome.

		Parameters
		----------
		X : np.array
			(n, p)-shaped array of covariates.
		W : np.array
			Optional n-length array of binary treatment indicators.
		Z : np.array
			Optional n-length array of binary instruments
			for the instrumental variables setting.

		Returns
		-------
		y_dists : stats.rv_continuous / stats.rv_discrete
			batched scipy distribution of shape (n,) where the ith
			distribution is the conditional law of :math:`Y_i | X_i, W_i, Z_i`
			(without conditioning on :math:`Z_i` if ``Z`` is not provided).
		"""
		raise NotImplementedError()

	def predict_counterfactuals(self, X: np.array):
		"""
		Predicts counterfactual distributions of Y (outcome).

		Parameters
		----------
		X : np.array
			(n,p)-shaped design matrix.

		Returns
		-------
		y0_dists : np.array
			y0_dists[i] = the law of :math:`Y_i(0) | X_i`.
			Only returned in a setting without instrumental variables.
		y1_dists : np.array
			y1_dists[i] = the law of :math:`Y_i(1) | X_i`.
			Only returned in a setting without instrumental variables.
		ydists : list
			Only returned if trained using instrumental variables.
			Then ydists[z][w] is a batched scipy distribution
			array whose ith element represents the law of
			:math:`Y_i(w) | X_i, W_i(z) = w`.
		"""
		n = len(X)
		# Case 1: instrument variables (4 counterfactuals)
		if self._iv:
			ydists = [[], []]
			for z in [0,1]:
				y0_dists = self.predict(X=X, W=np.zeros(n), Z=z*np.ones(n))
				y1_dists = self.predict(X=X, W=np.ones(n), Z=z*np.ones(n))
				ydists[z] = [y0_dists, y1_dists]
			return ydists
		# Case 2: intent to treat (2 counterfactuals)
		else:
			y0_dists = self.predict(X=X, W=np.zeros(n))
			y1_dists = self.predict(X=X, W=np.ones(n))
			return y0_dists, y1_dists

class ModelSelector():
	"""
	Class which selects a distributional regression model.

	Notes
	-----
	This class selects the model with the best out-of-sample
	R^2. However, by wrapping this class, other selection metrics
	can be used (simply overwrite the ``select_model`` method).
	"""
	def __init__(self):
		pass

	def select_model(
		self,
		models: list[DistReg],
		W: np.array,
		X: np.array,
		y: np.array,
		Z: Optional[np.array]=None,
		sample_weight: Optional[np.array]=None,
		propensities: Optional[np.array]=None,
		**kwargs
	) -> DistReg:
		"""
		Parameters
		----------
		models : list
			list of ``dist_reg.DistReg`` classes.
		W : np.array
			n-length array of binary treatment indicators.
		X : np.array
			(n, p)-shaped array of covariates.
		y : np.array
			n-length array of outcome measurements.
		Z : np.array
			Optional n-length array of binary instrument values
			for the instrumental variables setting.
		propensities : np.array
			Optional n-length array of propensity scores. 
			This argument is only used when ``model_selector``
			is provided.
		sample_weight : np.array
			Optional n-length array of weights to use when
			fitting the underlying model.

		Returns
		-------
		selected_model : ``dist_reg.DistReg``
			The selected model.
		"""
		# Evaluate out-of-sample R^2s
		r2s = np.zeros(len(models))
		for i, model in enumerate(models):
			_, _, oos_preds = cross_fit_predictions(
				W=W, X=X, y=y, Z=Z, 
				sample_weight=sample_weight,
				propensities=propensities,
				model=model,
				model_selector=None,
				verbose=False,
			)
			results = _evaluate_model_predictions(
				y, haty=np.concatenate([x.mean() for x in oos_preds])
			)
			r2s[i] = results.loc['Out-of-sample R^2', 'Model'].item()
		# Pick the best-performing model
		return models[np.argmax(r2s)]


def cross_fit_predictions(
	W: np.array,
	X: np.array,
	y: np.array,
	Z: Optional[np.array]=None,
	S: Optional[np.array]=None,
	propensities: Optional[np.array]=None,
	sample_weight: Optional[np.array]=None,
	nfolds: int=5,
	train_on_selections: bool=True,
	model: Optional[Union[list, DistReg]]=None,
	probs_only: bool=False,
	verbose: bool=False,
	model_selector: Optional[ModelSelector]=None,
):
	"""
	Performs cross-fitting for a model inheriting from ``dist_reg.DistReg.``

	Parameters
	----------
	W : np.array
		n-length array of binary treatment indicators.
	X : np.array
		(n, p)-shaped array of covariates.
	y : np.array
		n-length array of outcome measurements.
	Z : np.array
		Optional n-length array of binary instrument values
		for the instrumental variables setting.
	S : np.array
		Optional n-length array of selection indicators.
	propensities : np.array
		Optional n-length array of propensity scores. 
		This argument is only used when ``model_selector``
		is provided.
	sample_weight : np.array
		Optional n-length array of weights to use when
		fitting the underlying model.
	nfolds : int
		Number of cross-fitting folds to use.
	model : DistReg
		instantiation of ``dist_reg.DistReg`` class. This will
		be copied. E.g., 
		``model=dist_reg.CtsDistReg(model_type='ridge', eps_dist="empirical").``
		Alternatively, one may provide a list of
		``dist_reg.DistReg`` classes and a ``model_selector`` which
		adaptively chooses between them.
	train_on_selections : bool
		If True, trains model only on data where S[i] == 1.
	probs_only : bool
		For binary data, returns P(Y = 1 | X, W) instead of a distribution.
		Defaults to False.
	verbose : bool
		If True, provides progress reports.

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
		S_flag = False
		S = np.ones(n)
	else:
		S_flag = True
		X = np.concatenate([S.reshape(n, 1), X], axis=1)

	# create folds
	starts, ends = create_folds(n=n, nfolds=nfolds)
	# loop through folds
	counterfactual_preds = []# results for different values of W/Z
	oos_preds = [] # results for true W/Z
	fit_models = []
	for ii in vrange(len(starts), verbose=verbose):
		start = starts[ii]; end = ends[ii]
		# Pick out data from the other folds
		not_in_fold = [i for i in np.arange(n) if i < start or i >= end]
		if train_on_selections:
			opt2 = [i for i in not_in_fold if S[i] == 1]
			if len(opt2) == 0:
				warnings.warn(f"S=0 for all folds except {start}-{end}.")
			else:
				not_in_fold = opt2

		# Possibly perform model selection
		fold_args = dict(
			W=W[not_in_fold], 
			X=X[not_in_fold], 
			y=y[not_in_fold],
			Z=None if Z is None else Z[not_in_fold],
			sample_weight=None if sample_weight is None else sample_weight[not_in_fold],
		)
		if isinstance(model, list):
			if model_selector is None:
				model_selector = ModelSelector()
			reg_model = copy.deepcopy(model_selector.select_model(
				models=model,
				propensities=None if propensities is None else propensities[not_in_fold],
				**fold_args,
			))
		else:
			reg_model = copy.deepcopy(model)
		
		# Fit model
		reg_model.fit(**fold_args)

		# predict and append on this fold
		subX = X[start:end].copy(); 
		if S_flag:
			subX[:, 0] = 1 # set selection = 1 for predictions

		# Actual predictions
		oos_pred = reg_model.predict(
			subX, W[start:end], Z=None if Z is None else Z[start:end]
		)
		counterfactual_pred = reg_model.predict_counterfactuals(subX)

		# Save predictions and model fits
		counterfactual_preds.append(counterfactual_pred)
		oos_preds.append(oos_pred)
		fit_models.append(reg_model)

	# Appropriately concatenate counterfactual predictions. 
	# Requires different code for the IV/non-IV case.
	if Z is not None:
		cf_output = [[np.nan, np.nan], [np.nan, np.nan]]
		for z in [0,1]:
			for w in [0,1]:
				cf_output[z][w] = [x[z][w] for x in counterfactual_preds]
				# Concatenate probabilities if we're only interested in them
				if probs_only:
					cf_output[z][w] = np.concatenate(
						[x.probs[:, 1] for x in cf_output[z][w]]
					)

	else:
		cf_output = [[], []]
		for z in [0,1]:
			cf_output[z] = [x[z] for x in counterfactual_preds]
			# Concatenate probabilities if we're only interested in probabilities
			if probs_only:
				cf_output[z] = np.concatenate([x.probs[:, 1] for x in cf_output[z]])



	return cf_output, fit_models, oos_preds


class CtsDistReg(DistReg):
	"""
	Distributional regression for continuous outcomes.

	Parameters
	----------
	model_type : str or sklearn class
		Str specifying a sklearn model class to use; options include
		'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'. One can
		also directly pass an sklearn class, e.g., 
		``model_type=sklearn.ensemble.KNeighborsRegressor``. 
	how_transform : str
		Str specifying how to transform the features before fitting
		the underlying model. One of several options:

		- 'identity': does not transform the features
		- 'intercept': adds an intercept
		- 'interactions' : adds treatment-covariate interactions

		The default is ``interactions``.
	eps_dist : str
		Str specifying the distribution of the residuals. Options include
		['empirical', gaussian', 'laplace', 'expon', 'tdist', 'skewnorm']. 
		Defaults to ``empirical``, which uses the empirical law of the
		residuals of the training data.
	eps_kwargs : dict
		kwargs to ``utilities.parse_dist`` for the residual scipy distribution
	heterosked_model : str or sklearn class
		Str specifying a sklearn model class to use to estimate 
		Var(Y | X) as a function of X. Options are the same as 
		``model_type.`` Defaults to heterosked_model=None, in which
		case homoskedasticity is assumed (although the final bounds
		will still be valid in the presence of heteroskedasticity).
	heterosked_kwargs : dict
		kwargs for the heterosked model. E.g., 
		if ``heterosked_model=knn``,
		heterosked_kwargs could include ``n_neighbors``.
	**model_kwargs : dict
		kwargs for sklearn base model. E.g., if ``model_type=knn``,
		model_kwargs could include ``n_neighbors``.

	Examples
	--------
	Here we instantiate a model which assumes Gaussianity,
	uses a ridge to make predictions and a lasso to estimate
	the heteroskedasticity pattern: ::

		import numpy as np
		import dualbounds
		import sklearn.linear_model

		# Instantiate dist_reg
		cdreg = dualbounds.dist_reg.CtsDistReg(
			# Arguments for main model
			model_type=sklearn.linear_model.RidgeCV,
			fit_intercept=True,
			gcv_mode='auto',
			# How to estimate the law of the residuals
			eps_dist='gaussian',
			# How to estimate Var(Y | X)
			heterosked_model=sklearn.linear_model.LassoCV,
			heterosked_kwargs=dict(cv=5),
		)

		# Fit
		n, p = 300, 20
		W = np.random.binomial(1, 0.5, n)
		X = np.random.randn(n, p)
		y = np.random.randn(n)
		cdreg.fit(W=W, X=X, y=y)

		# Predict on new X
		m = 10
		Xnew = np.random.randn(m, p)
		y0_preds = cdreg.predict(X=Xnew, W=np.zeros(m))
	"""
	def __init__(
		self,
		model_type: Union[str, sklearn.base.BaseEstimator] = 'ridge',
		how_transform: str = 'interactions',
		eps_dist: str = 'empirical',
		eps_kwargs: Optional[dict]=None,
		heterosked_model: str='none',
		heterosked_kwargs: Optional[dict]=None,
		**model_kwargs,
	):
		# Distribution of residuals
		self.eps_dist = eps_dist
		self.eps_kwargs = {} if eps_kwargs is None else eps_kwargs
		self.use_loo = self.eps_kwargs.pop("use_loo", False)
		# Main predictive model
		self.model_type = parse_model_type(model_type, discrete=False)
		self.model_kwargs = model_kwargs
		self.how_transform = str(how_transform).lower()
		## Heteroskedasticity
		# Parse if we want to fit a model for heteroskedasticity
		if heterosked_model is not None:
			if isinstance(heterosked_model, str):
				heterosked_model = heterosked_model.lower()
			self.heterosked = heterosked_model != 'none'
		else:
			self.heterosked = False
		# If so, parse the model type
		if self.heterosked:
			self.sigma2_model_type = parse_model_type(heterosked_model, discrete=False)
			self.sigma2_model_kwargs = heterosked_kwargs
			if self.sigma2_model_kwargs is None:
				self.sigma2_model_kwargs = {}
		# Default kwargs
		if model_type == 'ridge':
			self.model_kwargs['alphas'] = self.model_kwargs.get("alphas", DEFAULT_ALPHAS)
		

	def fit(
		self, 
		W: np.array,
		X: np.array,
		y: np.array,
		Z: Optional[np.array]=None,
		sample_weight: Optional[np.array]=None
	) -> None:
		self._iv = Z is not None
		self.Wtrain = W
		# fit model
		features = self.feature_transform(W=W, X=X, Z=Z)
		self.model = self.model_type(**self.model_kwargs)
		# Possibly use sample_weight
		fit_kwargs = dict() if sample_weight is None else dict(sample_weight=sample_weight)
		self.model.fit(features, y, **fit_kwargs)

		# compute residuals. Use cheap LOO resids for ridge
		if isinstance(self.model, lm.RidgeCV) and self.use_loo:
			self.resids = ridge_loo_resids(features, y=y, ridge_cv_model=self.model)
		else:
			self.resids = y - self.model.predict(features)

		# Estimate E[Var(Y | X)] using residuals.
		self.hatsigma = np.sqrt(np.mean(self.resids**2))
		if self.heterosked:
			self.sigma2_model = self.sigma2_model_type(**self.sigma2_model_kwargs)
			self.sigma2_model.fit(features, self.resids**2, **fit_kwargs)
			self.hatsigma_preds = np.sqrt(np.clip(
				self.sigma2_model.predict(features),
				0.01 * self.hatsigma**2, 
				np.inf
			))
		else:
			self.hatsigma_preds = self.hatsigma

		# If eps_dist == empirical, fit empirical law
		if self.eps_dist == 'empirical':
			norm_resids = self.resids / self.hatsigma_preds
			self.rvals0 = np.sort(norm_resids[W == 0])
			# Center to ensure we don't scale a small mean by a large amount
			# when heterosked=True.
			if sample_weight is None:
				self.rprobs0 = np.ones(len(self.rvals0)) / len(self.rvals0)
			else:
				self.rprobs0 = sample_weight[W == 0].copy()
				self.rprobs0 /= self.rprobs0.sum()
			self.rvals0 -= self.rvals0 @ self.rprobs0
			# Repeat for treatment
			self.rvals1 = np.sort(norm_resids[W == 1])
			if sample_weight is None:
				self.rprobs1 = np.ones(len(self.rvals1)) / len(self.rvals1)
			else:
				self.rprobs1 = sample_weight[W == 1].copy()
				self.rprobs1 /= self.rprobs1.sum()
			self.rvals1 -= self.rvals1 @ self.rprobs1
			# adjust support size for computational reasons
			adj_kwargs = dict(
				new_nvals=self.eps_kwargs.get("nvals", 90), ymin=y.min(), ymax=y.max()
			)
			self.rvals0, self.rprobs0 = utilities._adjust_support_size_unbatched(
				self.rvals0, self.rprobs0, **adj_kwargs
			)
			self.rvals1, self.rprobs1 = utilities._adjust_support_size_unbatched(
				self.rvals1, self.rprobs1, **adj_kwargs
			)

	def predict(self, X: np.array, W: np.array, Z: Optional[np.array]=None):

		features = self.feature_transform(W, X=X, Z=Z)
		mu = self.model.predict(features)
		# heteroskedasticity
		if self.heterosked:
			scale = np.sqrt(np.clip(
				self.sigma2_model.predict(features),
				0.01 * self.hatsigma**2,
				np.inf,
			))
		else:
			scale = self.hatsigma
		# Option 1: nonparametric law of eps
		if self.eps_dist == 'empirical':
			# law of normalized residuals
			# only use empirical residuals from the same treatment/control group
			vals = np.stack(
				[self.rvals0 if Wi == 0 else self.rvals1 for Wi in W],
				axis=0
			)
			vals = vals * np.array([scale]).reshape(-1, 1) + mu.reshape(-1, 1)
			# Create probs and return
			probs = np.stack(
				[self.rprobs0 if Wi == 0 else self.rprobs1 for Wi in W], 
				axis=0
			)
			return utilities.BatchedCategorical(
				vals=vals, probs=probs
			)
		# Option 2: parametric law of eps, return scipy dists
		else:
			return parse_dist(
				self.eps_dist, mu=mu, sd=scale, **self.eps_kwargs,
			)

class QuantileDistReg(DistReg):
	"""
	A continuous distributional regression based on quantile regression.

	Parameters
	----------
	nquantiles : int
		The number of quantiles to fit quantile regressions for.
		Quantiles are evenly spaced between 0 and 1.
	alphas : list or np.array
		List of l1 regularization strengths to use in the quantile
		regression; the final strength is determined by cross-validation.  
		Default is alphas=[0] (no regularization).
	how_transform : str
		Str specifying how to transform the features before fitting
		the underlying model. One of several options:

		- 'identity': does not transform the features
		- 'intercept': adds an intercept
		- 'interactions' : adds treatment-covariate interactions

		The default is ``interactions``.

	Notes
	-----
	This method is computationally expensive for large datasets.
	"""
	def __init__(
		self, 
		nquantiles: int=50,
		alphas: list=[0],
		how_transform: str='interactions',
	):
		self.nq = nquantiles
		self.how_transform = how_transform
		self.quantiles = np.around(np.linspace(0, 1, self.nq), 8)
		self.probs = self.quantiles[1:] - self.quantiles[0:-1]
		self.alphas = alphas

	def fit(
		self,
		W: np.array, 
		X: np.array, 
		y: np.array, 
		Z: Optional[np.array]=None, 
		sample_weight: Optional[np.array]=None
	) -> None:
		self._iv = Z is not None
		# Pick regularization by using CV lasso reg. strength
		features = self.feature_transform(W=W, X=X, Z=Z)
		if len(self.alphas) > 0:
			from sklearn.metrics import d2_pinball_score, make_scorer
			from sklearn.model_selection import cross_validate
		# Fit many quantiles
		self.model = {}
		self.scores = {}
		self.ymin = y.min()
		self.ymax = y.max()
		fit_kwargs = dict() if sample_weight is None else dict(sample_weight=sample_weight)
		for quantile in self.quantiles:
			if quantile not in [0,1]:
				if len(self.alphas) == 1:
					qr = lm.QuantileRegressor(
						alpha=self.alphas[0], quantile=quantile, 
					)
					qr.fit(features, y, **fit_kwargs)
					self.model[quantile] = qr
				# Cross-validate the quantile regression
				# TODO: this currently does not use sample_weight even if provided
				else:
					scores = []
					for alpha in self.alphas:
						qrcand = lm.QuantileRegressor(
							alpha=alpha, quantile=quantile,
						)
						score = cross_validate(
							qrcand, X, y, cv=3,
							scoring=make_scorer(d2_pinball_score, alpha=quantile),
						)
						scores.append(score['test_score'].mean())
					qr = lm.QuantileRegressor(
						alpha=self.alphas[np.argmax(scores)], quantile=quantile, 
					)
					qr.fit(features, y)
					self.model[quantile] = qr
			else:
				self.model[quantile] = None
				self.scores[quantile] = None
			
	def predict(self, X: np.array, W: np.array, Z: Optional[np.array]=None):

		features = self.feature_transform(W, X=X, Z=Z)
		all_preds = [] 
		for quantile in self.quantiles:
			if quantile == 0:
				preds = np.zeros(len(features)) + self.ymin
			elif quantile == 1:
				preds = np.zeros(len(features)) + self.ymax
			else:
				preds = self.model[quantile].predict(features)
			all_preds.append(preds)

		all_preds = np.stack(all_preds, axis=1) # n x self.nq
		# sort to ensure coherence
		all_preds = np.sort(all_preds, axis=1)
		# take centers and return
		yvals = (all_preds[:, 1:] + all_preds[:, 0:-1]) / 2
		probs = np.stack([self.probs for _ in range(len(X))], axis=0)
		return utilities.BatchedCategorical(
			vals=yvals, probs=probs
		)

class BinaryDistReg(DistReg):
	"""
	Binary regression which inherits from ``DistReg``

	Parameters
	----------
	model_type : str or sklearn class.
		Str specifying a sklearn model class to use; options include
		'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'. One can
		also directly pass an sklearn class, e.g., 
		``model_type=sklearn.linear_model.LogisticRegressionCV``.
	how_transform : str
		Str specifying how to transform the features before fitting
		the underlying model. One of several options:

		- 'identity': does not transform the features
		- 'intercept': adds an intercept
		- 'interactions' : adds treatment-covariate interactions

		The default is ``interactions``.
	montonicity : bool
		If True, ensures :math:`P(Y_i(1) = 1 | X_i) - P(Y_i(0) = 1 | X_i)` >= 0.
	monotonicity_margin : float
		When ``self.monotonicity = True``, ensures that
		:math:`P(Y_i(1) = 1 | X_i) - P(Y_i(0) = 1 | X_i)` >= margin. 
		This is important for numerical stability but does not affect 
		validity.
	model_kwargs : dict
		kwargs to sklearn model class.
	"""
	def __init__(
		self, 
		model_type: Union[str, sklearn.base.BaseEstimator] = 'logistic', 
		monotonicity: bool = False,
		monotonicity_margin: float=0.005,
		how_transform: str = 'interactions',
		**model_kwargs
	) -> None:
		self.mtype_raw = model_type
		self.model_type = parse_model_type(model_type, discrete=True)
		self.monotonicity = monotonicity
		self.margin = monotonicity_margin
		self.how_transform = str(how_transform).lower()
		self.model_kwargs = model_kwargs

		## Default kwargs
		if self.mtype_raw == 'lasso':
			self.model_kwargs['penalty'] = 'l1'
			self.model_kwargs['solver'] = self.model_kwargs.get('solver', 'saga')
		elif self.mtype_raw in ['ridge']:
			self.model_kwargs['penalty'] = 'l2'
			self.model_kwargs['solver'] = self.model_kwargs.get('solver', 'lbfgs')
		elif self.mtype_raw in ['elasticnet', 'elastic']:
			self.model_kwargs['penalty'] = 'elasticnet'
			self.model_kwargs['solver'] = self.model_kwargs.get('solver', 'saga')
			self.model_kwargs['l1_ratios'] = np.array([0, 0.5, 1])


	def fit(
		self,
		W: np.array, 
		X: np.array,
		y: np.array,
		Z: Optional[np.array]=None,
		sample_weight: Optional[np.array]=None,
	) -> None:
		self._iv = Z is not None
		# check y is binary
		if set(np.unique(y).tolist()) != set([0,1]):
			raise ValueError(f"y must be binary; instead np.unique(y)={np.unique(y)}")
		# fit 
		features = self.feature_transform(W=W, X=X, Z=Z)
		self.model = self.model_type(**self.model_kwargs)
		# Possibly use sample weights
		fit_kwargs = dict() if sample_weight is None else dict(sample_weight=sample_weight)
		self.model.fit(features, y, **fit_kwargs)

	def predict_proba(
		self, X: np.array, W: np.array, Z: Optional[np.array]=None,
	):
		# return P(Y = 1 | X, W)
		return self.model.predict_proba(
			self.feature_transform(W, X, Z=Z)
		)[:, 1]

	def predict(self, X: np.array, W: np.array, Z: Optional[np.array]=None):
		return BatchedCategorical.from_binary_probs(
			self.predict_proba(X=X, W=W, Z=Z)
		)

	def predict_counterfactuals(self, X: np.array):
		n = len(X)
		ydists = DistReg.predict_counterfactuals(self, X=X)
		## enforce monotonicity
		if not self.monotonicity:
			return ydists

		# Case 1: instrument variables (4 counterfactuals)
		if self._iv:
			for z in [0,1]:
				ydists[z] = _enforce_monotonicity(
					ydists[z][0], ydists[z][1], margin=self.margin
				)
			return ydists
		# Case 2: intent to treat (2 counterfactuals)
		else:
			return _enforce_monotonicity(ydists[0], ydists[1], margin=self.margin)


class MonotoneLogisticReg:
	"""
	A logistic regression solver which ensures that beta[0] >= 0.
	Useful for computing bounds which assume monotonicity.

	Notes
	-----
	This is meant to be used a the underlying model in a
	 :class:`BinaryDistReg` object.
	"""
	def __init__(self, lmda: float=0.001):
		self.lmda = lmda

	def fit(self, X: np.array, y: np.array, solver='CLARABEL', **solver_kwargs):
		"""
		Fits the linear model using a cvxpy backend.

		Parameters
		----------
		X : np.array
			(n,p)-shaped design matrix.
		y : np.array
			n-length vector of binary response.
		solver : str
			Solver for cvxpy to use. Default: CLARABEL.
		solver_kwargs : dict
			Kwargs for the cvxpy solver.
		"""
		# Set up data and losses
		n, p = X.shape
		sig1 = X[:, 0].std()
		zeros = np.zeros(n)
		beta = cp.Variable(p)
		X1beta = X @ beta
		term1 = cp.multiply(y, X1beta)
		term2 = cp.log_sum_exp(cp.vstack([zeros, X1beta]), axis=0)
		term3 = n * self.lmda * cp.sum(cp.power(beta[1:], 2))
		# Objective and problem
		obj = cp.Maximize(cp.sum(term1 - term2) - term3)
		problem = cp.Problem(objective=obj, constraints=[beta[0] >= 0])
		problem.solve(solver=solver, **solver_kwargs)
		# Learn value
		self.beta = beta.value

	def predict_proba(self, X: np.array):
		"""
		Probability predictions for the response.

		Parameters
		----------
		X : np.array
			(n,p)-shaped design matrix.

		Returns
		-------
		probs : np.array
			(n,2)-shaped array of probabilities where
			probs[i, k] = P(Y = k | X = X[i]) for k in {0,1}.
		"""
		mu = X @ self.beta
		p1s = np.exp(mu) / (1 + np.exp(mu))
		return np.stack([1 - p1s, p1s], axis=1)