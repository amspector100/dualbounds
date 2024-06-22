import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import stats
from . import utilities
from . import dist_reg
from .utilities import BatchedCategorical
from . import interpolation, generic
from .delta import _delta_bootstrap_ses
from typing import Optional, Union


### Helper functions ###

def compute_cvar(dists, n: int, alpha: float, lower: bool=True, m: int=1000):
	"""
	Computes cvar using quantile approximation with m values.

	Parameters
	----------
	dists : stats.dist
		scipy distribution function of shape n
	n : int
		Batch dimension
	alpha : array or float	
		float or n-length array
	lower : bool
		If true, computes the lower CVAR. 
		Else computes the upper CVAR.
	m : int
		Number of interpolation points
	
	Returns
	-------
	cvars : array
		n-length array.
		E[Y | Y <= Q_{alpha}(Y)] from Y ~ dists if lower = True.
		If lower = False, replaces <= with >=.
	"""

	if isinstance(alpha, float) or isinstance(alpha, int):
		alpha = alpha * np.ones((n))
	# find quantiles
	if lower:
		qs = np.linspace(1/(m+1), alpha, m)
	else:
		qs = np.linspace(alpha, m/(m+1), m)
	# take average
	qmc = dists.ppf(q=qs)
	if qmc.shape != (m, n):
		raise ValueError(f"Unexpected shape of qmc={qmc.shape}")
	cvar_est = qmc.mean(axis=0)
	return cvar_est

def compute_analytical_lee_bound(
	s0_probs,
	s1_probs,
	y0_dists=None,
	y1_dists=None,
	# optional,
	y0_probs=None,
	y1_probs=None,
	y0_vals=None,
	y1_vals=None,
	m=1000,
):
	"""
	Helper function to compute semi-analytical Lee Bounds.


	Unlike dual bounds, this function is not at all
	robust to model misspecification. The estimand is 

	:math:`E[Y(1) - Y(0) | S(0) = S(1) = 1]`

	where :math:`Y(1), Y(0)` are potential outcomes and 
	:math:`S(1), S(0)` are post-treatment selection events.

	Parameters
	----------
	s0_probs : np.array
		n-length array where s0_probs[i] = P(S(0) = 1 | Xi)
	s1_probs : np.array
		n-length array where s1_probs[i] = P(S(1) = 1 | Xi)
	y0_dists : np.array
		batched scipy distribution of shape (n,) where the ith
		distribution is the conditional law of Yi(0) | S(0) = 1, Xi
	y1_dists : np.array
		batched scipy distribution of shape (n,) where the ith
		distribution is the conditional law of Yi(1) | S(1) = 1, Xi
	y0_vals : np.array
		nvals0-length array of values y0 can take.
	y0_probs : np.array
		(n, nvals0)-length array where
		y0_probs[i, j] = P(Y(0) = yvals0[j] | S(0) = 1, Xi)
	y1_vals : np.array
		(n,nvals1) array of values y1 can take.
	y1_probs : np.array
		(n, nvals1) array where
		y0_probs[i, j] = P(Y(1) = yvals1[j] | S(1) = 1, Xi)
	m : int
		Number of quantile discretizations to use when computing CVAR.
		m = 1000 (default) is more than sufficient.

	Returns
	-------
	agg_bounds : np.array
		(2,)-length array of lower and upper bound. This integrates 
		across all n y0_dists/y1_dists, etc.
	cond_bounds : np.array
		(2, n)-length array where bounds[0,i] is the ith lower bound
		and bounds[1,i] is the ith upper bound conditional on 
		:math:`X_i`.
	"""
	# Parse arguments
	n = s0_probs.shape[0]
	if y0_dists is None:
		y0_dists = BatchedCategorical(
			vals=y0_vals, probs=y0_probs
		)
	if y1_dists is None:
		y1_dists = BatchedCategorical(
			vals=y1_vals, probs=y1_probs
		)
	
	# always-takers share
	alphas = s0_probs / s1_probs
	if np.any(alphas > 1):
		raise ValueError(f"Monotonicity is violated for indices of alphas={alphas[alphas > 1]}")

	# compute E[Y(1) | Y(1) >= Q(alpha), S(1)=1]
	# (or <= depending on the value of lower)
	cvars_lower = compute_cvar(y1_dists, n, alpha=alphas, lower=True, m=m)
	cvars_upper = compute_cvar(y1_dists, n, alpha=1-alphas, lower=False, m=m)
	y0ms = y0_dists.mean()
	cond_bounds = np.stack([cvars_lower - y0ms, cvars_upper - y0ms], axis=0) # per-X bounds
	agg_bounds = np.mean(cond_bounds * s0_probs, axis=1) / np.mean(s0_probs) # aggregated bounds
	return agg_bounds, cond_bounds

def _lee_bound_no_covariates(
	y: np.array, W: np.array, S: np.array, pis: np.array
):
	"""
	Helper function which gets called by the public variant.
	"""
	# compute P(S | W)
	s0_prob = np.array([np.mean(S[W == 0])])
	s1_prob = np.array([np.mean(S[W == 1])])
	s1_prob = np.maximum(s0_prob, s1_prob)

	# compute P(Y(0))
	flags0 = (W == 0) & (S == 1)
	y0_vals = y[flags0]
	y0_probs = 1 / (1-pis[flags0]); y0_probs /= y0_probs.sum()
	inds0 = np.argsort(y0_vals)
	y0_vals = y0_vals[inds0]; y0_probs = y0_probs[inds0]

	# compute P(Y(1))
	flags1 = (W == 1) & (S == 1)
	y1_vals = y[flags1]
	y1_probs = 1 / (1-pis[flags1]); y1_probs /= y1_probs.sum()
	inds1 = np.argsort(y1_vals)
	y1_vals = y1_vals[inds1]; y1_probs = y1_probs[inds1]

	# Construct argument list
	args = dict(
		s0_probs=s0_prob,
		s1_probs=s1_prob,
		y0_probs=y0_probs.reshape(1, -1),
		y0_vals=y0_vals.reshape(1, -1),
		y1_probs=y1_probs.reshape(1, -1),
		y1_vals=y1_vals.reshape(1, -1)
	)
	# compute lower, upper bounds
	return compute_analytical_lee_bound(**args)[1][:, 0]
	

def lee_bound_no_covariates(
	outcome: np.array, 
	treatment: np.array, 
	selections: np.array, 
	propensities: Optional[np.array]=None,
	clusters: Optional[np.array]=None,
	B: int=200,
	alpha: float=0.05,
	verbose=False,
):
	"""
	Computes plug-in Lee bounds without using covariates.

	Parameters
	----------
	outcome : np.array
		n-length array of outcomes (y)
	treatment : np.array
		n-length array of treatments (W).
	selections : np.array
		n-length array of selection indicators (S).
	propensities : np.array
		n-length array of propensity scores (pis).
		Default: all equal to treatment.mean().
	clusters : np.array
		Optional n-length array of clusters, so ``clusters[i] = j``
		indicates that observation i is in cluster j.
	B : int
		Number of bootstrap replications to compute standard errors.
		Defaults to 0 (no standard errors).
	alpha : float
		nominal Type I error level.
	verbose : bool
		Show progress bar while bootstrapping if verbose=True.

	Returns
	-------
	results : dict
		Dictionary containing up to three keys:

		- estimates: 2-length array of lower/upper estimates.
		- ses: 2-length array of lower/upper standard errors.
		- cis: 2-length array of lower/upper confidence intervals.
	"""
	# Infer propensities
	if propensities is None:
		propensities = np.ones(len(treatment)) * treatment.mean()
	# Create estimates
	estimates = _lee_bound_no_covariates(
		y=outcome,
		W=treatment,
		S=selections,
		pis=propensities,
	)
	if B == 0:
		return dict(estimates=estimates)

	# Compute bootstrapped SEs
	data = np.stack([outcome, treatment, propensities, selections], axis=1)
	func = lambda data: _lee_bound_no_covariates(
		y=data[:, 0],
		W=data[:, 1],
		pis=data[:, 2],
		S=data[:, 3],
	)
	ses = utilities.cluster_bootstrap_se(
		data=data,
		clusters=clusters,
		func=func,
		B=B,
		verbose=verbose,
	)[0]
	cis = estimates.copy()
	cis[0] -= stats.norm.ppf(1-alpha/2) * ses[0]
	cis[1] += stats.norm.ppf(1-alpha/2) * ses[1]
	return dict(
		estimates=estimates,
		ses=ses,
		cis=cis,
	)

def lee_delta_method_se(
	sbetas, skappas, sgammas
):			
	# estimate
	hat_beta = sbetas.mean()
	hat_kappa = skappas.mean()
	hat_gamma = sgammas.mean()
	hattheta = (hat_beta - hat_kappa) / hat_gamma
	# standard error
	hatSigma = np.cov(
		np.stack([sbetas, skappas, sgammas], axis=0)
	) # 3 x 3 cov matrix
	grad = np.array([
		1 / hat_gamma,
		- 1 / hat_gamma,
		-(hat_beta - hat_kappa) / (hat_gamma**2)
	])
	# estimate
	se = np.sqrt(grad @ hatSigma @ grad / len(sbetas))
	return hattheta, se

class LeeDualBounds(generic.DualBounds):
	"""
	Computes dual bounds on the ATE under selection bias.

	Precisely, this class bounds 

	:math:`E[Y(1) - Y(0) | S(0) = S(1) = 1]`

	where :math:`Y(1), Y(0)` are potential outcomes and 
	:math:`S(1), S(0)` are post-treatment selection events.
	These bounds assume monotonicity, i.e., 
	:math:`S(1) >= S(0)` a.s. (see Lee 2009).

	Parameters
	----------
	selections : np.array
		n-length array-like of binary selection indicators
	outcome : np.array | pd.Series
		n-length array of outcome measurements (Y).
	treatment : np.array | pd.Series
		n-length array of binary treatment (W).
	covariates : np.array | pd.Series
		(n, p)-shaped array of covariates (X).
	propensities : np.array | pd.Series
		n-length array-like of propensity scores :math:`P(W=1 | X)`. 
		If ``None``, will be estimated from the data.
	outcome_model : str | dist_reg.DistReg
		The model for estimating the law of :math:`Y | X, W`.
		Two options:

		- A str identifier, e.g., 'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'.
		- An object inheriting from ``dist_reg.DistReg``. 

		E.g., when ``outcome`` is continuous, the default is
		``outcome_model=dist_reg.CtsDistReg(model_type='ridge')``.
	propensity_model : str | sklearn classifier
		How to estimate the propensity scores if they are not provided.
		Two options:

		- A str identifier, e.g., 'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'.
		- An sklearn classifier, e.g., ``sklearn.linear_model.LogisticRegressionCV()``.
	selection_model : str | dist_reg.BinaryDistReg
		How to estimate the selection probabilities :math:`P(S =1 | W, X)`.
		Two options:

		- A str identifier, i.e., 'monotone_logistic', 'ridge', 'lasso'.
		- An object inheriting from ``dist_reg.BinaryDistReg``.

		The default is ``monotone_logistic``. 
	support : np.array
		Optional support of the outcome, if known and discrete.
		Defaults to ``None`` (inferred from the data).
	model_kwargs : dict
		Additional kwargs for the ``outcome_model``, e.g.,
		``feature_transform``. See 
		:class:`dualbounds.dist_reg.CtsDistReg` or 
		:class:`dualbounds.dist_reg.BinaryDistReg` for more kwargs.
	"""	

	def __init__(
		self,
		selections: Union[np.array, pd.Series],
		*args, 
		selection_model: Optional[Union[str, dist_reg.BinaryDistReg]]=None,
		**kwargs
	):
		# Main initialization
		kwargs['f'] = None
		super().__init__(*args, **kwargs)
		# Initialization for S
		self.selection_model = selection_model
		if isinstance(selections, pd.Series) or isinstance(selections, pd.DataFrame):
			selections = selections.values.reshape(self.n)
		self.S = utilities._binarize_variable(selections, var_name='selections')
	
	def _ensure_feasibility(
		self,
		i,
		nu0,
		nu1,
		lower,
		ymin,
		ymax,
		grid_size=10000,
	):
		"""
		ensures nu0 + nu1 <= fvals (if lower)
		or nu0 + nu1 >= fvals (if upper)
		by performing a gridsearch.

		Parameters
		----------
		nu1 : nvals1-length array
			note nu1[0] is the dual variable for when s1=1.

		Returns
		-------
		new_nu0 : np.array
			``nvals0`` length array of new dual vars. for Y(0)
		new_nu1 : np.array
			``nvals1`` length array of new dual vars. for Y(1)
		dx : float
			Constant which can be subtracted off to obtain valid 
			dual variables. I.e., new_nk = nuk - dx/2.
		"""
		y1_vals = self.y1_vals_adj[i]
		if y1_vals[0] != 0:
			raise ValueError(
				"Expected y1_vals[0] to equal zero; this should be the case when s1 = 0."
			)
		dxs = []
		new_yvals = np.linspace(ymin, ymax, grid_size)
		interp_nu = self.interp_fn(
			x=y1_vals[1:], y=nu1[1:], newx=new_yvals,
		)
		# only three options due to monotonicity
		# if lower: nu0[s0] + nu1[s1 * y1] <= s0 s1 y1
		# if not lower: nu0[s0] + nu1[s1 * y1] >= s0 s1 y1
		for s0, s1 in zip(
			[0, 0, 1], [0, 1, 1]
		):
			if s1 == 0:
				dxs.append(nu0[s0] + nu1[0])
			else:
				deltas = nu0[s0] + interp_nu - s0 * s1 * new_yvals
				if lower:
					dxs.append(np.max(deltas))
				else:
					dxs.append(np.min(deltas))

		# return valid dual variables
		if lower:
			dx = np.max(np.array(dxs))
		else:
			dx = np.min(np.array(dxs))
		return nu0 - dx/2, nu1 - dx/2, dx

	def compute_dual_variables(
		self,
		s0_probs,
		s1_probs,
		y1_dists=None,
		y1_vals=None,
		y1_probs=None,
		verbose=False,
		nvals=100,
		ymin=None,
		ymax=None,
		**kwargs,
	):
		"""
		Estimates dual variables using the outcome model.

		We generally recommend that the user call .fit() instead of 
		calling this function directly.

		Parameters
		----------
		s0_probs : np.array
			n-length array where s0_probs[i] = :math:`P(S_i(0) = 1 | X_i)`.
		s1_probs : np.array
			n-length array where s1_probs[i] = :math:`P(S_i(1) = 1 | X_i)`.
		y1_dists : list
			The ith distribution of y1_dists represents the conditional
			law of :math:`Y_i(1) | X_i, S_i(1) =1`. There are two input formats:

			- batched scipy distribution of shape (n,)
			- list of scipy dists whose shapes add up to n.

		y1_vals : np.array
			(n, nvals1)-length array where ``y1_vals[i]`` is the support 
			of :math:`Y_i(1)`.	Ignored if ``y1_dists`` is provided.
		y1_probs : np.array
			(n, nvals1)-length array where ``y1_probs[i, j]``
			is the estimated probability that :math:`Y_i(1)`
			equals ``y1_vals[i, j].``
		verbose : bool
			If True, prints progress reports.
		nvals : int
			Number of values to use when discretizing Y(1).
		ymin : float
			Minimum value of Y(1) to use numerically.
		ymax : float
			Maximum value of Y(1) to use numerically.
		kwargs : dict
			kwargs for _ensure_feasibility method.
			Includes ymin, ymax, grid_size.
		"""
		# Constants for solver
		if verbose:
			print("Estimating optimal dual variables.")
		n = s0_probs.shape[0]
		self.nvals0 = 2 # because S is binary
		if self.discrete:
			self.nvals1 = len(self.support) + 1
		else:
			self.nvals1 = nvals
		self.interp_fn = interpolation.linear_interpolate
		ymin = ymin if ymin is not None else self.y.min()
		ymax = ymax if ymax is not None else self.y.max()

		# discretize if Y is continuous
		if y1_vals is None or y1_probs is None:
			# tolerance parameter
			min_quantile = min([np.min(s0_probs/s1_probs), np.min(1 - s0_probs/s1_probs)])
			# law of Y(1) | S(1) = 1, Xi
			y1_vals, y1_probs = self._discretize(
				y1_dists, 
				nvals=self.nvals1-1, 
				min_quantile=min_quantile/2,
				ninterp=1,
				ymin=ymin,
				ymax=ymax,
			)

		# ensure y1_vals, y1_probs are sorted
		self.y1_vals, self.y1_probs = utilities._sort_disc_dist(vals=y1_vals, probs=y1_probs)
		del y1_vals, y1_probs

		## Adjust to make it law of Y(1) S(1) | Xi
		# note: it is important that the value when S(1) = 0
		# is the first value on the second axis
		# so that the monotonicity constraint is correct.
		self.y1_vals_adj = np.concatenate(
			[np.zeros((n, 1)), self.y1_vals], axis=1
		)
		self.y1_probs_adj = np.concatenate(
			[
				1 - s1_probs.reshape(-1, 1),
				s1_probs.reshape(-1, 1) * self.y1_probs
			], 
			axis=1
		)

		# useful constants
		s0_vals = np.array([0, 1]).reshape(-1, 1)

		# Initialize results
		self.n = self.y1_vals.shape[0]
		self.nu0s = np.zeros((2, n, self.nvals0)) # first dimension = [lower, upper]
		self.nu1s = np.zeros((2, n, self.nvals1))
		self.hatnu0s = np.zeros((2, self.n))
		self.hatnu1s = np.zeros((2, self.n)) 
		# estimated cond means of nu0s, nu1s
		self.c0s = np.zeros((2, self.n))
		self.c1s = np.zeros((2, self.n))
		# objective values
		self.objvals = np.zeros((2, self.n))
		self.dxs = np.zeros((2, self.n))
		# loop through
		for i in utilities.vrange(self.n, verbose=verbose):
			# set parameter values
			fvals = (
				s0_vals * self.y1_vals_adj[i].reshape(1, -1)
			).astype(float)
			# below is the max val. used instead of inf to relax 
			# constraints. this provably has no effect compared
			# to using np.inf, but allows the use of ot instead
			# of cvxpy, leading to a substantial speedup.
			max_fval = np.abs(fvals).max() * 1e7
			# helpful concatenation
			probs0 = np.array([1 - s0_probs[i], s0_probs[i]])
			# solve 
			for lower in [1, 0]:
				# Relax constraints due to monotonicity
				fvals[1][0] = max_fval if lower else -max_fval
				nu0x, nu1x, objval = self._solve_single_instance(
					i=i,
					probs0=probs0,
					probs1=self.y1_probs_adj[i],
					y0_vals=np.array([0,1]),
					y1_vals=self.y1_vals_adj[i],
					fvals=fvals,
					not_binding=np.zeros(fvals.shape).astype(bool),
					lower=lower
				)
				self.objvals[1 - lower, i] = objval
				if not self.discrete:
					nu0x, nu1x, dx = self._ensure_feasibility(
						i=i, nu0=nu0x, nu1=nu1x,
						lower=lower, ymin=ymin, ymax=ymax,
						**kwargs,
					)
				else:
					dx = 0

				# Save intermediate quantities
				self.nu0s[1 - lower, i] = nu0x
				self.nu1s[1 - lower, i] = nu1x
				self.c0s[1 - lower, i] = nu0x @ probs0
				self.c1s[1 - lower, i] = nu1x @ self.y1_probs_adj[i]
				self.dxs[1 - lower, i] = dx

		self._compute_realized_dual_variables(y=self.y, S=self.S)

	def _compute_realized_dual_variables(self, y=None, S=None):
		y = self.y if y is None else y
		S = self.S if S is None else S
		### Compute realized hatnu1s/hatnu0s
		self.hatnu0s = np.zeros((2, self.n))
		self.hatnu1s = np.zeros((2, self.n))
		for i in range(self.n):
			for lower in [0, 1]:
				nu0x = self.nu0s[1-lower, i]
				nu1x = self.nu1s[1-lower, i]
				# Set values
				self.hatnu0s[1 - lower, i] = nu0x[S[i]]
				if S[i] == 0:
					self.hatnu1s[1 - lower, i] = nu1x[0]
				if not self.discrete and S[i] == 1:
					self.hatnu1s[1 - lower, i] = self.interp_fn(
						x=self.y1_vals_adj[i][1:], y=nu1x[1:], newx=y[i],
					)[0]
				if self.discrete and S[i] == 1:
					j = np.argmin(np.abs(self.y1_vals_adj[i][1:] - y[i]))
					self.hatnu1s[1 - lower, i] = nu1x[j+1]

	def cross_fit(
		self,
		nfolds: int=5,
		suppress_warning: bool=False,
		verbose: bool=True,
	):
		"""
		Cross-fits the outcome and selection models.

		Parameters
		----------
		nfolds : int
			Number of folds to use in cross-fitting.
		suppress_warning : bool
			If True, suppresses the warning about manual crossfitting.
		verbose : bool
			If True, prints progress reports.

		Returns
		-------
		s0_probs : np.array
			n-length array where s0_probs[i] = :math:`P(S_i(0) = 1 | X_i)`.
		s1_probs : np.array
			n-length array where s1_probs[i] = :math:`P(S_i(1) = 1 | X_i)`.
		y0_dists : np.array
			list of batched scipy distributions whose shapes sum to n.
			the ith dist. is the conditional law of :math:`Y_i(0) | S_i(0) = 1, X_i`.
		y1_dists : list
			list of batched scipy distributions whose shapes sum to n.
			the ith dist. is the conditional law of :math:`Y_i(1) | S_i(1) = 1, X_i`.
		"""
		# estimate selection probs
		if self.s0_probs is None or self.s1_probs is None:
			if self.selection_model is None:
				self.selection_model = 'monotone_logistic'
			self.selection_model = generic.get_default_model(
				outcome_model=self.selection_model, 
				# the following args are ignored if selection_model already
				# inherits from dist_reg.DistReg class
				support=set([0,1]), 
				discrete=True,
				monotonicity=True, 
				how_transform='intercept',
			)
			if verbose:
				print("Cross-fitting the selection model.")
			sout = dist_reg.cross_fit_predictions(
				W=self.W, X=self.X, y=self.S, 
				nfolds=nfolds, 
				model=self.selection_model,
				verbose=verbose,
				probs_only=True,
			)
			counterfactuals, self.selection_model_fits, self.S_oos_preds = sout
			self.s0_probs, self.s1_probs = counterfactuals
		elif not suppress_warning:
			warnings.warn(
				generic.CROSSFIT_WARNING.replace("y0_", "s0_").replace("y1_", "s1_")
			)

		# Estimate outcome model
		if self.y0_dists is None or self.y1_dists is None:
			self.outcome_model = generic.get_default_model(
				discrete=self.discrete, support=self.support, outcome_model=self.outcome_model,
			)
			if verbose:
				print("Cross-fitting the outcome model.")
			yout = dist_reg.cross_fit_predictions(
				W=self.W, X=self.X, S=self.S, y=self.y, 
				nfolds=nfolds, 
				model=self.outcome_model,
				verbose=verbose,
			)
			counterfactuals, self.model_fits, self.oos_dist_preds = yout
			self.y0_dists, self.y1_dists = counterfactuals
		elif not suppress_warning:
			warnings.warn(generic.CROSSFIT_WARNING)

		# return
		return self.s0_probs, self.s1_probs, self.y0_dists, self.y1_dists


	def fit(
		self,
		nfolds=5,
		alpha=0.05,
		aipw=True,
		s0_probs=None,
		s1_probs=None,
		y0_dists=None,
		y1_dists=None,
		suppress_warning=False,
		verbose=True,
		**solve_kwargs,
	):
		"""
		Main function which (1) performs cross-fitting, (2) computes 
		optimal dual variables, and (3) computes final dual bounds.

		Parameters
		----------
		nfolds : int
			Number of folds to use when cross-fitting. Defaults to 5,
		alpha : float
			Nominal coverage level. Defaults to 0.05.
		aipw : bool
			If true, returns AIPW estimator.
		s0_probs : np.array
			Optional n-length array where s0_probs[i] = 
			:math:`P(S_i(0) = 1 | X_i)`.
			If not provided, will be estimated from the data.
		s1_probs : np.array
			Optional n-length array where s1_probs[i] = 
			:math:`P(S_i(1) = 1 | X_i)`.
			If not provided, will be estimated from the data.
		y0_dists : np.array
			Optional list of batched scipy distributions whose shapes sum to n.
			the ith dist. is the conditional law of 
			:math:`Y_i(0) | S_i(0) = 1, X_i`.
			If not provided, will be estimated from the data.
		y1_dists : np.array
			Optional list of batched scipy distributions whose shapes sum to n.
			The ith dist. is the conditional law of 
			:math:`Y_i(1) | S_i(1) = 1, X_i`.
			If not provided, will be estimated from the data.
		suppress_warning : bool
			If True, suppresses warning about cross-fitting.
		verbose : bool
			If True, gives occasional progress reports..
		solve_kwargs : dict
			kwargs to self.compute_dual_variables(), 
			e.g., ``verbose``, ``nvals``, ``grid_size``

		Returns
		-------
		self : object
		"""
		# Save data
		self.s0_probs, self.s1_probs = s0_probs, s1_probs
		self.y0_dists, self.y1_dists = y0_dists, y1_dists

		# if pis not supplied: will use cross-fitting
		if self.pis is None:
			self.fit_propensity_scores(verbose=verbose, nfolds=nfolds)

		# fit outcome models using cross-fitting
		self.cross_fit(
			verbose=verbose, nfolds=nfolds, suppress_warning=suppress_warning,
		)

		# compute dual variables
		self.compute_dual_variables(
			s0_probs=self.s0_probs,
			s1_probs=self.s1_probs,
			y1_dists=self.y1_dists,
			ymin=self.y.min(),
			ymax=self.y.max(),
			verbose=verbose,
			**solve_kwargs,
		)

		# compute dual bounds
		self._compute_final_bounds(aipw=aipw, alpha=alpha)
		return self

	def _compute_final_bounds(self, aipw=True, alpha=0.05):
		"""
		Computes final bounds based in (A)IPW summands,
		using the delta method for E[Y(1) - Y(0) | S(1) = S(0) = 1].
		Uses the bootstrap for clustered standard errors.
		"""
		self._compute_ipw_summands()
		summands = self.aipw_summands if aipw else self.ipw_summands
		self._compute_cond_means()
		self.y0s0_cond_means = self.mu0 * self.s0_probs
		self.s_probs = self.s0_probs.copy()
		self.s_probs[self.W == 1] = self.s1_probs[self.W == 1]
		ests = []
		ses = []
		bounds = []
		scale = stats.norm.ppf(1-alpha/2)
		# kappa = E[Y(0) S(0)]
		skappas = (1 - self.W) * (self.y * self.S - self.y0s0_cond_means) 
		skappas = skappas / (1-self.pis)
		skappas += self.y0s0_cond_means
		self.skappas = skappas
		# gamma = E[S(0)]
		sgammas = (1-self.W) * (self.S - self.s0_probs) / (1-self.pis)
		sgammas += self.s0_probs
		self.sgammas = sgammas
		if self.clusters is None:
			for lower in [1, 0]:
				# beta = part. identifiable component E[Y(1) S(0)]
				sbetas = summands[1-lower]
				hattheta, se = lee_delta_method_se(
					sbetas=sbetas, skappas=skappas, sgammas=sgammas,
				)
				ests.append(hattheta)
				ses.append(se)
				if lower:
					bounds.append(hattheta - scale * se)
				else:
					bounds.append(hattheta + scale * se)

			self.estimates = np.array(ests)
			self.ses = np.array(ses)
			self.cis = np.array(bounds)
			return dict(
				estimates=self.estimates,
				ses=self.ses,
				cis=self.cis
			)
		else:
			self.estimates, self.ses, self.cis = _delta_bootstrap_ses(
				h=lambda fval, z0, z1: (fval - z0[0]) / z0[1],
				summands=summands,
				z0summands=np.stack([skappas, sgammas], axis=1),
				z1summands=np.zeros((self.n, 1)),
				clusters=self.clusters,
				alpha=alpha,
			)
			# Return
			return dict(
				estimates=self.estimates,
				ses=self.ses,
				cis=self.cis,
			)


	def summary(self, minval=-np.inf, maxval=np.inf):
		"""
		Prints a summary of main results from the class.

		Parameters
		----------
		minval : float
			Analytical lower bound on estimand used to clip results. 
			Defaults to -np.inf.
		maxval : float
			Analytical upper bound on estimand used to clip results.
			Defaults to np.inf.
		"""
		print("___________________Inference_____________________")
		print(self.results(minval=minval, maxval=maxval))
		print()
		print("________________Selection model__________________")
		sumstats = dist_reg._evaluate_model_predictions(
			y=self.S, haty=self.s_probs
		)
		print(sumstats)
		print()
		print("_________________Outcome model___________________")
		self._compute_oos_resids()
		sumstats = dist_reg._evaluate_model_predictions(
			y=self.y[self.S == 1], haty=self.oos_preds[self.S == 1],
		)
		print(sumstats)
		print()
		print("________________Treatment model__________________")
		sumstats = dist_reg._evaluate_model_predictions(
			y=self.W, haty=self.pis
		)
		print(sumstats)
		print()