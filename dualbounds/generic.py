import copy
import warnings
import numpy as np
import ot
from scipy import stats
from . import utilities, dist_reg, interpolation
from .utilities import BatchedCategorical
import pandas as pd
import cvxpy as cp
# typing
from scipy.stats._distn_infrastructure import rv_generic
import sklearn.base
from typing import Optional, Union

MIN_NVALS = 7
DISC_THRESH = 2 # treat vars. with <= DISC_THRESH values as discrete
CROSSFIT_WARNING = """
==================================================
Not fitting a model because y0_dists/y1_dists were
directly provided. Please ensure cross-fitting is
employed correctly, else inference will be invalid
(see https://arxiv.org/abs/2310.08115). To suppress
this warning, set ``suppress_warning=True``.
==================================================
"""

def get_default_model(discrete, support, outcome_model=None, **model_kwargs):
	# Prevent errors when eps_dist is provided for discrete data
	if discrete:
		model_kwargs.pop("eps_dist", None)

	# Handle the case where we have a list of outcome models
	if isinstance(outcome_model, list):
		return [
			get_default_model(outcome_model=x, discrete=discrete, support=support, **model_kwargs)
			for x in outcome_model
		]
	if isinstance(outcome_model, dist_reg.DistReg):
		return outcome_model
	outcome_model = 'ridge' if outcome_model is None else outcome_model
	if not discrete:
		return dist_reg.CtsDistReg(
			model_type=outcome_model, **model_kwargs
		)
	elif discrete and set(support) == set([0, 1]):
		return dist_reg.BinaryDistReg(
			model_type=outcome_model, **model_kwargs
		)
	else:
		raise NotImplementedError("Currently no default for non-binary discrete data")

def infer_discrete(discrete, support, y):
	n = len(y)
	### Check if discrete
	if n <= 10:
		if discrete is None:
			raise ValueError("Please specify the value of discrete as n <= 10")
		if discrete and support is None:
			raise ValueError("Please specify the value of support as n <= 10")
	if support is None:
		support = np.unique(y)
	if discrete is None:
		if len(support) <= DISC_THRESH:
			discrete = True
		else:
			discrete = False

	# Adjust support to avoid being misleading
	# in the continuous case
	if not discrete:
		support = None
	return discrete, support

def _default_ylims(
	y, 
	y0_min=None,
	y1_min=None,
	y0_max=None,
	y1_max=None,
	**kwargs # ignored
):
	# Computes default bounds on y
	if y0_min is None:
		y0_min = 1.5 * y.min() - y.max() / 2
	if y1_min is None:
		y1_min = 1.5 * y.min() - y.max() / 2
	if y0_max is None:
		y0_max = 1.5 * y.max() - y.min() / 2
	if y1_max is None:
		y1_max = 1.5 * y.max() - y.min() / 2
	return y0_min, y1_min, y0_max, y1_max

def _dualvals_are_too_large(nu0, nu1, almost_inf):
	return max(np.max(np.abs(nu0)), np.max(np.abs(nu1))) > 0.1 * almost_inf

class DualBounds:
	"""
	Computes dual bounds on :math:`E[f(Y(0),Y(1), X)].`

	Here, :math:`X` are covariates and :math:`Y(0), Y(1)` are 
	potential outcomes.

	Parameters
	----------
	f : function
		Function which defines the partially identified estimand.
		Must be a function of three arguments: y0, y1, x 
		(in that order). E.g.,
		``f = lambda y0, y1, x : y0 <= y1``
	outcome : np.array | pd.Series
		n-length array of outcome measurements (Y).
	treatment : np.array | pd.Series
		n-length array of binary treatment (W).
	covariates : np.array | pd.Series
		(n, p)-shaped array of covariates (X).
	propensities : np.array | pd.Series
		n-length array of propensity scores :math:`P(W=1 | X)`. 
		If ``None``, will be estimated from the data.
	clusters : np.array | pd.Series
		Optional n-length array of clusters, so ``clusters[i] = j``
		indicates that observation i is in cluster j.
	outcome_model : str | dist_reg.DistReg | list
		The model for estimating the law of :math:`Y | X, W`.
		Three options:

		- A str identifier, e.g., 'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'.
		- An object inheriting from ``dist_reg.DistReg``. 
		- A list of ``dist_reg.DistReg`` objects to automatically choose between.

		E.g., when ``outcome`` is continuous, the default is
		``outcome_model=dist_reg.CtsDistReg(model_type='ridge')``.
	propensity_model : str | sklearn classifier
		How to estimate the propensity scores if they are not provided.
		Two options:

		- A str identifier, e.g., 'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'.
		- An sklearn classifier, e.g., ``sklearn.linear_model.LogisticRegressionCV()``.
	model_selector : dist_reg.ModelSelector
		A ModelSelector object which can choose between several outcome models.
		The default performs within-fold nested cross-validation. Note: this
		argument is ignored unless ``outcome_model`` is a list.
	discrete : bool
		If True, treats the outcome as a discrete variable. 
		Defaults to ``None`` (inferred from the data).
	support : np.array
		Optional support of the outcome, if known and discrete.
		Defaults to ``None`` (inferred from the data).
	support_restriction : function
		Boolean-valued function of y0, y1, x where 
		``support_restriction(y0, y1, x) = False`` asserts that 
		y0, y1, x is not in the support of :math:`Y(0), Y(1), X`.
		Defaults to ``None`` (no a-priori support restrictions).
		See the user guide for important usage tips.
	model_kwargs : dict
		Additional kwargs for the ``outcome_model``, e.g.,
		``feature_transform``. See 
		:class:`dualbounds.dist_reg.CtsDistReg` or 
		:class:`dualbounds.dist_reg.BinaryDistReg` for more kwargs.

	Notes
	-----
	``DualBounds`` will do limited preprocessing to (e.g.) create 
	dummies for discrete covariates. However, we recommended doing
	custom preprocessing for optimal results.

	Examples
	--------
	Here we fit DualBounds on :math:`P(Y(0) < Y(1))` based on
	synthetic regression data: ::
		import dualbounds as db

		# Generate synthetic data
		data = db.gen_data.gen_regression_data(n=900, p=30)

		# Initialize dual bounds object
		dbnd = db.generic.DualBounds(
			f=lambda y0, y1, x: y0 < y1,
			covariates=data['X'],
			treatment=data['W'],
			outcome=data['y'],
			propensities=data['pis'],
			outcome_model='ridge',
		)

		# Compute dual bounds and observe output
		dbnd.fit(alpha=0.05).summary()
	"""
	def __init__(
		self, 
		f: callable,
		outcome: Union[np.array, pd.Series],
		treatment: Union[np.array, pd.Series],
		covariates: Optional[Union[np.array, pd.DataFrame]]=None,
		propensities: Optional[Union[np.array, pd.Series]]=None,
		clusters: Optional[Union[np.array, pd.Series]]=None,
		outcome_model: Union[str, dist_reg.DistReg, list]='ridge',
		propensity_model: Union[str, sklearn.base.BaseEstimator]='ridge',
		model_selector: Optional[dist_reg.ModelSelector]=None,
		discrete: Optional[np.array]=None,
		support: Optional[np.array]=None,
		support_restriction: Optional[callable]=None,
		**model_kwargs,
	) -> None:
		# Estimand
		self.f = f

		### Process outcome
		self.y = outcome
		if isinstance(self.y, pd.Series) or isinstance(self.y, pd.DataFrame):
			self.y = self.y.values
		if np.any(np.isnan(self.y)):
			raise ValueError("outcome contains nans")
		if len(self.y.shape) > 1:
			if len(self.y.shape) == 2 and self.y.shape[1] == 1:
				self.y = self.y.flatten()
			else:
				raise ValueError("outcome should be a flat array not a matrix")
		self.n = len(self.y)

		### Treatment
		if isinstance(treatment, pd.Series) or isinstance(treatment, pd.DataFrame):
			treatment = treatment.values.reshape(self.n)
		self.W = utilities._binarize_variable(treatment, var_name='treatment')

		### Process covariates
		self.X = covariates
		if self.X is None:
			self.X = np.zeros((self.n, 1))
		# limited preprocessing of covariates
		if isinstance(self.X, pd.DataFrame):
			self.X = utilities.process_covariates(self.X)
			self.cov_names = self.X.columns
			self.X = self.X.values
		else:
			if len(self.X.shape) == 1:
				self.X = self.X.reshape(-1, 1)
			self.cov_names = np.arange(self.X.shape[1])
		# fill NAs if existing
		if np.any(np.isnan(self.X)):
			self.X = self.X.copy()
			naninds = np.where(np.isnan(self.X))
			self.X[naninds] = np.take(np.nanmean(self.X, axis=0), naninds[1])

		### process propensities
		self.pis = propensities
		if isinstance(self.pis, pd.Series) or isinstance(self.pis, pd.DataFrame):
			self.pis = self.pis.values.reshape(self.n)
		if self.pis is not None:
			if np.any(np.isnan(self.pis)):
				raise ValueError(f"propensities (pis) are provided but contains missing values")
			if np.any(self.pis < 0) or np.any(self.pis > 1):
				raise ValueError(f"propensities (pis) do not lie within [0,1]")

		### Clusters
		if clusters is not None:
			if isinstance(clusters, pd.Series) or isinstance(clusters, pd.DataFrame):
				clusters = clusters.values.reshape(self.n)
		self.clusters = clusters

		### Check if discrete
		self.discrete, self.support = infer_discrete(
			discrete=discrete, support=support, y=self.y,
		)
		self.outcome_model = outcome_model
		self.propensity_model = propensity_model
		self.model_selector = model_selector
		self.model_kwargs = model_kwargs

		## Support restrictions
		self.support_restriction = support_restriction

		## Initialize core objects
		self.y0_dists = None
		self.y1_dists = None
		self.oos_dist_preds = None

	def _apply_ot_fn(
		self, fn: callable, y0: Union[float, np.array], y1: Union[float, np.array], **kwargs
	):
		"""
		Applies fn and infers appropriate broadcasting for the OT setting.

		Parameters
		----------
		y0 : float | np.array
		y1 : float | np.array
		kwargs : dict
			Other parameters, e.g., x, or w0 and w1 in the IV setting
			(since DualIVBounds inherits this function).

		Returns
		-------
		fn_val : float | np.array
			Array of shape (len(y0) x len(y1)).

		Notes
		-----
		Internally, this is used to apply self.f and self.support_restriction.
		"""
		y0_haslength = utilities.haslength(y0)
		y1_haslength = utilities.haslength(y1)
		# Handle the scalar case
		if not y0_haslength and not y1_haslength:
			return fn(y0=y0, y1=y1, **kwargs)
		# Otherwise vectorize appropriately 
		if not y0_haslength:
			y0 = np.array([y0])
		if not y1_haslength:
			y1 = np.array([y1])

		# Adding extra zeros ensures correct broadcasting
		orig = fn(y0=y0.reshape(-1, 1), y1=y1.reshape(1, -1), **kwargs)
		return orig + np.zeros((len(y0), len(y1)))
		
	def _discretize(
		self, 
		ydists,
		nvals,
		ymin,
		ymax,
		min_quantile=0.001,
		ninterp=None,
		min_yprob=1e-8,
	):
		"""
		Helper method which discretizes Y before solving +
		interpolating to obtain dual variables.
		
		Notes
		-----
		For discrete Y: extracts conditional PMF/support points.
		For continuous Y: discretizes ydists along evenly spaced
		quantiles. Also adds extra points near the edge of the
		support to ensure numerical stability.
		"""
		# allow for batched setting
		if not isinstance(ydists, list):
			ydists = [ydists]

		### Discrete case ###
		if self.discrete:
			yvals = []
			yprobs = []
			for ydist in ydists:
				if not isinstance(ydist, utilities.BatchedCategorical):
					msg = "self.discrete=True, so the predicted ydists must be a BatchedCategorical"
					msg += "distributions. To get discrete-like behavior using other distribution"
					msg += "types, set discrete=False and grid_size=0."
					raise ValueError(msg)
				yvals.append(ydist.vals)
				yprobs.append(ydist.probs)
			return np.concatenate(yvals, axis=0), np.concatenate(yprobs, axis=0)


		### Continuous case ###
		if ninterp is None:
			ninterp = min(max(int(0.1 * (nvals-2)), 1), 5)
		# Case 2(a): we already have a discrete distribution
		if isinstance(ydists[0], utilities.BatchedCategorical):
			for ii, ydist in enumerate(ydists):
				if not isinstance(ydist, utilities.BatchedCategorical):
					raise ValueError(
						f"ydists[0] is a BatchedCategorical, ydists[{ii}] is not---mixing cts/discrete dists is unsupported."
					)
			# Extract and adjust everything to ensure the same support size
			nvals_adj = nvals - 2 * ninterp
			dists_adj = [
				utilities.adjust_support_size(
					vals=ydist.vals, probs=ydist.probs, new_nvals=nvals_adj, ymin=ymin, ymax=ymax
				)
				for ydist in ydists
			]
			yvals = np.concatenate([x[0] for x in dists_adj], axis=0)
			yprobs = np.concatenate([x[1] for x in dists_adj], axis=0)

		# Case 2(b): we have a continuous distribution
		else:
			if nvals <= MIN_NVALS:
				raise ValueError(f"nvals must be larger than {nvals}.")
			
			# make sure we get small enough quantiles
			alpha = min_quantile
			if alpha is None:
				alpha = 1 / (2*nvals)
			alpha = min(1/(2*nvals), max(alpha, 1e-8))

			# num of interp. pts between min/max quantiles
			# and ymin/ymax, added to ensure feasbility
			nvals_adj = nvals - 2 * ninterp - 3

			### Main discretization based on evenly spaced quantiles
			# choose endpoints of bins for disc. approx
			endpoints = np.sort(np.concatenate(
				[[0, alpha],
				np.linspace(1/nvals_adj, (nvals_adj-1)/nvals_adj, nvals_adj),
				[1-alpha, 1]],
			))
			qs = (endpoints[1:] + endpoints[0:-1])/2
			# loop through batches and concatenate
			yvals = []
			for dists in ydists:
				yvals.append(dists.ppf(qs.reshape(-1, 1)).T)
			yvals = np.concatenate(yvals, axis=0)
			n = len(yvals)
			# concatenate y probs
			yprobs = endpoints[1:] - endpoints[0:-1]
			yprobs = np.stack([yprobs for _ in range(n)], axis=0)

		### Insert additional support points with zero prob
		if ninterp > 0:
			to_add = np.linspace(ymin, yvals.min(axis=1), ninterp+1)[0:-1].T
			to_add = np.concatenate(
				[to_add, np.linspace(ymax, yvals.max(axis=1), ninterp+1)[0:-1].T],
				axis=1
			)
			yvals = np.concatenate([to_add, yvals], axis=1)
			yprobs = np.concatenate([np.zeros(to_add.shape), yprobs], axis=1)
	
		# Minimum yprob for numerical stability; then return
		yprobs = np.maximum(yprobs, min_yprob)
		yprobs /= yprobs.sum(axis=1).reshape(-1, 1)
		return yvals, yprobs

	def _ensure_feasibility(
		self,
		i,
		nu0,
		nu1,
		lower,
		y0_min,
		y0_max,
		y1_min,
		y1_max,
		grid_size=100,
		tol=5e-4,
	):
		"""
		ensures nu0 + nu1 <= fvals (if lower)
		or nu0 + nu1 >= fvals (if upper)
		by performing a gridsearch.

		Parameters
		----------
		i : int
			Index of which data-point we are performing
			this operation for.
		nu0 : np.array
			nvals0-length array of dual variables 
			associated with Y(0)
		nu1 : np.array
			nvals1-length array of dual variables
			associated with Y(1)
		lower : bool
			Specifies lower vs. upper bound.
		ymin : float
			Minimum value of y
		ymax : float
			Maximum value of y
		grid_size : int
			Grid size along each dimension (y(0) and y(1)).

		Returns
		-------
		new_y0_vals : np.array
			``nvals0 + grid_size`` length array of new y0_vals.
			*Exact size may change to avoid duplicate values
		new_nu0 : np.array
			``nvals0 + grid_size`` length array of interpolated
			and feasible dual vars. for Y(0).
		new_y1_vals : np.array
			``nvals1 + grid_size`` length array of new y1_vals
		new_nu1 : np.array
			``nvals1 + grid_size`` length array of interpolated
			and feasible dual vars. for Y(1).
			*Exact size may change to avoid duplicate values.
		dx : float
			Maximum numerical error induced by interpolation
			process.
		objval_diff : np.array
			The estimated objective value change for the new
			dual variables.
		"""
		obj_orig = nu0 @ self.y0_probs[i] + nu1 @ self.y1_probs[i]
		if grid_size > 0:
			# interpolate to compute new dual variables
			new_y0_vals = np.unique(np.sort(np.concatenate([
				np.linspace(y0_min, y0_max, grid_size), self.y0_vals[i]
			])))
			new_y1_vals = np.unique(np.sort(np.concatenate([
				np.linspace(y1_min, y1_max, grid_size), self.y1_vals[i]
			])))
			new_nu0 = self.interp_fn(
				x=self.y0_vals[i], y=nu0, newx=new_y0_vals,
			)
			new_nu1 = self.interp_fn(
				x=self.y1_vals[i], y=nu1, newx=new_y1_vals,
			)
			# compute required bounds
			fvals = self._apply_ot_fn(
				fn=self.f,
				y0=new_y0_vals,
				y1=new_y1_vals,
				x=self.X[i],
			)
			# support restrictions relax constraints
			if self.support_restriction is not None:
				not_in_support = ~(self._apply_ot_fn(
					fn=self.support_restriction,
					y0=new_y0_vals,
					y1=new_y1_vals,
					x=self.X[i],
				).astype(bool))
				# Relax constraints
				#almost_inf = 1e5 * (np.abs(fvals) + 1).max()
				if lower:
					fvals[not_in_support] = np.inf
				else:
					fvals[not_in_support] = - np.inf


			## Adjust elementwise (as opposed to a constant subtraction)
			# Calculate how far we are from feasibility
			deltas = new_nu0.reshape(-1, 1) + new_nu1.reshape(1, -1)
			deltas = deltas - fvals
			if lower:
				deltas0 = deltas.max(axis=1)
				dx0 = np.mean(np.maximum(deltas0, 0))
				deltas1 = deltas.max(axis=0)
				dx1 = np.mean(np.maximum(deltas1, 0))
				adj_axis = 1 if dx1 <= dx0 else 0
			else:
				deltas0 = deltas.min(axis=1)
				dx0 = np.mean(np.minimum(deltas0, 0))
				deltas1 = deltas.min(axis=0)
				dx1 = np.mean(np.minimum(deltas1, 0))
				adj_axis = 1 if dx1 >= dx0 else 0

			# Adjust and recompute interp_nu0/interp_nu1
			if adj_axis == 0:
				new_nu0 -= deltas0
				adj_nu0 = self.interp_fn(
					x=new_y0_vals, y=new_nu0, newx=self.y0_vals[i],
				)
				adj_nu1 = nu1
			else:
				new_nu1 -= deltas1
				adj_nu1 = self.interp_fn(
					x=new_y1_vals, y=new_nu1, newx=self.y1_vals[i],
				)
				adj_nu0 = nu0

			## Track change in objective
			new_objval = adj_nu0 @ self.y0_probs[i] + adj_nu1 @ self.y1_probs[i]
			objval_diff = obj_orig - new_objval

		else:
			new_y0_vals = self.y0_vals[i]
			new_nu0 = nu0
			new_y1_vals = self.y1_vals[i]
			new_nu1 = nu1
			objval_diff = 0

		# return
		return new_y0_vals, new_nu0, new_y1_vals, new_nu1, objval_diff

	def _ensure_primal_feasibility(
		self,
		y0_probs: np.array,
		y1_probs: np.array,
		y0_vals: np.array,
		y1_vals: np.array,
		not_in_support: np.array,
		i: int,
		fuzz=0.0001,
		solver='SCIPY',
		min_prob=0, ## new
	):
		"""
		fuzz : float
			Adds fuzz * np.random.uniform() to nonzero values to push 
			jointprobs into the interior of the primal feasible set.
		"""
		nvals0 = len(y0_vals)
		nvals1 = len(y1_vals)
		min_ratio = cp.Parameter()
		# Create variable
		jointprobs = cp.Variable((nvals0, nvals1), pos=True)
		y0p = jointprobs.sum(axis=1)
		y1p = jointprobs.sum(axis=0)
		constraints = [
			cp.sum(jointprobs)==1,
			jointprobs[not_in_support] == 0,
			y0p >= y0_probs * min_ratio,
			y1p >= y1_probs * min_ratio,
		]
		diffs0 = y0_vals[1:] - y0_vals[:-1]
		diffs1 = y1_vals[1:] - y1_vals[:-1]
		obj = cp.abs(cp.cumsum(y0p) - np.cumsum(y0_probs))[:-1] @ diffs0
		obj += cp.abs(cp.cumsum(y1p) - np.cumsum(y1_probs))[:-1] @ diffs1
		problem = cp.Problem(cp.Minimize(obj), constraints)

		for mr in [1/2, 1/5, 1/100, 0]:
			min_ratio.value = mr
			problem.solve(solver=solver)
			if problem.status in ['optimal', 'optimal_inaccurate']:
				break

		if problem.status not in ['optimal', 'optimal_inaccurate']:
			raise ValueError(f"For i={i}, failed to enforce primal feasibility.")
		# Clip
		jp = jointprobs.value
		jp += fuzz * np.random.uniform(0, 1, size=jp.shape) / (nvals0 * nvals1)
		jp[not_in_support] = 0
		jp = jp / jp.sum()
		return jp.sum(axis=1), jp.sum(axis=0)

	def _solve_single_instance(
		self, 
		i: int,
		probs0: np.array,
		probs1: np.array,
		y0_vals: np.array,
		y1_vals: np.array,
		fvals: np.array,
		not_binding: np.array,
		lower: bool,
		dual_strategy='ot',
		lp_solver='SCIPY',
		qp_solver='CLARABEL',
		se_solver='CLARABEL',
		enforce_primal_feas=True,
		debug=False,
		**kwargs,
	):
		"""
		Parameters
		----------
		not_binding : boolean array
			not_binding[i,j] = 1 iff the (i,j)th
			dual constraint should be ignored.
		debug : bool
			If True, becomes highly verbose.
		dual_strategy : str
			One of 'ot', 'lp', 'qp', 'se'
		"""
		if not lower:
			nu0, nu1, objval = self._solve_single_instance(
				probs0=probs0, 
				probs1=probs1,
				y0_vals=y0_vals,
				y1_vals=y1_vals,
				fvals=-1*fvals,
				i=i,
				not_binding=not_binding,
				lower=1,
				dual_strategy=dual_strategy,
				lp_solver=lp_solver,
				qp_solver=qp_solver,
				se_solver=se_solver,
				debug=debug,
				enforce_primal_feas=enforce_primal_feas,
				**kwargs,
			)
			return -nu0, -nu1, -objval

		### Parse dual solver
		dual_strategy = str(dual_strategy).lower()
		if dual_strategy not in ['ot', 'lp', 'qp', 'se']:
			raise ValueError(
				f"dual_strategy={dual_strategy} must be one of 'ot', 'lp', 'qp', 'se'"
			)

		### Structure
		# 1. initial OT solve  [used by ot]
		# 2. Ensure_primal_feas [used by all if any constraints are not binding]
		# 3. OT solve [used by ot]
		# 4. LP solve [used by lp]
		# 5. QP solve [used by qp]

		### 1. Initial OT solve
		if dual_strategy == 'ot':
			# Use a hack for non-binding constraints.
			# This only influences the OT solver.
			almost_inf = 1e5 * np.abs(fvals).max()
			if np.any(not_binding):
				fvals = fvals.copy()
				fvals[not_binding] = almost_inf

			# Attempt to solve with pyot
			objval, log = ot.lp.emd2(
				a=probs0,
				b=probs1,
				M=fvals,
				log=True,
				**kwargs,
			)
			nu0, nu1 = log['u'], log['v']
			too_large = _dualvals_are_too_large(nu0, nu1, almost_inf)
			if (not too_large) or (not np.any(not_binding)):
				nu1 += nu0.mean()
				nu0 -= nu0.mean()
				return nu0, nu1, objval

		### 2. Ensure primal feasibility
		if np.any(not_binding) and enforce_primal_feas:
			if debug:
				print(f"Ensuring primal feasibility. enforce_primal_feas={enforce_primal_feas}")
			probs0, probs1 = self._ensure_primal_feasibility(
				y0_probs=probs0,
				y1_probs=probs1, 
				y0_vals=y0_vals, 
				y1_vals=y1_vals, 
				not_in_support=not_binding, 
				i=i,
			)

		## 3. Second OT solve
		if dual_strategy == 'ot':
			if debug:
				print("Trying secondary ot.lp optimization.")
			objval, log = ot.lp.emd2(
				a=probs0,
				b=probs1,
				M=fvals,
				log=True,
				**kwargs,
			)
			nu0, nu1 = log['u'], log['v'] 
			nu1 += nu0.mean()
			nu0 -= nu0.mean()
			if _dualvals_are_too_large(nu0, nu1, almost_inf):
				dual_strategy = 'se'
			
		## 4/5: solve with LP/QP.
		if dual_strategy in ['lp', 'qp', 'se']:

			###NOTE: this is 25% faster but less general bc we can't swap in
			# new solvers.
			# c = np.concatenate([y0_probs, y1_probs])
			# i0s, i1s = np.where(in_support)
			# col_inds = np.concatenate([i0s, i1s + nvals0])
			# row_inds = np.concatenate([np.arange(len(i0s)), np.arange(len(i0s))])
			# b = fvals[(i0s, i1s)]
			# A_sparse = scipy.sparse.csr_matrix(
			#     (np.ones(len(row_inds)), (row_inds, col_inds)), shape=(len(i0s), len(c))
			# )
			# out_upper = scipy.optimize.linprog(
			#     c=-c, A_ub=A_sparse, b_ub=-b, bounds=(None, None),
			# )

			nu0 = cp.Variable(len(probs0))
			nu1 = cp.Variable(len(probs1))
			nusum = cp.reshape(nu0, (-1, 1)) + cp.reshape(nu1, (1, -1))
			linobj = nu0 @ probs0 + nu1 @ probs1
			binding = (~not_binding).astype(int)
			constraints = [cp.multiply(binding, nusum) <= cp.multiply(binding, fvals)]
			if dual_strategy == 'lp':
				if debug:
					print(f"Solving linear problem with {lp_solver} (not PyOT).")
				problem = cp.Problem(cp.Maximize(linobj), constraints=constraints)
				problem.solve(solver=lp_solver)
				if debug:
					print(f"LP problem status is {problem.status}.")
			elif dual_strategy == 'qp':
				if debug:
					print(f"Solving quadratic program with {qp_solver} (not PyOT).")
				pi = self.pis[i]
				pcat = np.maximum(
					np.concatenate([probs0 / (1-pi), probs1 / pi]), 
					1e-6
				)
				quad_term = (pcat @ cp.hstack([nu0, nu1]))**2 / self.n
				problem = cp.Problem(
					cp.Maximize(linobj - quad_term), constraints=constraints
				)
				problem.solve(solver=qp_solver)

			# maximize mean - 2 standard errors
			else:
				if debug:
					print(f"Solving se-adjusted problem with {se_solver} (not PyOT).")
				pcat = np.concatenate([probs0, probs1])
				pi = self.pis[i]
				Q = np.diag(np.concatenate([probs0 / (1-pi), probs1 / pi])) - np.outer(pcat, pcat)
				evals, evecs = np.linalg.eigh(Q)
				sqrtQ = evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.T
				stdev = cp.norm2(sqrtQ @ cp.hstack([nu0, nu1]))
				problem = cp.Problem(
					cp.Maximize(linobj - 2 * stdev / np.sqrt(self.n)), constraints=constraints
				)
				problem.solve(solver=se_solver)

			# Warn the user of failures
			if problem.status not in ['optimal', 'optimal_inaccurate']:
				msg = "The final bounds may be vacuous."
				if problem.status == 'unbounded':
					msg = f"Dual solver failed due to primal infeasibility for i={i}, lower={lower}. " + msg
				else:
					msg = f"Dual solver failed for i={i}, lower={lower} with status {problem.status}. " + msg
				warnings.warn(msg)
				nu0 = np.zeros(len(probs0))
				nu1 = np.zeros(len(probs1))
				objval = 0
			else:
				nu0 = nu0.value
				nu1 = nu1.value
				objval = linobj.value

		# center
		if dual_strategy not in ['qp', 'se']:
			nu1 += nu0.mean()
			nu0 -= nu0.mean()
		return nu0, nu1, objval

	def compute_dual_variables(
		self,
		y0_dists: Optional[list]=None,
		y0_vals: Optional[np.array]=None,
		y0_probs: Optional[np.array]=None,
		y1_dists: Optional[list]=None,
		y1_vals: Optional[np.array]=None,
		y1_probs: Optional[np.array]=None,
		verbose: bool=True,
		min_quantile: Optional[float]=None,
		ninterp: Optional[int]=None,
		dual_strategy: str='ot',
		lp_solver: str='SCIPY',
		qp_solver: str='CLARABEL',
		se_solver: str='CLARABEL',
		nvals0: int=100,
		nvals1: int=100,
		interp_fn: callable=interpolation.adaptive_interpolate,
		y0_min: Optional[float]=None,
		y0_max: Optional[float]=None,
		y1_min: Optional[float]=None,
		y1_max: Optional[float]=None,
		**kwargs,
	):
		"""
		Estimates dual variables using the outcome model.

		We generally recommend that the user call .fit() instead of 
		calling this function directly.

		Parameters
		----------
		y0_dists : list
			The ith distribution of y0_dists represents the conditional
			law of :math:`Y_i(0) | X_i`. There are two input formats:

			- batched scipy distribution of shape (n,)
			- list of scipy dists whose shapes add up to n.

		y0_vals : list
			Alternatively, specify a (n, nvals0)-length array
			where ``y0_vals[i]`` is the support of :math:`Y_i(0)`.
			Ignored if ``y0_dists`` is provided.
		y0_probs : np.array
			A (n, nvals0)-length array where ``y0_probs[i, j]``
			is the estimated probability that :math:`Y_i(0)`
			equals ``y0_vals[i, j].``
		y1_dists : list
			The ith distribution of y1_dists represents the conditional
			law of :math:`Y_i(1) | X_i`. There are two input formats:

			- batched scipy distribution of shape (n,)
			- list of scipy dists whose shapes add up to n.

		y1_vals : np.array
			(n, nvals1)-length array where ``y1_vals[i]`` is the support 
			of :math:`Y_i(1)`.	Ignored if ``y1_dists`` is provided.
		y1_probs : np.array
			(n, nvals1)-length array where ``y1_probs[i, j]``
			is the estimated probability that :math:`Y_i(1)`
			equals ``y1_vals[i, j].``
		dual_strategy : str
			Specifies the strategy used to find the dual variables.
			One of the following:

			- 'ot': solves a standard-form optimal transport problem.
			- 'lp': solves a full linear program.
			- 'qp': solves a quadratic program which approximately accounts for standard errors.
			- 'se': solves a convex program which exactly accounts for standard errors.

			'ot' is the default and the fastest, but 'se' can reduce 
			standard errors in noisy problems.

		lp_solver : str
			When ``dual_strategy='lp'``, specifies which cvxpy solver 
			to use to solve the optimal transport LP. Default: SCIPY.
		qp_solver : str
			When ``dual_strategy='qp'``, specifies which cvxpy solver 
			to use to solve the optimal transport QP. Default: CLARABEL.
		se_solver : str
			When ``dual_strategy='se'``, specifies which cvxpy solver 
			to use to solve the convex program. Default: CLARABEL.
		min_quantile : float
			Minimum quantile to consider when discretizing. 
			Defaults to 1 / (2*nvals).		
		nvals0 : int
			How many values to use to discretize Y(0). 
			Defaults to 100. Ignored for discrete Y.
		nvals1 : int
			How many values to use to discretize Y(1).
			Defaults to 100. Ignored for discrete Y.
		interp_fn : function 
			An interpolation function with the same input/output
			signature as ``interpolation.adaptive_interpolate``,
			which is the default. Ignored for discrete Y.
		y0_min : float
			Minimum support for Y(0).
			Defaults to ``self.y.min() - 0.5 * (self.y.max() - self.y.min())``
		y1_min : float
			Minimum support for Y(1). 
			Defaults to ``self.y.min() - 0.5 * (self.y.max() - self.y.min())``
		y0_max : float
			Maximum support for Y(0). 
			Defaults to ``self.y.max() + 0.5 * (self.y.max() - self.y.min())``
		y1_max : float
			Maximum support for Y(1). 
			Defaults to ``self.y.max() + 0.5 * (self.y.max() - self.y.min())``
		kwargs : dict
			kwargs for ``_ensure_feasibility`` method, e.g., ``grid_size``.

		Returns
		-------
		None
		"""
		### Key quantities for optimizer
		# to ensure numerical stability, we add extra quantiles
		if verbose:
			print("Estimating optimal dual variables.")
		if min([nvals0, nvals1]) <= MIN_NVALS:
			raise ValueError(f"nvals0={nvals0}, nvals1={nvals1} must be larger than {MIN_NVALS}")
		
		# If discrete=True and ydists are BatchedCategorical distributions,
		# these values are ignored.
		self.nvals0 = nvals0
		self.nvals1 = nvals1
		self.interp_fn = interp_fn

		# max/min yvals
		y0_min, y1_min, y0_max, y1_max = _default_ylims(
			self.y, y0_min, y1_min, y0_max=y0_max, y1_max=y1_max
		)

		# this discretizes if Y is continuous
		if y0_vals is None or y0_probs is None:
			y0_vals, y0_probs = self._discretize(
				y0_dists, nvals=self.nvals0, min_quantile=min_quantile,
				ymin=y0_min, ymax=y0_max, ninterp=ninterp,
			)
			if self.discrete:
				self.nvals0 = y0_vals.shape[1]
		if y1_vals is None or y1_probs is None:
			y1_vals, y1_probs = self._discretize(
				y1_dists, nvals=self.nvals1, min_quantile=min_quantile,
				ymin=y1_min, ymax=y1_max, ninterp=ninterp,
			)
			if self.discrete:
				self.nvals1 = y1_vals.shape[1]

		# ensure y1_vals, y1_probs are sorted
		self.y0_vals, self.y0_probs = utilities._sort_disc_dist(y0_vals, probs=y0_probs)
		self.y1_vals, self.y1_probs = utilities._sort_disc_dist(y1_vals, probs=y1_probs)
		# Delete old values
		del y0_vals, y1_vals, y0_probs, y1_probs

		# Initialize results
		self.nu0s = np.zeros((2, self.n, self.nvals0)) # first dimension = [lower, upper]
		self.nu1s = np.zeros((2, self.n, self.nvals1))
		# Realized dual variables
		self.hatnu0s = np.zeros((2, self.n))
		self.hatnu1s = np.zeros((2, self.n))
		# estimated cond means of nu0s, nu1s
		self.c0s = np.zeros((2, self.n))
		self.c1s = np.zeros((2, self.n))
		# objective values
		self.objvals = np.zeros((2, self.n))
		self.dxs = np.zeros((2, self.n)) # ignored, only for backward compatability
		self.objdiffs = np.zeros((2, self.n))
		# loop through
		for i in utilities.vrange(self.n, verbose=verbose):
			# set parameter values
			fvals = self._apply_ot_fn(
				fn=self.f,
				y0=self.y0_vals[i],
				y1=self.y1_vals[i],
				x=self.X[i],
			)
			if self.support_restriction is not None:
				not_binding = ~(self._apply_ot_fn(
					fn=self.support_restriction,
					y0=self.y0_vals[i],
					y1=self.y1_vals[i],
					x=self.X[i],
				).astype(bool))
			else:
				not_binding = np.zeros(fvals.shape).astype(bool)
			# solve
			for lower in [1, 0]:
				nu0x, nu1x, objval = self._solve_single_instance(
					probs0=self.y0_probs[i],
					probs1=self.y1_probs[i],
					y0_vals=self.y0_vals[i],
					y1_vals=self.y1_vals[i],
					fvals=fvals,
					lower=lower,
					i=i,
					not_binding=not_binding,
					dual_strategy=dual_strategy,
					lp_solver=lp_solver,
					qp_solver=qp_solver,
					se_solver=se_solver,
				)
				self.objvals[1 - lower, i] = objval
				# Save solutions
				self.nu0s[1 - lower, i] = nu0x
				self.nu1s[1 - lower, i] = nu1x
				self.c0s[1 - lower, i] = nu0x @ self.y0_probs[i]
				self.c1s[1 - lower, i] = nu1x @ self.y1_probs[i]
				# Compute realized dual variables
				self._interpolate_and_ensure_feas(
					yi=self.y[i], 
					i=i,
					lower=lower, 
					y0_min=y0_min, 
					y0_max=y0_max,
					y1_min=y1_min,
					y1_max=y1_max,
					**kwargs
				)

	def _interpolate_and_ensure_feas(
		self, yi, lower, i, y0_min, y1_min, y0_max, y1_max, **kwargs
	):  
		"""
		This is used in two places:
		1. _compute_realized_dual_variables
		2. Main loop of compute_dual_variables

		We use this in the main loop of compute_dual_variables
		instead of just calling _compute_realized_dual_variables
		so that the progress report is accurate.
		"""
		nu0x = self.nu0s[1-lower, i]
		nu1x = self.nu1s[1-lower, i]
		# Ensure feasibility
		if not self.discrete:
			y0v, nu0adj, y1v, nu1adj, odx = self._ensure_feasibility(
				i=i, nu0=nu0x, nu1=nu1x, lower=lower, 
				y0_min=y0_min, y0_max=y0_max,
				y1_min=y1_min, y1_max=y1_max, 
				**kwargs
			)
			self.objdiffs[1-lower, i] = odx
			# interpolate to find realized values 
			self.hatnu0s[1 - lower, i] = self.interp_fn(
				x=y0v, y=nu0adj, newx=yi,
			)[0]
			self.hatnu1s[1 - lower, i] = self.interp_fn(
				x=y1v, y=nu1adj, newx=yi,
			)[0]
		else:
			j0 = np.argmin(np.abs(self.y0_vals[i] - yi))#.item()
			j1 = np.argmin(np.abs(self.y1_vals[i] - yi))#.item()
			self.hatnu0s[1 - lower, i] = nu0x[j0]
			self.hatnu1s[1 - lower, i] = nu1x[j1]   



	def _compute_realized_dual_variables(self, y=None, **kwargs):
		"""
		Helper function which applies interpolation 
		(for continuous Y) to compute realized dual variables.
		It also ensures feasibility through a 2D gridsearch.
		This is used primarily in unit tests.
		"""
		y = self.y if y is None else y

		# Create ylims
		y0_min, y1_min, y0_max, y1_max = _default_ylims(y=y, **kwargs)
		kwargs['y0_min'] = y0_min
		kwargs['y1_min'] = y1_min
		kwargs['y0_max'] = y0_max
		kwargs['y1_max'] = y1_max

		### Compute realized hatnu1s/hatnu0s
		for i in range(self.n):
			for lower in [0, 1]:
				self._interpolate_and_ensure_feas(
					i=i,
					yi=y[i],
					lower=lower, 
					**kwargs
				)

	def _compute_ipw_summands(self):
		"""
		Helper method to compute (A)IPW estimator summands.
		Must be run after running ``self.compute_dual_variables``
		"""
		# initialize outputs
		self.ipw_summands = np.zeros((2, self.n))
		self.aipw_summands = np.zeros((2, self.n))

		# compute IPW summands and AIPW summands
		for lower in [0, 1]:
			# IPW
			self.ipw_summands[1-lower] = self.hatnu1s[1-lower] / self.pis
			self.ipw_summands[1-lower][self.W == 0] = (
				self.hatnu0s[1-lower] / (1 - self.pis)
			)[self.W == 0]
			# AIPW
			mus = self.c1s[1-lower] + self.c0s[1-lower]
			aipw1 = (self.hatnu1s[1-lower] - self.c1s[1-lower]) / self.pis + mus 
			aipw0 = (self.hatnu0s[1-lower] - self.c0s[1-lower]) / (1-self.pis) + mus
			self.aipw_summands[1-lower] = aipw1 * self.W + (1 - self.W) * aipw0

	def _compute_final_bounds(self, aipw=True, alpha=0.05):
		"""
		Computes final results from the estimated dual variables.
		"""
		self._compute_ipw_summands()
		self.estimates, self.ses, self.cis = utilities.compute_est_bounds(
			summands=self.aipw_summands if aipw else self.ipw_summands,
			alpha=alpha,
			clusters=self.clusters,
		)
		return dict(
			estimates=self.estimates,
			ses=self.ses,
			cis=self.cis,
		)

	def _compute_cond_means(self):
		"""
		Helper function which computes self.mu0 = E[Y(0) | X],
		self.mu1 = E[Y(1) | X]. Can only be run after the 
		conditional distributions have been estimated.
		"""
		# make lists
		if isinstance(self.y0_dists, list):
			y0_dists = self.y0_dists
		else:
			y0_dists = [self.y0_dists]
		if isinstance(self.y1_dists, list):
			y1_dists = self.y1_dists
		else:
			y1_dists = [self.y1_dists]
		# Compute
		self.mu0 = np.concatenate([x.mean() for x in y0_dists])
		self.mu1 = np.concatenate([x.mean() for x in y1_dists])

	def fit_propensity_scores(
		self, nfolds: int, clip: float=1e-2, verbose: bool=True,
	):
		"""
		Cross-fits the propensity scores.

		Parameters
		----------
		nfolds : int
			Number of folds.
		clip : float
			Clip propensity scores to be in [clip, 1-clip].
		verbose : bool
			If True, prints progress reports.

		Returns
		-------
		None
		"""
		# Parse model
		if self.propensity_model is None:
			self.propensity_model = 'ridge'
		if isinstance(self.propensity_model, str):
			model_cls = dist_reg.parse_model_type(
				self.propensity_model, discrete=True
			)
			self.propensity_model = model_cls()
		
		# Loop through
		if verbose:
			print("Fitting propensity scores.")
		starts, ends = dist_reg.create_folds(n=self.n, nfolds=nfolds)
		# loop through and fit
		self.propensity_model_fits = []
		self.pis = np.zeros(self.n)
		for ii in utilities.vrange(len(starts), verbose=verbose):
			start, end = starts[ii], ends[ii]
			# Pick out data from the other folds
			not_in_fold = [
				i for i in np.arange(self.n) if i < start or i >= end
			]
			# Fit 
			model_fit = copy.deepcopy(self.propensity_model)
			model_fit.fit(
				X=self.X[not_in_fold], y=self.W[not_in_fold]
			)
			self.propensity_model_fits.append(model_fit)
			# Predict out of sample
			self.pis[start:end] = model_fit.predict_proba(
				self.X[start:end]
			)[:, 1]

		# Clip
		self.pis = np.minimum(np.maximum(self.pis, clip), 1-clip)

	def cross_fit(
		self,
		nfolds: int=5,
		suppress_warning: bool=False,
		verbose: bool=True,
		weight_by_propensities: bool=False,
	):
		"""
		Cross-fits the outcome model.

		Parameters
		----------
		nfolds : int
			Number of folds to use in cross-fitting.
		suppress_warning : bool
			If True, suppresses a potential warning about cross-fitting.
		verbose : bool
			If True, prints progress reports.
		weight_by_propensities : bool
			If True, when cross-fitting the outcome model, upweights
			observations with low propensity scores.

		Returns
		-------
		y0_dists : list
			list of batched scipy distributions whose shapes sum to n.
			the ith distribution is the out-of-sample estimate of
			the conditional law of :math:`Y_i(0) | X[i]`
		y1_dists : list
			list of batched scipy distributions whose shapes sum to n.
			the ith distribution is the out-of-sample estimate of
			the conditional law of :math:`Y_i(1) | X[i]`
		"""

		# if pis not supplied: will use cross-fitting
		if self.pis is None:
			self.fit_propensity_scores(nfolds=nfolds, verbose=verbose)

		# Fit model
		if self.y0_dists is None or self.y1_dists is None:
			# Note: this returns the existing model
			# if an existing model is provided
			self.outcome_model = get_default_model(
				discrete=self.discrete, 
				support=self.support,
				outcome_model=self.outcome_model,
				**self.model_kwargs
			)
			if verbose:
				print("Cross-fitting the outcome model.")
			# Possibly create sample weights
			if weight_by_propensities:
				self.sample_weight = 1 / self.pis.copy()
				self.sample_weight[self.W == 0] = 1 / (1 - self.pis[self.W == 0])
			else:
				self.sample_weight = None

			y_out = dist_reg.cross_fit_predictions(
				W=self.W, X=self.X, y=self.y, 
				sample_weight=self.sample_weight, 
				nfolds=nfolds, 
				model=self.outcome_model,
				model_selector=self.model_selector,
				verbose=verbose,
			)
			counterfactuals, self.model_fits, self.oos_dist_preds = y_out
			self.y0_dists = counterfactuals[0]
			self.y1_dists = counterfactuals[1]
		elif not suppress_warning:
			warnings.warn(CROSSFIT_WARNING)

		return self.y0_dists, self.y1_dists

	def _compute_oos_resids(self):
		# compute out-of-sample predictions
		self._compute_cond_means()
		self.oos_preds = self.mu0.copy()
		self.oos_preds[self.W == 1] = self.mu1[self.W == 1]
		# residuals and return
		self.oos_resids = self.y - self.oos_preds
		return self.oos_resids

	def fit(
		self,
		nfolds: int = 5,
		aipw: bool = True,
		alpha: float = 0.05,
		y0_dists: Optional[list[rv_generic]] = None,
		y1_dists: Optional[list[rv_generic]] = None,
		verbose: bool = True,
		suppress_warning: bool = False,
		weight_by_propensities: bool = False,
		**solve_kwargs,
	):
		"""
		Main function which (1) performs cross-fitting, (2) computes 
		optimal dual variables, and (3) computes final dual bounds.

		Parameters
		----------  
		nfolds : int
			Number of folds to use when cross-fitting. Defaults to 5.
		alpha : float
			Nominal coverage level. Defaults to 0.05.
		aipw : bool
			If true, returns AIPW estimator.
		y0_dists : list
			The ith distribution of y0_dists represents the conditional
			law of :math:`Y_i(0) | X_i`. There are two input formats:

			- batched scipy distribution of shape (n,)
			- list of scipy dists whose shapes add up to n.

			This is an optional input; if provided, ``outcome_model``
			will be ignored.
		y1_dists : list
			The ith distribution of y1_dists represents the conditional
			law of :math:`Y_i(1) | X_i`. There are two input formats:

			- batched scipy distribution of shape (n,)
			- list of scipy dists whose shapes add up to n.

			This is an optional input; if provided, ``outcome_model``
			will be ignored.
		verbose : bool
			If True, gives occasional progress reports.
		suppress_warning : bool
			If True, suppresses a warning about cross-fitting.
		weight_by_propensities : bool
			If True, when cross-fitting the outcome model, upweights
			observations with low propensity scores.
		solve_kwargs : dict
			Additional (optional) kwargs for the ``compute_dual_variables``
			method, e.g. ``nvals0``, ``nvals1``, ``grid_size``.

		Returns
		-------
		self
		"""
		# Fit model of W | X and Y | X if not provided
		self.y0_dists, self.y1_dists = y0_dists, y1_dists
		self.cross_fit(
			nfolds=nfolds, 
			suppress_warning=suppress_warning, 
			verbose=verbose,
			weight_by_propensities=weight_by_propensities,
		)

		# compute dual variables
		self.compute_dual_variables(
			y0_dists=self.y0_dists,
			y1_dists=self.y1_dists,
			verbose=verbose,
			**solve_kwargs,
		)
		# compute dual bounds
		self.alpha = alpha
		self._compute_final_bounds(aipw=aipw, alpha=alpha)
		return self

	def _plug_in_results(self):
		pests, pses, pcis = utilities.compute_est_bounds(
			summands=self.objvals,
			alpha=self.alpha
		)
		return pd.DataFrame(
			np.stack(
				[pests, pses, pcis], 
				axis=0
			),
			index=['Estimate', 'SE', 'Conf. Int.'],
			columns=['Lower', 'Upper']
		)

	def results(self, minval: float=-np.inf, maxval: float=np.inf):
		"""
		Returns a dataframe of key inferential results.

		Parameters
		----------
		minval : float
			Analytical lower bound on estimand used to clip results. 
			Defaults to -np.inf.
		maxval : float
			Analytical upper bound on estimand used to clip results.
			Defaults to np.inf.

		Returns
		-------
		results : pd.DataFrame
			DataFrame of key inferential results.
		"""
		self.results_ = pd.DataFrame(
			np.stack(
				[
					np.clip(self.estimates, minval, maxval), 
					self.ses,
					np.clip(self.cis, minval, maxval),
				], 
				axis=0
			),
			index=['Estimate', 'SE', 'Conf. Int.'],
			columns=['Lower', 'Upper']
		)
		return self.results_

	def plot_dual_variables(self, i=0):
		"""
		Plots the estimated dual variables for the ith data-point.

		Parameters
		----------
		i : int
			Integer ranging from 0 to n-1, specifying which datapoint to plot.
		"""
		import matplotlib.pyplot as plt
		fig, axes = plt.subplots(1, 2, figsize=(10, 4))
		for name, j, ax in zip(['Lower', 'Upper'], [0,1], axes):
			# Plot dual variables
			for yvals, nus, color, label in zip(
				[self.y0_vals[i], self.y1_vals[i]],
				[self.nu0s[j, i], self.nu1s[j, i]],
				['red', 'blue'],
				['Control', 'Treatment'],
			):
				axes[j].scatter(yvals, nus, color=color, label=label)
			# Plot realized value
			hatnu = self.hatnu0s[j, i] if self.W[i] == 0 else self.hatnu1s[j, i]
			axes[j].axvline(self.y[i], color='black', label='Realized outcome value')
			axes[j].scatter(self.y[i], hatnu, color='red' if self.W[i] == 0 else 'blue')
			axes[j].set(xlabel='Outcome', ylabel='Dual variable', title=f'{name} Bound')
			axes[j].legend()
		plt.show()

	def diagnostics(self, plot=False, aipw=True):
		"""
		Reports a set of technical diagnostics.

		Parameters
		----------
		plot : bool
			If True, creates a set of diagnostic plots.

		Returns
		-------
		df : pd.DataFrame
			DataFrame of technical diagnostic information.

		Notes
		-----
		Please see the user guide for more details on the
		meaning of the outputs.
		"""
		summands = self.aipw_summands if aipw else self.ipw_summands
		aipw_name = 'AIPW' if aipw else 'IPW'
		# Plot AIPW dual summands
		if plot:
			import matplotlib.pyplot as plt
			fig, axes = plt.subplots(1, 2, figsize=(10, 4))
			for j, name in zip([0,1], ['Lower', 'Upper']):
				for w, color, label in zip([0,1], ['red', 'blue'], ['Control', 'Treatment']):
					axes[j].scatter(
						self.y[self.W == w], 
						self.aipw_summands[j][self.W == w], 
						color=color, 
						label=label
					)
				axes[j].legend()
				axes[j].set(xlabel='Outcome', ylabel=f'Dual {aipw_name} Summand', title=f"{name} Dual Bound")
			plt.show()
		# Leverage
		s2s = (summands - summands.mean(axis=1).reshape(-1, 1))**2
		leverages = np.max(s2s, axis=1) / s2s.sum(axis=1)
		# Max contribution
		max_contribs = np.array(
			[summands[0].min() / self.n, summands[1].max() / self.n]
		)
		return pd.DataFrame(
			np.stack(
				[self.objdiffs.mean(axis=1), leverages, max_contribs],
				axis=0
			),
			index=['Loss from gridsearch', 'Max leverage', f'Worst dual {aipw_name} summand'],
			columns=['Lower', 'Upper']
		)
		

	def eval_outcome_model(self):
		"""
		Thinly wraps ``dist_reg._evaluate_model_predictions``.

		Returns
		-------
		sumstats : pd.DataFrame
			DataFrame summarizing goodness-of-fit metrics for
			the cross-fit outcome model.
		"""
		self._compute_oos_resids()
		return dist_reg._evaluate_model_predictions(
			y=self.y, haty=self.oos_preds
		)

	def eval_treatment_model(self):
		"""
		Thinly wraps ``dist_reg._evaluate_model_predictions``.

		Returns
		-------
		sumstats : pd.DataFrame
			DataFrame summarizing goodness-of-fit metrics for
			the cross-fit propensity scores.
		"""
		return dist_reg._evaluate_model_predictions(
			y=self.W, haty=self.pis
		)

	def summary(self, minval: float=-np.inf, maxval: float=np.inf):
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

		Returns
		-------
		None

		Notes
		-----
		To access the results a DataFrame, call:

		- ``DualBounds.results()`` for inferential results
		- ``DualBounds.eval_outcome_model()`` for outcome model metrics
		- ``DualBounds.eval_treatment_model()`` for treatment model metrics
		- ``DualBounds._plug_in_results()`` for nonrobust plug-in bounds
		- ``DualBounds.diagnostics()`` for technical diagnostic information
		"""
		print("___________________Inference_____________________")
		print(self.results(minval=minval, maxval=maxval))
		print()
		print("_________________Outcome model___________________")
		print(self.eval_outcome_model())
		print()
		print("_________________Treatment model_________________")
		print(self.eval_treatment_model())
		print()
		print("______________Nonrobust plug-in bounds___________")
		print(self._plug_in_results())
		print()
		# possibly print diagnostics. This logic is useful
		# because some classes inheriting from 
		# DualBounds (e.g. VarCATEDualBounds) don't produce diagnostics.
		diagnostics = self.diagnostics(plot=False)
		if diagnostics is not None:
			print("_______________Technical diagnostics_____________")
			print(diagnostics)
			print()

def _plug_in_no_covariates(
	f: callable,
	y: np.array,
	W: np.array,
	pis: Optional[np.array]=None,
	max_nvals: int=1000,
) -> float:
	"""
	Internal function which does the work for plug_in_no_covariates
	and gets called in each bootstrap iteration.
	"""
	n = len(y)
	# Create empirical distributions for treatment/control
	y0_vals = y[W == 0]
	y0_probs = 1 / (1-pis[W==0]); y0_probs /= y0_probs.sum()
	y1_vals = y[W == 1]
	y1_probs = 1 / (pis[W==1]); y1_probs /= y1_probs.sum()
	# Reduce dimension to prevent memory errors for huge datasets
	qs = np.linspace(1/(max_nvals+1), max_nvals/(max_nvals+1), max_nvals)
	if len(y0_vals) > max_nvals:
		y0_vals = utilities.weighted_quantile(y0_vals, y0_probs, quantiles=qs)
		y0_probs = np.ones(len(y0_vals)) / len(y0_vals)
	if len(y1_vals) > max_nvals:
		y1_vals = utilities.weighted_quantile(y1_vals, y1_probs, quantiles=qs)
		y1_probs = np.ones(len(y1_vals)) / len(y1_vals)
	# Fvals
	fvals = f(y0_vals.reshape(-1, 1), y1_vals.reshape(1, -1), x=0)
	# lower and upper
	lower = ot.lp.emd2(a=y0_probs, b=y1_probs, M=fvals)
	upper = -ot.lp.emd2(a=y0_probs, b=y1_probs, M=-fvals)
	return np.array([lower, upper])

def plug_in_no_covariates(
	outcome: np.array, 
	treatment: np.array, 
	f: callable, 
	propensities: Optional[np.array]=None,
	clusters: Optional[np.array]=None,
	B: int=0,
	verbose: bool=True,
	alpha: float=0.05,
	max_nvals: int=1000,
	_which_bound='both',
) -> dict:
	"""
	Computes plug-in bounds on :math:`E[f(Y(0),Y(1))]` without using covariates.

	Parameters
	----------
	outcome : np.array
		n-length array of outcomes (y)
	treatment : np.array
		n-length array of treatments (W).
	f : function
		f(y0, y1, x) defines the objective.
	propensities : np.array
		n-length array of propensity scores (pis).
		Default: all equal to treatment.mean().
	clusters : np.array
		Optional n-length array of clusters, so ``clusters[i] = j``
		indicates that observation i is in cluster j.
	B : int
		Number of bootstrap replications to compute standard errors.
		Defaults to 0 (no standard errors).
	verbose : bool
		Show progress bar while bootstrapping if verbose=True.
	alpha : float
		nominal Type I error level.
	max_nvals : int
		Maximum dimension of optimal transport problem.
	_which_bound : str
		One of 'both', 'lower', 'upper'.

	Returns
	-------
	results : dict
		Dictionary containing up to three keys:

		- estimates: 2-length array of lower/upper estimates.
		- ses: 2-length array of lower/upper standard errors.
		- cis: 2-length array of lower/upper confidence intervals.

		These arrays will be length 1 (instead of 2) if which_bound != 'both'.
	"""
	# Infer propensities
	if propensities is None:
		propensities = np.ones(len(treatment)) * treatment.mean()
	# Create estimates
	estimates = _plug_in_no_covariates(
		f=f,
		y=outcome,
		W=treatment,
		pis=propensities,
		max_nvals=max_nvals
	)	
	if B == 0:
		return dict(estimates=estimates)

	# Compute bootstrapped SEs
	data = np.stack([outcome, treatment, propensities], axis=1)
	func = lambda data: _plug_in_no_covariates(
		f=f,
		y=data[:, 0],
		W=data[:, 1],
		pis=data[:, 2],
		max_nvals=max_nvals,
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