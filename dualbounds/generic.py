import copy
import warnings
import numpy as np
import ot
from scipy import stats
from . import utilities, dist_reg, interpolation
from .utilities import BatchedCategorical

MIN_NVALS = 7
DISC_THRESH = 2 # treat vars. with <= DISC_THRESH values as discrete
CROSSFIT_WARNING = """
==================================================
Not fitting a model because y0_dists/y1_dists were
directly provided. Please ensure cross-fitting is
employed correctly, else inference will be invalid
(see https://arxiv.org/abs/2310.08115). To suppress
this warning, set ``supress_warning=True``.
==================================================
"""

def get_default_model(discrete, support, Y_model=None, **model_kwargs):
	if isinstance(Y_model, dist_reg.DistReg):
		return Y_model
	Y_model = 'ridge' if Y_model is None else Y_model
	if not discrete:
		return dist_reg.CtsDistReg(
			model_type=Y_model, **model_kwargs
		)
	elif discrete and set(support) == set([0, 1]):
		return dist_reg.BinaryDistReg(
			model_type=Y_model, **model_kwargs
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

class DualBounds:
	"""
	Computes dual bounds on E[f(Y(0),Y(1), X)].

	Parameters
	----------
	f : function
		Function which defines the partially identified estimand.
		Must be a function of three arguments: y0, y1, x 
		(in that order). E.g.,
		``f = lambda y0, y1, x : y0 <= y1``
	X : np.array
		(n, p)-shaped array of covariates.
	W : np.array
		n-length array of binary treatment indicators.
	Y : np.array
		n-length array of outcome measurements.
	pis : np.array
		n-length array of propensity scores P(W=1 | X). 
		If ``None``, will be estimated from the data.
	Y_model : str or dist_reg.DistReg
		One of ['ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'].
		Alternatively, a distributional regression class inheriting 
		from ``dist_reg.DistReg``. E.g., when ``y`` is continuous,
		defaults to ``Y_model=dist_reg.CtsDistReg(model_type='ridge')``.
	W_model : str or sklearn classifier
		Specifies how to estimate the propensity scores if ``pis`` is
		not known.  Either a str identifier as above or an sklearn
		classifier---see the tutorial for examples.
	discrete : bool
		If True, treats y as a discrete variable. 
		Defaults to ``None`` (inferred from the data).
	support : np.array
		Optinal support of y, if known.
		Defaults to ``None`` (inferred from the data).
	**model_kwargs : dict
		Additional kwargs for an underlying ``DistReg`` model,
		e.g., ``eps_dist`` (for cts. y) or ``feature_transform``.

	Examples
	--------
	Here we fit DualBounds on P(Y(0) < Y(1)) based on
	synthetic regression data: ::
		import dualbounds as db

		# Generate synthetic data from a heavy-tailed linear model
		data = db.gen_data.gen_regression_data(
			n=900, # Num. datapoints
			p=30, # Num. covariates
			r2=0.95, # population R^2
			tau=3, # average treatment effect
			interactions=True, # ensures treatment effect is heterogenous
			eps_dist='laplace', # heavy-tailed residuals
			sample_seed=123, # random seed
		)

		# Initialize dual bounds object
		dbnd = db.generic.DualBounds(
			f=lambda y0, y1, x: y0 < y1,
			X=data['X'], # n x p covariate matrix
			W=data['W'], # n-length treatment vector
			y=data['y'], # n-length outcome vector
			pis=data['pis'], # n-length propensity scores (optional)
			Y_model='ridge', # model for Y | X, W
		)

		# Compute dual bounds and observe output
		dbnd.compute_dual_bounds(
			alpha=0.05 # nominal level
		)
	"""
	def __init__(
		self, 
		f,
		X,
		W,
		y,
		pis=None,
		Y_model=None,
		W_model=None,
		discrete=None,
		support=None,
		**model_kwargs,
	):
		### Data
		self.f = f
		self.X = X
		self.y = y
		self.W = W
		self.pis = pis
		self.n = len(self.y)

		### Check if discrete
		self.discrete, self.support = infer_discrete(
			discrete=discrete, support=support, y=self.y,
		)
		self.Y_model = Y_model
		self.W_model = W_model
		self.model_kwargs = model_kwargs

		# Initialize
		self.y0_dists = None
		self.y1_dists = None

	def _discretize(
		self, 
		ydists,
		nvals,
		ymin,
		ymax,
		alpha=0.001,
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
					raise ValueError("self.discrete=True, but ydist is not a BatchedCategorical distribution.")
				yvals.append(ydist.vals)
				yprobs.append(ydist.probs)
			return np.concatenate(yvals, axis=0), np.concatenate(yprobs, axis=0)

		### Continuous case ###
		if nvals <= MIN_NVALS:
			raise ValueError(f"nvals must be larger than {nvals}.")
		
		# make sure we get small enough quantiles
		if alpha is None:
			alpha = 1 / (2*nvals)
		alpha = min(1/(2*nvals), max(alpha, 1e-8))

		# num of interp. pts between min/max quantiles
		# and ymin/ymax, added to ensure feasbility
		if ninterp is None:
			ninterp = min(max(int(0.1 * (nvals-2)), 1), 5)
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
		ymin : float
			Minimum value of y
		ymax : float
			Maximum value of y
		grid_size : int
			Grid size along each dimension (y(0) and y(1)).

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
		# interpolate to compute new dual variables
		new_y0vals = np.linspace(y0_min, y0_max, grid_size)
		new_y1vals = np.linspace(y1_min, y1_max, grid_size)
		interp_nu0 = self.interp_fn(
			x=self.y0_vals[i], y=nu0, newx=new_y0vals,
		)
		interp_nu1 = self.interp_fn(
			x=self.y1_vals[i], y=nu1, newx=new_y1vals,
		)
		# compute required bounds
		fx = lambda y0, y1: self.f(y0, y1, self.X[i])
		fvals = fx(
			new_y0vals.reshape(-1, 1), new_y1vals.reshape(1, -1)
		)
		## Step 1: adjust elementwise (as opposed to a constant subtraction)
		for ii in range(10):
			# Calculate how far we are from feasibility
			deltas = interp_nu0.reshape(-1, 1) + interp_nu1.reshape(1, -1)
			deltas = deltas - fvals
			if lower:
				deltas0 = np.maximum(deltas.max(axis=1), 0)
				deltas1 = np.maximum(deltas.max(axis=0), 0)
				dx = np.max(deltas0)
				if ii == 0:
					init_axis = 1 if np.mean(deltas1) <= np.mean(deltas0) else 0
			else:
				deltas0 = np.minimum(deltas.min(axis=1), 0)
				deltas1 = np.minimum(deltas.min(axis=0), 0)
				dx = np.min(deltas0)
				if ii == 0:
					init_axis = 1 if np.mean(deltas1) >= np.mean(deltas0) else 0
			# Stopping condition
			#obj = nu0 @ self.y0_probs[i] + nu1 @ self.y1_probs[i]
			#print(f"At iter={ii}, obj={obj}, dx={dx}")
			if lower and dx <= tol:
				break
			if not lower and dx >= - tol:
				break
			# Adjust and recompute interp_nu0/interp_nu1
			if ii % 2 == init_axis:
				to_sub = self.interp_fn(
					x=new_y0vals, y=deltas0, newx=self.y0_vals[i]
				)
				nu0 = nu0 - to_sub
				interp_nu0 = self.interp_fn(
					x=self.y0_vals[i], y=nu0, newx=new_y0vals,
				)
			else:
				to_sub = self.interp_fn(
					x=new_y1vals, y=deltas1, newx=self.y1_vals[i]
				)
				nu1 = nu1 - to_sub
				interp_nu1 = self.interp_fn(
					x=self.y1_vals[i], y=nu1, newx=new_y1vals,
				)

		## Step 2: subtract global constant to ensure feasibility
		deltas = interp_nu0.reshape(-1, 1) + interp_nu1.reshape(1, -1) - fvals
		if lower:
			dx = np.max(deltas)
		else:
			dx = np.min(deltas)

		## For debugging only, delete later
		self.new_y0vals = new_y0vals
		self.new_y1vals = new_y1vals
		self.interp_nu0 = interp_nu0
		self.interp_nu1 = interp_nu1
		self.orig_nu0 = nu0
		self.orig_nu1 = nu1
		self.deltas = deltas
		self.fvals_debug = fvals

		# return
		return nu0 - dx/2, nu1 - dx/2, dx

	def _solve_single_instance(
		self, 
		probs0,
		probs1,
		fvals,
		lower,
		**kwargs,
	):
		if lower:
			objval, log = ot.lp.emd2(
				a=probs0,
				b=probs1,
				M=fvals,
				log=True,
				**kwargs,
			)
			nu0, nu1 = log['u'], log['v']
			# center
			nu1 += nu0.mean()
			nu0 -= nu0.mean()
			return nu0, nu1, objval
		else:
			nu0, nu1, objval = self._solve_single_instance(
				probs0=probs0, probs1=probs1, fvals=-1*fvals,
				lower=1,
				**kwargs,
			)
			return -nu0, -nu1, -objval

	def compute_dual_variables(
		self,
		y0_dists=None,
		y0_vals=None,
		y0_probs=None,
		y1_dists=None,
		y1_vals=None,
		y1_probs=None,
		verbose=True,
		alpha=None,
		ninterp=None,
		nvals0=100,
		nvals1=100,
		interp_fn=interpolation.linear_interpolate,
		y0_min=None,
		y0_max=None,
		y1_min=None,
		y1_max=None,
		**kwargs,
	):
		"""
		Uses the estimated outcome model to solve the dual optimal
		transport problem to obtain optimal dual variables.

		Parameters
		----------
		y0_dists : np.array
			batched scipy distribution of shape (n,) where the ith
			distribution is the conditional law of Yi(0) | Xi
			OR 
			list of scipy dists whose shapes add up to n.
		y0_vals : np.array
			(n, nvals0)-length array of values y0 can take.
		y0_probs : np.array
			(n, nvals0)-length array where
			y0_probs[i, j] = P(Y(0) = yvals0[j] | Xi)
		y1_dists : np.array
			batched scipy distribution of shape (n,) where the ith
			distribution is the conditional law of Yi(1) | Xi
			OR 
			list of scipy dists whose shapes add up to n.
		y1_vals : np.array
			(n, nvals1)-length array of values y1 can take.
			Ignored if ``y1_dists`` is provided.
		y1_probs : np.array
			(n, nvals1)-length array where
			y0_probs[i, j] = P(Y(1) = yvals1[j] | Xi).
			Ignored if ``y1_dists`` is provided.
		nvals0 : int
			How many values to use to discretize Y(0). 
			Defaults to 100. Ignored for discrete Y.
		nvals1 : int
			How many values to use to discretize Y(1).
			Defaults to 100. Ignored for discrete Y.
		interp_fn : function 
			An interpolation function with the same input/output
			signature as ``interpolation.linear_interpolate``,
			which is the default. Ignored for discrete Y.
		y0_min : float
			Minimum support for Y(0). Defaults to self.y.min()
		y1_min : float
			Minimum support for Y(1). Defaults to self.y.min()
		y0_max : float
			Maximum support for Y(0). Defaults to self.y.max()
		y1_max : float
			Maximum support for Y(1). Defaults to self.y.max()
		kwargs : dict
			kwargs for ``_ensure_feasibility`` method, e.g., ``grid_size``.
		"""
		### Key quantities for optimizer
		# to ensure numerical stability, we add extra quantiles
		if min([nvals0, nvals1]) <= MIN_NVALS:
			raise ValueError(f"nvals0={nvals0}, nvals1={nvals1} must be larger than {MIN_NVALS}")
		if self.discrete:
			self.nvals0 = len(self.support)
			self.nvals1 = len(self.support)
		else:
			self.nvals0 = nvals0
			self.nvals1 = nvals1
		self.interp_fn = interp_fn

		## Parse defaults
		if y0_min is None:
			y0_min = self.y.min()
		if y1_min is None:
			y1_min = self.y.min()
		if y0_max is None:
			y0_max = self.y.max()
		if y1_max is None:
			y1_max = self.y.max()

		# this discretizes if Y is continuous
		if y0_vals is None or y0_probs is None:
			y0_vals, y0_probs = self._discretize(
				y0_dists, nvals=self.nvals0, alpha=alpha,
				ymin=y0_min,
				ymax=y0_max,
				ninterp=ninterp,
			)
		if y1_vals is None or y1_probs is None:
			y1_vals, y1_probs = self._discretize(
				y1_dists, nvals=self.nvals1, alpha=alpha,
				ymin=y1_min,
				ymax=y1_max,
				ninterp=ninterp,
			)

		# ensure y1_vals, y1_probs are sorted
		self.y0_vals, self.y0_probs = utilities._sort_disc_dist(y0_vals, probs=y0_probs)
		self.y1_vals, self.y1_probs = utilities._sort_disc_dist(y1_vals, probs=y1_probs)
		# Delete old values
		del y0_vals, y1_vals, y0_probs, y1_probs

		# Initialize results
		self.nu0s = np.zeros((2, self.n, self.nvals0)) # first dimension = [lower, upper]
		self.nu1s = np.zeros((2, self.n, self.nvals1))
		# estimated cond means of nu0s, nu1s
		self.c0s = np.zeros((2, self.n))
		self.c1s = np.zeros((2, self.n))
		# objective values
		self.objvals = np.zeros((2, self.n))
		self.dxs = np.zeros((2, self.n))
		# loop through
		for i in utilities.vrange(self.n, verbose=verbose):
			# set parameter values
			fx = lambda y0, y1: self.f(y0=y0, y1=y1, x=self.X[i])
			fvals = np.array(fx(
				y0=self.y0_vals[i].reshape(-1, 1), 
				y1=self.y1_vals[i].reshape(1, -1),
			))
			# solve
			for lower in [1, 0]:
				nu0x, nu1x, objval = self._solve_single_instance(
					probs0=self.y0_probs[i],
					probs1=self.y1_probs[i],
					fvals=fvals,
					lower=lower,
				)
				self.objvals[1 - lower, i] = objval
				if not self.discrete:
					nu0x, nu1x, dx = self._ensure_feasibility(
						i=i, nu0=nu0x, nu1=nu1x, lower=lower, 
						y0_min=y0_min, y0_max=y0_max,
						y1_min=y1_min, y1_max=y1_max,
						**kwargs,
					)
				else:
					dx = 0
				# Save solutions
				self.nu0s[1 - lower, i] = nu0x
				self.nu1s[1 - lower, i] = nu1x
				self.c0s[1 - lower, i] = nu0x @ self.y0_probs[i]
				self.c1s[1 - lower, i] = nu1x @ self.y1_probs[i]
				self.dxs[1 - lower, i] = dx

		# Compute realized dual variables
		self._compute_realized_dual_variables(y=self.y)

	def _compute_realized_dual_variables(self, y=None):
		"""
		Helper function which applies interpolation 
		(for continuous Y) to compute realized dual variables.
		"""
		y = self.y if y is None else y
		### Compute realized hatnu1s/hatnu0s
		self.hatnu0s = np.zeros((2, self.n))
		self.hatnu1s = np.zeros((2, self.n))
		for i in range(self.n):
			for lower in [0, 1]:
				nu0x = self.nu0s[1-lower, i]
				nu1x = self.nu1s[1-lower, i]
				if not self.discrete:
					# interpolate to find realized values for cts case
					self.hatnu0s[1 - lower, i] = self.interp_fn(
						x=self.y0_vals[i], y=nu0x, newx=y[i],
					)[0]
					self.hatnu1s[1 - lower, i] = self.interp_fn(
						x=self.y1_vals[i], y=nu1x, newx=y[i],
					)[0]
				else:
					j0 = np.where(self.y0_vals[i] == y[i])[0][0]
					j1 = np.where(self.y1_vals[i] == y[i])[0][0]
					self.hatnu0s[1 - lower, i] = nu0x[j0]
					self.hatnu1s[1 - lower, i] = nu1x[j1]

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

	def compute_final_bounds(self, aipw=True, alpha=0.05):
		"""
		Computes final estimators/ses based on the estimated
		dual variables.
		"""
		self._compute_ipw_summands()
		self.estimates, self.ses, self.cis = utilities.compute_est_bounds(
			summands=self.aipw_summands if aipw else self.ipw_summands,
			alpha=alpha
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
		if self.y0_dists is None:
			self.y0_dists = BatchedCategorical(
				vals=self.y0_vals, probs=self.y0_probs
			)
		if self.y1_dists is None:
			self.y1_dists = BatchedCategorical(
				vals=self.y1_vals, probs=self.y1_probs
			)
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
		self, nfolds, clip=1e-2,
	):
		"""
		Performs cross-fitting to fit the propensity scores.
		"""
		# Parse model
		if isinstance(self.W_model, str):
			model_cls = dist_reg.parse_model_type(
				self.W_model, discrete=True
			)
			self.W_model = model_cls()
		
		# Loop through
		starts, ends = dist_reg.create_folds(n=self.n, nfolds=nfolds)
		# loop through and fit
		self.W_model_fits = []
		self.pis = np.zeros(self.n)
		for start, end in zip(starts, ends):
			# Pick out data from the other folds
			not_in_fold = [
				i for i in np.arange(self.n) if i < start or i >= end
			]
			# Fit 
			model_fit = copy.deepcopy(self.W_model)
			model_fit.fit(
				X=self.X[not_in_fold], y=self.W[not_in_fold]
			)
			self.W_model_fits.append(model_fit)
			# Predict out of sample
			self.pis[start:end] = model_fit.predict_proba(
				self.X[start:end]
			)[:, 1]

		# Clip
		self.pis = np.minimum(np.maximum(self.pis, clip), 1-clip)

	def cross_fit(
		self,
		nfolds=5,
		suppress_warning=False,
	):
		"""
		Performs cross-fitting to fit the outcome model.


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

		# if pis not supplied: will use cross-fitting
		if self.pis is None:
			self.fit_propensity_scores(nfolds=nfolds)

		# Fit model
		if self.y0_dists is None or self.y1_dists is None:
			self.Y_model = get_default_model(
				discrete=self.discrete, support=self.support, Y_model=self.Y_model,
				**self.model_kwargs
			)
			self.y0_dists, self.y1_dists, self.model_fits = dist_reg.cross_fit_predictions(
				W=self.W, X=self.X, y=self.y, 
				nfolds=nfolds, model=self.Y_model,
			)
		elif not suppress_warning:
			warnings.warn(CROSSFIT_WARNING)

		return self.y0_dists, self.y1_dists



	def compute_dual_bounds(
		self,
		nfolds=5,
		aipw=True,
		alpha=0.05,
		y0_dists=None,
		y1_dists=None,
		verbose=True,
		suppress_warning=False,
		**solve_kwargs,
	):
		"""
		Main function which computes dual bounds in three steps:
		(1) cross-fitting, (2) computing optimal dual variables,
		and (3) computing final dual bounds.

		Parameters
		----------	
		nfolds : int
			Number of folds to use when cross-fitting. Defaults to 5.
		alpha : float
			Nominal coverage level. Defaults to 0.05.
		aipw : bool
			If true, returns AIPW estimator.
		y0_dists : list
			list of batched scipy distributions whose shapes sum to n.
			the ith distribution is an out-of-sample estimate of
			the law of Yi(0) | X[i]. This is an optional input;
			if provided, ``Y_model`` will be ignored.
		y1_dists : list
			list of batched scipy distributions whose shapes sum to n.
			the ith distribution is an out-of-sample estimate of
			the law of Yi(1) | X[i]. This is an optional input;
			if provided, ``Y_model`` will be ignored.
		suppress_warning : bool
			If True, suppresses warning about cross-fitting.
		verbose : bool
			If True, gives occasional progress reports.
			Defaults to True.
		solve_kwargs : dict
			Additional (optional) kwargs for the ``compute_dual_variables``
			method, e.g. ``nvals0``, ``nvals1``, ``grid_size``.

		Returns
		-------
		estimates : np.array
			2-length array, estimates of lower/upper partial ident. bound
		ses : np.array
			Standard errors of ests
		cis : np.array
			Confidence bounds based on ests and ses. 
		"""
		# Fit model of W | X and Y | X if not provided
		self.y0_dists, self.y1_dists = y0_dists, y1_dists
		self.cross_fit(
			nfolds=nfolds, suppress_warning=suppress_warning
		)

		# compute dual variables
		self.compute_dual_variables(
			y0_dists=self.y0_dists,
			y1_dists=self.y1_dists,
			y0_min=self.y.min(),
			y0_max=self.y.max(),
			y1_min=self.y.min(),
			y1_max=self.y.max(),
			verbose=verbose,
			**solve_kwargs,
		)
		# compute dual bounds
		return self.compute_final_bounds(aipw=aipw, alpha=alpha)