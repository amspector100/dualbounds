import numpy as np
import cvxpy as cp
from scipy import stats
from . import utilities
from . import dist_reg
from .utilities import BatchedCategorical
from . import interpolation
from .generic import get_default_model,DualBounds

"""
Helper functions
"""
def compute_cvar(dists, n, alpha, lower=True, m=1000):
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
	Computes partial ident. bounds on E[Y(1) - Y(0) | S(0) = S(1) = 1]
	under monotonicity.

	Parameters
	----------
	n : int
		Number of samples.
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
		and bounds[1,i] is the ith upper bound.
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
	cond_bounds = np.stack([cvars_lower - y0ms, cvars_upper - y0ms], axis=0)
	agg_bounds = np.mean(cond_bounds * s0_probs, axis=1) / np.mean(s0_probs)
	return agg_bounds, cond_bounds

def lee_bound_no_covariates(
	W, S, Y,
):
	n = len(W)
	# compute P(S | W)
	s0_prob = np.array([np.mean(S[W == 0])])
	s1_prob = np.array([np.mean(S[W == 1])])
	s1_prob = np.maximum(s0_prob, s1_prob)
	# compute P(Y(0)) and P(Y(1))
	y0_vals = np.sort(Y[(W == 0) & (S == 1)])
	y0_probs = np.ones(len(y0_vals)) / len(y0_vals)
	y1_vals = np.sort(Y[(W == 1) & (S == 1)])
	y1_probs = np.ones(len(y1_vals)) / len(y1_vals)
	args = dict(
		s0_probs=s0_prob,
		s1_probs=s1_prob,
		y0_probs=y0_probs.reshape(1, -1),
		y0_vals=y0_vals.reshape(1, -1),
		y1_probs=y1_probs.reshape(1, -1),
		y1_vals=y1_vals.reshape(1, -1)
	)
	# compute lower, upper bounds
	abnds = compute_analytical_lee_bound(**args)[1][:, 0]
	return abnds

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

class LeeDualBounds(DualBounds):
	"""
	Computes dual bounds on

	E[Y(1) - Y(0) | S(1) = S(0) = 1]

	for binary S(0) and either continuous or discrete Y(1).
	These bounds assume monotonicity.

	Parameters
	----------
	X : np.array
		(n, p)-shaped array of covariates.
	S : np.array
		n-length array of binary selection indicators
	W : np.array
		n-length array of treatment indicators.
	Y : np.array
		n-length array of outcome measurements
	pis : np.array
		n-length array of propensity scores P(W=1 | X). 
		If ``None``, will be estimated from the data itself.
	discrete : bool
		If True, treats y as a discrete variable. 
		Defaults to None (inferred from the data).
	"""	

	def __init__(
		self,
		S,
		monotonicity='conditional',
		*args,  
		**kwargs
	):
		self.monotonicity = monotonicity
		self.S = S
		kwargs['f'] = None
		super().__init__(*args, **kwargs)

	def ensure_feasibility(
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
		Parameters
		----------
		s0_probs : np.array
			n-length array where s0_probs[i] = P(S(0) = 1 | Xi)
		s1_probs : np.array
			n-length array where s1_probs[i] = P(S(1) = 1 | Xi)
		y1_dists : np.array
			batched scipy distribution of shape (n,) where the ith
			distribution is the conditional law of Yi(1) | S(1) = 1, Xi
			OR 
			list of scipy dists whose shapes add up to n.
		y1_vals : np.array
			(n, nvals1)-length array of values y1 can take.
		y1_probs : np.array
			(n, nvals1)-length array where
			y0_probs[i, j] = P(Y(1) = yvals1[j] | S(1) = 1, Xi)
		kwargs : dict
			kwargs for ensure_feasibility method.
			Includes ymin, ymax, grid_size.
		"""
		# Constants for solver
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
			alpha = min([np.min(s0_probs/s1_probs), np.min(1 - s0_probs/s1_probs)])
			# law of Y(1) | S(1) = 1, Xi
			y1_vals, y1_probs = self.discretize(
				y1_dists, 
				nvals=self.nvals1-1, 
				alpha=alpha/2,
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
			max_fval = np.abs(fvals).max() * 1e5
			# helpful concatenation
			probs0 = np.array([1 - s0_probs[i], s0_probs[i]])
			# solve 
			for lower in [1, 0]:
				# Relax constraints due to monotonicity
				fvals[1][0] = max_fval if lower else -max_fval
				nu0x, nu1x, objval = self._solve_single_instance(
					probs0=probs0,
					probs1=self.y1_probs_adj[i],
					fvals=fvals,
					lower=lower
				)
				self.objvals[1 - lower, i] = objval
				if not self.discrete:
					nu0x, nu1x, dx = self.ensure_feasibility(
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

	def fit_outcome_model(
		self,
		S_model,
		Y_model,
		nfolds=2,
	):
		"""
		Returns
		-------
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
		"""
		# estimated selection probs and outcome model 
		self.s0_probs, self.s1_probs, self.fit_S_models = dist_reg._cross_fit_predictions(
			W=self.W, X=self.X, Y=self.S, 
			nfolds=nfolds, model=S_model,
			probs_only=True,
			#model_cls=S_model_cls, **model_kwargs,
		)
		self.y0_dists, self.y1_dists, self.fit_Y_models = dist_reg._cross_fit_predictions(
			W=self.W, X=self.X, S=self.S, Y=self.y, 
			nfolds=nfolds, model=Y_model,
		)
		# return
		return self.s0_probs, self.s1_probs, self.y0_dists, self.y1_dists


	def compute_dual_bounds(
		self,
		Y_model=None,
		S_model=None,
		nfolds=5,
		aipw=True,
		alpha=0.05,
		**solve_kwargs,
	):
		"""
		Parameters
		----------
		alpha : float
			Nominal coverage level. Defaults to 5.
		aipw : bool
			If true, returns AIPW estimator.
		solve_kwargs : dict
			kwargs to self.compute_dual_variables(), 
			e.g., ``verbose``, ``nvals0``, ``grid_size``
		"""
		# if pis not supplied: will use cross-fitting
		if self.pis is None:
			self.fit_propensity_scores(nfolds=nfolds)
		# fit outcome models using cross-fitting
		if S_model is None:
			S_model = dist_reg.LogisticCV(monotonicity=True)
		self.S_model = S_model
		self.Y_model = get_default_model(
			discrete=self.discrete, support=self.support, Y_model=Y_model
		)
		self.fit_outcome_model(
			S_model=self.S_model,
			Y_model=self.Y_model,
			nfolds=nfolds
		)

		# compute dual variables
		self.compute_dual_variables(
			s0_probs=self.s0_probs,
			s1_probs=self.s1_probs,
			y1_dists=self.y1_dists,
			ymin=self.y.min(),
			ymax=self.y.max(),
			**solve_kwargs,
		)

		# compute dual bounds
		return self.compute_final_bounds(aipw=aipw, alpha=alpha)


	def compute_final_bounds(self, aipw=True, alpha=0.05):
		"""
		Computes final bounds based in (A)IPW summands,
		using the delta method for E[Y(1) - Y(0) | S(1) = S(0) = 1].
		"""
		self.compute_ipw_summands()
		summands = self.aipw_summands if aipw else self.ipw_summands
		self._compute_cond_means()
		self.y0s0_cond_means = self.mu0 * self.s0_probs
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

		self.ests = np.maximum(np.array(ests), 0)
		self.ses = np.array(ses)
		self.bounds = np.maximum(np.array(bounds), 0)
		return self.ests, self.bounds

class OracleLeeBounds(LeeDualBounds):

	def compute_oracle_bounds(
		self,
		s0_probs,
		s1_probs,
		y0_dists,
		y1_dists,
		aipw=True,
		**solve_kwargs,
	):	
		# dual variables
		self.y0_dists = y0_dists
		self.s0_probs = s0_probs
		self.compute_dual_variables(
			s0_probs=s0_probs,
			s1_probs=s1_probs,
			y1_dists=y1_dists,
			ymin=self.y.min(),
			ymax=self.y.max(),
			**solve_kwargs,
		)
		self.compute_final_bounds()
		return self.ests, self.bounds
		