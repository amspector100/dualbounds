import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import stats
from . import gen_data
from . import utilities
from . import dist_reg
from .utilities import BatchedCategorical

"""
Helper functions
"""
def interpolate(x, y, newx):
	"""
	x : np.array
		n-length array of inputs. Must be sorted, although
		this is not explicitly enforced to save time.
	y : np.array
		n-length array of outputs
	newx : np.array
		m-length array of new inputs
	"""
	if not utilities.haslength(newx):
		newx = np.array([newx])
	# for now, check sorting (TODO DELETE)
	# if np.any(np.sort(x) != x):
	# 	raise ValueError("NOT SORTED")
	# interpolate points in the range of x
	haty = np.interp(newx, x, y)
	# adjust for points < x.min()
	lflags = newx < x[0]
	ldx = (y[1] - y[0]) / (x[1] - x[0])
	haty[lflags] = y[0] + (newx[lflags] - x[0]) * ldx
	# adjust for points > x.max()
	uflags = newx > x[-1]
	udx = (y[-1] - y[-2]) / (x[-1] - x[-2])
	haty[uflags] = y[-1] + (newx[uflags] - x[-1]) * udx
	return haty

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
	lower=True,
	both=False,
	m=1000,
):
	"""
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
	bounds : np.array
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
	return s0_probs * np.stack([cvars_lower - y0ms, cvars_upper - y0ms], axis=0)

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
	abnds = compute_analytical_lee_bound(**args)[:, 0]
	return abnds

class LeeDualBounds:
	"""
	This class helps compute a lower (or upper) confidence
	bound on

	E[Y(1) S(1) S(0)].

	for binary S(0) and either continuous or discrete Y(1).

	Parameters
	----------
	nvals : int
		How many values to use to discretize Y(1)
	"""
	def __init__(self, nvals=50, monotonicity=True):
		self.nvals0 = 2 # in this case computes S(0)
		# we add 3 because:
		# 1. there is an extra value when S(1) = 0
		# 2. we have to add four quantiles very near the tails to 
		# ensure numerical stability (e.g., 0.0001 and 0.9999)
		self.nvals1 = max(nvals + 5, 6) 
		# parameters
		self.nu0 = cp.Variable((self.nvals0, 1))
		self.nu1 = cp.Variable((1, self.nvals1))
		self.f = cp.Parameter((self.nvals0, self.nvals1))
		self.probs0 = cp.Parameter((self.nvals0, 1))
		self.probs1 = cp.Parameter((1, self.nvals1)) # np.ones((1, nvals1)) / nvals1

		# ignore constraint [1][0] by monotonicity 
		mask = np.ones((self.nvals0, self.nvals1))
		if monotonicity:
			mask[1][0] = 0
		nusum_mask = cp.multiply(self.nu0 + self.nu1, mask)

		# Lower and upper constraints
		constraints_lower = [
			nusum_mask <= cp.multiply(self.f, mask), 
			cp.sum(self.nu0) == 0,
		]
		constraints_upper = [
			nusum_mask >= cp.multiply(self.f, mask),
			cp.sum(self.nu0) == 0,
		]
		# assemble objective and constraints
		self.obj = cp.sum(cp.multiply(self.nu0, self.probs0))
		self.obj = self.obj + cp.sum(cp.multiply(self.nu1, self.probs1))
		self.lproblem = cp.Problem(
			cp.Maximize(self.obj), 
			constraints_lower
		)
		self.uproblem = cp.Problem(
			cp.Minimize(self.obj), 
			constraints_upper
		)

	def ensure_feasibility(
		self,
		nu0,
		nu1,
		y1_vals,
		lower,
		ymin=-100,
		ymax=100,
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
		dx : subtract dx from nu0 and we get valid dual variables.
		"""
		if y1_vals[0] != 0:
			raise ValueError(
				"Expected y1_vals[0] to equal zero; this should be the case when s1 = 0."
			)
		dxs = []
		new_yvals = np.linspace(ymin, ymax, grid_size)
		interp_nu = interpolate(
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

	def discretize(
		self, ydists, nvals, alphas=None,
	):
		"""
		alphas : n-length array
			alphas = s0_probs / s1_probs
		"""
		# make sure we get small enough quantiles
		max_alpha = 1 / (2*nvals)
		if alphas is not None:
			alpha = min(max_alpha, alphas.min() / 2.1)
			alpha = min(alpha, (1 - alphas).min() / 2.1)
		else:
			alpha = max_alpha
		alpha = max(alpha, 1e-8)

		# choose endpoints of bins for disc. approx
		endpoints = np.sort(np.concatenate(
			[[0, alpha/2, alpha],
			np.linspace(1/(nvals-1), (nvals-2)/(nvals-1), nvals-1),
			[1-alpha, 1-alpha/2, 1]],
		))
		qs = (endpoints[1:] + endpoints[0:-1])/2
		# allow for batched setting
		if not isinstance(ydists, list):
			ydists = [ydists]
		# loop through batches and concatenate
		yvals = []
		for dists in ydists:
			yvals.append(dists.ppf(qs.reshape(-1, 1)).T)
		yvals = np.concatenate(yvals, axis=0)
		n = len(yvals)
		# return
		yprobs = endpoints[1:] - endpoints[0:-1]
		yprobs = np.stack([yprobs for _ in range(n)], axis=0)
		return yvals, yprobs

	def solve_instances(
		self,
		s0_probs,
		s1_probs,
		y1_dists=None,
		y1_vals=None,
		y1_probs=None,
		solver=None,
		verbose=False,
		**kwargs,
	):
		"""
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
		n = s0_probs.shape[0]
		# discretize if Y is continuous
		if y1_vals is None or y1_probs is None:
			# law of Y(1) | S(1) = 1, Xi
			y1_vals, y1_probs = self.discretize(y1_dists, nvals=self.nvals1-5)

		# ensure y1_vals, y1_probs are sorted
		inds = np.argsort(y1_vals, axis=1)
		y1_vals = np.take_along_axis(y1_vals, inds, axis=1)
		y1_probs = np.take_along_axis(y1_probs, inds, axis=1)
		
		## Adjust to make it law of Y(1) S(1) | Xi
		# note: it is important that the value when S(1) = 0
		# is the first value on the second axis
		# so that the monotonicity constraint is correct.
		self.y1_vals_adj = np.concatenate(
			[np.zeros((n, 1)), y1_vals], axis=1
		)
		self.y1_probs_adj = np.concatenate(
			[
				1 - s1_probs.reshape(-1, 1),
				s1_probs.reshape(-1, 1) * y1_probs
			], 
			axis=1
		)

		# useful constants
		s0_vals = np.array([0, 1]).reshape(-1, 1)

		# Initialize results
		self.nu0s = np.zeros((2, n, self.nvals0)) # first dimension = [lower, upper]
		self.nu1s = np.zeros((2, n, self.nvals1))
		# estimated cond means of nu0s, nu1s
		self.c0s = np.zeros((2, n))
		self.c1s = np.zeros((2, n))
		# objective values
		self.objvals = np.zeros((2, n))
		self.dxs = np.zeros((2, n))
		# loop through
		for i in utilities.vrange(n, verbose=verbose):
			# set parameter values
			self.f.value = (
				s0_vals * self.y1_vals_adj[i].reshape(1, -1)
			).astype(float)
			self.probs0.value = np.array(
				[1 - s0_probs[i], s0_probs[i]]
			).reshape(self.nvals0, 1)
			self.probs1.value = self.y1_probs_adj[i].astype(float).reshape(1, self.nvals1)
			# solve
			for lower in [0, 1]:
				if lower:
					objval = self.lproblem.solve(solver=solver)
				else:
					objval = self.uproblem.solve(solver=solver)
				self.objvals[1 - lower, i] = objval
				nu0x, nu1x, dx = self.ensure_feasibility(
					nu0=self.nu0.value.reshape(-1),
					nu1=self.nu1.value.reshape(-1),
					y1_vals=self.y1_vals_adj[i],
					lower=lower,
					**kwargs
				)
				self.nu0s[1 - lower, i] = nu0x
				self.nu1s[1 - lower, i] = nu1x
				self.c0s[1 - lower, i] = nu0x @ self.probs0.value.reshape(-1)
				self.c1s[1 - lower, i] = nu1x @ self.probs1.value.reshape(-1)
				self.dxs[1 - lower, i] = dx

	def compute_ipw_summands(
		self,
		Y,
		S,
		W,
		pis,
		y0s0_cond_means=None,
	):
		"""
		Method to compute IPW estimator summands.

		Parameters
		----------
		Y : np.array
			n-length array of treatments
		S : np.array
			n-length array of binary selection indicators
		W : np.array
			n-length array of treatment indicators.
		pis : np.array
			n-length array of P(W=1 | X)
		y0s0_cond_means : np.array
			n-length array of E[Y(0) S(0) | X].
			Optional argument, but required for AIPW estimation.
		"""
		# initialize outputs
		n = len(Y)
		self.ipw_summands = np.zeros((2, n))
		if y0s0_cond_means is not None:
			self.aipw_summands = np.zeros((2, n))

		# loop through data
		for i in range(n):
			pix_denom = pis[i] if W[i] == 1 else 1 - pis[i] 
			for lower in [0, 1]:
				# compute dual summand (ds)
				if W[i] == 0:
					ds = self.nu0s[1 - lower, i, S[i]]
					ds = ds - Y[i] * S[i] # identifiable component, e.g., E[Y(0) S(0)]
				else:
					if S[i] == 0:
						ds = self.nu1s[1 - lower, i, 0]
					else:
						ds = interpolate(
							x=self.y1_vals_adj[i, 1:], y=self.nu1s[1-lower, i, 1:],
							newx=Y[i],
						)
				# IPW summand 
				self.ipw_summands[1 - int(lower), i] = ds / pix_denom
				# AIPW summand; c0, c1 is estimated cond mean of ds given X
				if y0s0_cond_means is not None:
					c0 = self.c0s[1-lower, i] - y0s0_cond_means[i] # from ident. component
					c1 = self.c1s[1-lower, i]
					if W[i] == 0:
						aipws = (ds - c0) / pix_denom + c1 + c0
					else:
						aipws = (ds - c1) / pix_denom + c1 + c0
					self.aipw_summands[1 - int(lower), i] = aipws

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
			distribution is the conditional law of Yi(1) | S(1) = 1, Xi
		y1_dists : np.array
			batched scipy distribution of shape (n,) where the ith
			distribution is the conditional law of Yi(1) | S(1) = 1, Xi
		"""
		# estimated selection probs and outcome model 
		self.s0_probs, self.s1_probs = dist_reg._cross_fit_predictions(
			W=self.W, X=self.X, Y=self.S, 
			nfolds=nfolds, model=S_model,
			#model_cls=S_model_cls, **model_kwargs,
		)
		self.y0_dists, self.y1_dists = dist_reg._cross_fit_predictions(
			W=self.W, X=self.X, S=self.S, Y=self.Y, 
			nfolds=nfolds, model=Y_model,
		)
		# compute predicted mean of y0 s0---
		# to do this we need to loop through the batches
		self.y0s0_cond_means = []
		self.nstarts, self.nends = dist_reg.create_folds(n=len(self.Y), nfolds=nfolds)
		for y0dist, nstart, nend in zip(
			self.y0_dists, self.nstarts, self.nends
		):
			self.y0s0_cond_means.append(y0dist.mean() * self.s0_probs[nstart:nend])
		self.y0s0_cond_means = np.concatenate(self.y0s0_cond_means, axis=0)
		# returm
		return self.s0_probs, self.s1_probs, self.y0_dists, self.y1_dists


	def fit_propensity_scores(self, nfolds):
		raise NotImplementedError()

	def compute_dual_bounds(
		self,
		X,
		W,
		S,
		Y,
		S_model=None,
		Y_model=None,
		pis=None,
		nfolds=2,
		aipw=True,
		**solve_kwargs,
	):
		"""
		pis : n-length array
			Array of propensity scores. If None, will be estimated
			from the data itself.
		aipw : bool
			If true, returns AIPW estimator.
		solve_kwargs : dict
			kwargs to self.solve_instances(), 
			e.g., ``verbose``, ``solver``, ``grid_size``
		"""
		# Fit outcome model
		self.X = X
		self.Y = Y
		self.S = S
		self.W = W
		self.pis = pis
		# if pis not supplied: will use cross-fitting
		if self.pis is None:
			self.fit_propensity_scores(nfolds=nfolds)
		# fit outcome model using cross-fitting
		if S_model is None:
			self.S_model = dist_reg.LogisticCV(monotonicity=True)
		if Y_model is None:
			self.Y_model = dist_reg.RidgeDistReg(eps_dist='gaussian')
		self.fit_outcome_model(
			S_model=self.S_model,
			Y_model=self.Y_model,
			nfolds=nfolds
		)

		# compute dual variables
		self.solve_instances(
			s0_probs=self.s0_probs,
			s1_probs=self.s1_probs,
			y1_dists=self.y1_dists,
			ymin=self.Y.min(),
			ymax=self.Y.max(),
			**solve_kwargs,
		)

		# compute dual bounds
		self.compute_ipw_summands(
			Y=self.Y,
			S=self.S,
			W=self.W,
			pis=self.pis,
			y0s0_cond_means=self.y0s0_cond_means,
		)
		# estimators and bounds
		self.ests, self.bounds = utilities.compute_est_bounds(
			summands = self.aipw_summands if aipw else self.ipw_summands
		)
		return self.ests, self.bounds

class OracleLeeBounds(LeeDualBounds):

	def compute_oracle_bounds(
		self,
		X,
		W,
		S,
		Y,
		pis,
		s0_probs,
		s1_probs,
		y0_dists,
		y1_dists,
		aipw=True,
		**solve_kwargs,
	):	
		# dual variables
		self.solve_instances(
			s0_probs=s0_probs,
			s1_probs=s1_probs,
			y1_dists=y1_dists,
			ymin=Y.min(),
			ymax=Y.max(),
			**solve_kwargs,
		)
		y0s0_cond_means = y0_dists.mean() * s0_probs
		# ipw summands
		self.compute_ipw_summands(
			Y=Y,
			S=S,
			W=W,
			pis=pis,
			y0s0_cond_means=y0s0_cond_means,
		)
		# estimators and bounds
		self.ests, self.bounds = utilities.compute_est_bounds(
			summands = self.aipw_summands if aipw else self.ipw_summands
		)
		return self.ests, self.bounds