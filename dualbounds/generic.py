import numpy as np
import cvxpy as cp
from scipy import stats
from . import utilities, dist_reg, interpolation
from .utilities import BatchedCategorical

class DualBounds:
	"""
	Computes dual bounds on E[f(Y(0),Y(1), X)].


	Parameters
	----------
	Parameters
	----------
	Y : np.array
		n-length array of treatments
	X : np.array
		(n, p)-shaped array of covariates.
	W : np.array
		n-length array of treatment indicators.
	pis : np.array
		n-length array of propensity scores P(W=1 | X). 
		If ``None``, will be estimated from the data itself.
	f : function
		Function of three arguments, defines the expectation
		which is the partially identified estimand.
	nvals0 : int
		How many values to use to discretize Y(0). 
		Defaults to 25.
	nvals1 : int
		How many values to use to discretize Y(1).
		Defaults to 25.
	interp_fn : function
		An interpolation function with the same input/output
		signature as ``interpolation.linear_interpolate``,
		which is the default.
	"""
	def __init__(
		self, 
		X,
		W,
		y,
		f,
		pis=None,
		nvals0=25,
		nvals1=25,
		interp_fn=interpolation.linear_interpolate,
	):
		### Data
		self.X = X
		self.y = y
		self.W = W
		self.f = f
		self.pis = pis
		self.n, self.p = self.X.shape
		self.interp_fn = interp_fn

		### Key quantities for optimizer
		# to ensure numerical stability, we add extra quantiles
		self.nvals0 = nvals0
		self.nvals1 = nvals1
		# parameters
		self.nu0 = cp.Variable((nvals0, 1))
		self.nu1 = cp.Variable((1, nvals1))
		self.fparam = cp.Parameter((nvals0, nvals1))
		self.probs0 = cp.Parameter((nvals0, 1))
		self.probs1 = cp.Parameter((1, nvals1))
		constraints_lower = [
			self.nu0 + self.nu1 <= self.fparam,
			cp.sum(self.nu0) == 0,
		]
		constraints_upper = [
			self.nu0 + self.nu1 >= self.fparam,
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

	def discretize(
		self, ydists, nvals, alphas=None,
	):
		"""
		alphas : n-length array
			Additional quantiles to add to ensure numerical
			stability. For lee bounds,
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

	def ensure_feasibility(
		i,
		nu0,
		nu1,
		lower,
		ymin,
		ymax,
		grid_size=100,
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
		new_y0vals = np.linspace(ymin, ymax, grid_size)
		new_y1vals = np.linspace(ymin, ymax, grid_size)
		interp_nu0 = self.interp_fn(
			x=self.y0_vals[i], y=nu0, newx=new_y0vals,
		)
		interp_nu1 = self.interp_fn(
			x=self.y1_vals[i], y=nu1, newx=new_y1vals,
		)
		# compute required bounds
		fx = lambda y0, y1: self.f(y0, y1, self.X[i])
		fvals = fx(
			new_y0vals[i].reshape(-1, 1), 
			new_y1vals[i].reshape(1, -1)
		)
		# compute dx
		deltas = interp_nu0.reshape(-1, 1) + interp_nu1.reshape(1, -1)
		deltas = deltas - fvals 
		if lower:
			dx = np.max(deltas)
		else:
			dx = np.min(deltas)
		# return
		return nu0 - dx/2, nu1 - dx/2, dx

	def solve_instances(
		self,
		y0_dists=None,
		y0_vals=None,
		y0_probs=None,
		y1_dists=None,
		y1_vals=None,
		y1_probs=None,
		solver=None,
		verbose=False,
		**kwargs,
	):
		"""
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
		y1_probs : np.array
			(n, nvals1)-length array where
			y0_probs[i, j] = P(Y(1) = yvals1[j] | Xi)
		kwargs : dict
			kwargs for ensure_feasibility method.
			Includes ymin, ymax, grid_size.
		"""
		# discretize if Y is continuous
		if y0_vals is None or y0_probs is None:
			y0_vals, y0_probs = self.discretize(y0_dists, nvals=self.nvals0-5)
		if y1_vals is None or y1_probs is None:
			y1_vals, y1_probs = self.discretize(y1_dists, nvals=self.nvals1-5)

		# ensure y1_vals, y1_probs are sorted
		self.y0_vals, self.y0_probs = utilities._sort_disc_dist(y0_vals, y0_probs)
		self.y1_vals, self.y1_probs = utilities._sort_disc_dist(y1_vals, y1_probs)
		
		# Initialize results
		self.all_nu0s = np.zeros((2, self.n, self.nvals0)) # first dimension = [lower, upper]
		self.all_nu1s = np.zeros((2, self.n, self.nvals1))
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
			fx = lambda y0, y1: self.c(y0, y1, self.X[i])
			fvals = fx(y0_vals[i].reshape(-1, 1), y1_vals[i].reshape(1, -1))
			self.fparam.value = np.array(fvals).astype(float)
			self.probs0.value = self.y0_probs[i].astype(float).reshape(1, self.nvals0)
			self.probs1.value = self.y1_probs[i].astype(float).reshape(1, self.nvals1)
			# solve
			for lower in [0, 1]:
				if lower:
					objval = self.lproblem.solve(solver=solver)
				else:
					objval = self.uproblem.solve(solver=solver)
				self.objvals[1 - lower, i] = objval
				nu0x, nu1x, dx = self.ensure_feasibility(
					i=i,
					nu0=self.nu0.value.reshape(-1),
					nu1=self.nu1.value.reshape(-1),
					lower=lower,
					**kwargs
				)
				self.all_nu0s[1 - lower, i] = nu0x
				self.all_nu1s[1 - lower, i] = nu1x
				self.c0s[1 - lower, i] = nu0x @ self.probs0.value.reshape(-1)
				self.c1s[1 - lower, i] = nu1x @ self.probs1.value.reshape(-1)
				self.dxs[1 - lower, i] = dx
				# interpolate to find realized values
				self.hatnu0s[1 - lower, i] = self.interp_fn(
					x=self.y0_vals[i], y=nu0x, newx=self.Y[i],
				)
				self.hatnu1s[1 - lower, i] = self.interp_fn(
					x=self.y1_vals[i], y=nu1x, newx=self.Y[i],
				)

	def compute_ipw_summands(self):
		"""
		Method to compute (A)IPW estimator summands.
		Must be run AFTER running ``self.solve_instances``
		"""
		# initialize outputs
		self.ipw_summands = np.zeros((2, n))
		self.aipw_summands = np.zeros((2, n))

		# compute IPW summands and AIPW summands
		for lower in [0, 1]:
			# IPW
			self.ipw_summands[1-lower] = self.hatnu1s[1-lower] / self.pis
			self.ipw_summands[1-lower][self.W == 0] = (
				self.hatnu0s[1-lower] / (1 - self.pis)
			)[self.W == 0]
			# AIPW
			mus = self.c1s[1-lower] - self.c0s[1-lower]
			aipw1 = (self.hatnus1[1-lower] - self.c1s[1-lower]) / self.pis + mus 
			aipw0 = (self.hatnus0[1-lower] - self.c0s[1-lower]) / (1-self.pis) + mus
			self.aipw_summands[1-lower] = aipw1 * self.W + (1 - self.W) * aipw0

	def compute_dual_bounds(
		self,
		Y_model=None,
		nfolds=5,
		aipw=True,
		**solve_kwargs,
	):
		"""
		aipw : bool
			If true, returns AIPW estimator.
		solve_kwargs : dict
			kwargs to self.solve_instances(), 
			e.g., ``verbose``, ``solver``, ``grid_size``
		"""
		# if pis not supplied: will use cross-fitting
		if self.pis is None:
			self.fit_propensity_scores(nfolds=nfolds)

		# Fit model
		if Y_model is None:
			Y_model = dist_reg.RidgeDistReg(eps_dist='gaussian')
		self.Y_model = Y_model
		self.y0_dists, self.y1_dists = dist_reg._cross_fit_predictions(
			W=self.W, X=self.X, Y=self.Y, 
			nfolds=nfolds, model=Y_model,
		)

		# compute dual variables
		self.solve_instances(
			y0_dists=self.y0_dists,
			y1_dists=self.y1_dists,
			ymin=self.Y.min(),
			ymax=self.Y.max(),
			**solve_kwargs,
		)
		# compute dual bounds
		self.compute_ipw_summands()
		# estimators and bounds
		self.ests, self.bounds = utilities.compute_est_bounds(
			summands = self.aipw_summands if aipw else self.ipw_summands
		)
		return self.ests, self.bounds