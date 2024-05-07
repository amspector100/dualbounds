import copy
import warnings
import numpy as np
import ot
from scipy import stats
from . import utilities, generic, dist_reg, interpolation
from .utilities import BatchedCategorical
import pandas as pd
# typing
from scipy.stats._distn_infrastructure import rv_generic
import sklearn.base
from typing import Optional, Union

"""
Note to self: do we need to define an IVDistReg wrapper? And use
multipler inheritance? IDK

Also, do we need to define a ConcatScipyDist wrapper...?
"""

class DualIVBounds(generic.DualBounds):
	"""
	This docstring needs to be constructed. Notes:
	(1) many parameter descriptions need to be changed etc,
	(2) instrument = Z, exposure = W. So propensities are Z | X.

	Other notes:
	1. self._discretize does not need to be changed.
	2. self._ensure_feasibility needs to be changed
	3. self._solve_single_instance can be wrapped, I think.
	4. compute_dual_variables def needs to be changed
	5. so does _interpolate_and_ensure_feas, 
	_compute_realized_dual_variables, _compute_ipw_summands, fit, etc.
	"""

	def __init__(
		self,
		exposure: Union[np.array, pd.Series],
		instrument: Union[np.array, pd.Series],
		*args, 
		**kwargs,
	):
		# Initialize base class
		super().__init__(
			treatment=exposure,
			*args,
			**kwargs,
		)
		# Save instrument
		if isinstance(instrument, pd.Series) or isinstance(instrument, pd.DataFrame):
			instrument = instrument.values.reshape(self.n)
		self.Z = utilities._binarize_variable(instrument, var_name='instrument')

	def _solve_single_instance(
		self,
		yvals: np.array,
		probs: np.array,
		x: np.array,
		**kwargs,
	):
		"""
		Parameters
		----------
		probs : np.array
			(2, 2, nvals)-shaped array where probs[z, w, y] =
			P(W(z) = w, Y(w) = yvals[z, w, y] | X = x).
		yvals : np.array
			(2, 2, nvals)-shaped array where
			yvals[z, w] is the support of Y(w) | W(z) = w
		x : np.array
			p-length covariate.

		Returns
		-------
		objvals : np.array
			array of the form np.array([lower_objval, upper_objval])
		nus : np.array
			(2, 2, 2, nvals)-shaped array where
			nus[1-lower, 0, w0] are the values of hatnu0(w0, y0)
			nus[1-lower, 1, w1] are the values of hatnu1(w1, y1)
		cs : np.array
			(2, 2)-shaped array of estimated cond means.
			The first axis corresponds to [lower, upper].
			The second axis corresponds to [z=0, z=1].
		"""
		# marginal probs of  W(k), Y(W(k)) | X
		## Reduce this problem to a regular OT problem.
		# Concatenate dists when Z = 0
		probs0 = np.concatenate([probs[0, 0], probs[0, 1]], axis=0)
		vals0 = np.concatenate([yvals[0, 0], yvals[0, 1]], axis=0)
		probs1 = np.concatenate([probs[1, 0], probs[1, 1]])
		vals1 = np.concatenate([yvals[1, 0], yvals[1, 1]], axis=0)

		# Compute dual variables
		nvals = yvals.shape[-1]
		objvals = np.zeros(2)
		nu0s = np.zeros((2, 2, nvals))
		nu1s = np.zeros((2, 2, nvals))
		c0s = np.zeros(2)
		c1s = np.zeros(2)
		for lower in [True, False]:
			# Evaluate f(w0, w1, y0, y1, x). 
			# This is not a generic OT constraint matrix
			# and needs to be adjusted depending on the value of lower.
			fvals = np.zeros((2*nvals, 2*nvals))
			for w0 in [0,1]:
				w0_vals = np.arange(nvals) + w0 * nvals
				for w1 in [0,1]:
					w1_vals = np.arange(nvals) + w1 * nvals
					fblock = self.f(
						w0=w0,
						w1=w1,
						y0=yvals[0, w0].reshape(-1, 1),
						y1=yvals[1, w1].reshape(1, -1),
						x=x,
					)
					# when w0 == w1, this is a linear non-OT constraint.
					# we use a hack to encode this within the ot 
					# framework so that we can use the efficient ot functions
					if w0 == w1:
						almost_inf = (1 + np.abs(fblock)).max() * 1e5
						if lower:
							fblock = fblock.min(axis=1-w0)
						else:
							fblock = fblock.max(axis=1-w0)
							almost_inf *= -1
						# A hack to make the non-diagonal constraints unimportant
						fblock = np.ones((nvals, nvals)) * almost_inf + np.diag(fblock - almost_inf)

					fvals[np.ix_(w0_vals, w1_vals)] = fblock

			# Solve
			if lower:
				objval, log = ot.lp.emd2(
					a=probs0,
					b=probs1,
					M=fvals,
					log=True,
				)
			else:
				objval, log = ot.lp.emd2(
					a=probs0,
					b=probs1,
					M=-1*fvals,
					log=True,
				)
				objval *= -1
				log['u'] *= -1
				log['v'] *= -1
			# Extract objective value
			objvals[int(1-lower)] = objval
			# Extract dual variables
			nu0s[int(1-lower)] = np.stack([log['u'][0:nvals], log['u'][nvals:]], axis=0)
			nu1s[int(1-lower)] = np.stack([log['v'][0:nvals], log['v'][nvals:]], axis=0)
			# Center
			shift = nu0s[int(1-lower)].mean()
			nu1s[int(1-lower)] += shift
			nu0s[int(1-lower)] -= shift
			# Estimated conditional means
			c0s[int(1-lower)] = np.sum(nu0s[int(1-lower)] * probs[0])
			c1s[int(1-lower)] = np.sum(nu1s[int(1-lower)] * probs[1])
 

		# Concatenate and return
		nus = np.stack([nu0s, nu1s], axis=1)
		cs = np.stack([c0s, c1s], axis=1)

		return objvals, nus, cs

	def _ensure_feasibility(
		self,
		i,
		orig_nus,
		lower,
		ymin,
		ymax,
		grid_size=100,
		tol=5e-4,
	):
		"""
		Parameters
		----------
		i : int
			Index of which data-point we are performing
			this operation for.
		orig_nus : np.array
			(2, 2, nvals)-length array of dual variables 
			First axis is associated with [z=0, z=1]
			Second axis is associated with [w=0, w=1]
			Third axis is associated with [self._yvals[i, z, w]]
		lower : bool
			Specifies lower vs. upper bound.
		ymin : float
			Minimum value of y
		ymax : float
			Maximum value of y
		grid_size : int
			Grid size along each dimension.

		Returns
		-------
		new_yvals : list
			new_yvals[z][w] contains a  ``nvals + grid_size`` length
			array of new yvals.
			*Exact size may change to avoid duplicate values
		new_nus : list
			new_nus are the dual variables for new_yvals.
		dx : float
			Maximum numerical error induced by interpolation
			process.
		objval_diff : np.array
			The estimated objective value change for the new
			dual variables.
		"""
		# original objective value
		probs = self._jointprobs[i]
		obj_orig = np.sum(probs * orig_nus)
		if grid_size > 0:
			# Construct new yvals
			extra_yvals = np.linspace(ymin, ymax, grid_size)
			new_yvals = [[], []]
			for z in [0,1]:
				for w in [0,1]:
					# yvals
					new_yvals[z].append(np.sort(np.unique(np.concatenate(
						[self._yvals[i, z, w], extra_yvals], axis=0
					))))
			# New fvals
			raise NotImplementedError("FVALS ARE WRONG NEED TO CHANGE THIS")
			new_fvals = [[], []]
			for w0 in [0,1]:
				for w1 in [0,1]:
					# fvals
					new_fvals[w0].append(
						self.f(
							w0=w0,
							w1=w1,
							y0=new_yvals[0][w0].reshape(-1, 1),
							y1=new_yvals[1][w1].reshape(1, -1),
							x=self.X[i],
						)
					)
			# Interpolate nus
			new_nus = [[], []]
			for z in [0,1]:
				for w in [0,1]:
					new_nus[z].append(self.interp_fn(
						x=self._yvals[i, z, w], y=orig_nus[z, w], newx=new_yvals[z][w]
					))
			# Check feasibility for each w0, w1
			for w0 in [0,1]:
				for w1 in [0,1]:
					deltas = new_nus[0][w0].reshape(-1, 1) + new_nus[1][w1].reshape(1, -1)
					deltas = deltas - new_fvals[w0][w1]
					# Figure out which axis to adjust
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
					# Adjust
					if adj_axis == 0:
						new_nus[0][w0] -= deltas0
					else:
						new_nus[1][w1] -= deltas1
			
			# Compute difference in objective values by remapping to original shape
			adj_nus = np.zeros(orig_nus.shape)
			for z in [0,1]:
				for w in [0,1]:
					adj_nus[z, w] = self.interp_fn(
						x=new_yvals[z][w], y=new_nus[z][w], newx=self._yvals[i, z, w],
					)
			objval_diff = obj_orig - np.sum(adj_nus * probs)
		else:
			new_yvals = self._yvals[i]
			new_nus = orig_nus
			objval_diff = 0
		# return
		return new_yvals, new_nus, objval_diff

	def _interpolate_and_ensure_feas(
		self, yi, wi, zi, lower, i, ymin, ymax, **kwargs
	):  
		"""
		This is used in two places:
		1. _compute_realized_dual_variables
		2. Main loop of compute_dual_variables

		We use this in the main loop of compute_dual_variables
		instead of just calling _compute_realized_dual_variables
		so that the progress report timing is accurate.
		"""
		nusx = self.nus[i, 1-lower]
		# Ensure feasibilitynus=nusx,
		if not self.discrete:
			new_yvals, new_nus, odx = self._ensure_feasibility(
				i=i, 
				orig_nus=nusx, 
				lower=lower, 
				ymin=ymin,
				ymax=ymax, 
				**kwargs
			)
			self.objdiffs[1-lower, i] = odx
			# interpolate to find realized values 
			self.hatnus[1-lower, i] = self.interp_fn(
				x=new_yvals[zi][wi], y=new_nus[zi][wi], newx=yi,
			)
		else:
			j = np.where(self._yvals[i, zi, wi] == yi)[0][0]
			self.hatnus[1-lower, i] = nusx[zi, wi, j]

	def compute_dual_variables(
		self,
		wprobs: np.array,
		ydists: list,
		ymin: Optional[float]=None,
		ymax: Optional[float]=None,
		min_quantile: float=1e-5,
		interp_fn: callable=interpolation.adaptive_interpolate,
		nvals: int=100,
		ninterp: Optional[int]=None,
		verbose: bool=True,
		**kwargs
	):
		"""
		Docs in progress.

		Parameters
		----------
		wprobs : np.array
			wprobs[i, z] is the estimated probability 
			:math:`P(W(z) = i | X_i)`
		ydists : list
			For z, w in {0,1}, ydists[z][w] must be a list of batched
			scipy dists whose shapes sum to n. The ith element 
			of ydists[z][w] represents the law of 
			:math:`Y_i(w) | W_i(z) = w, X_i`.
		min_quantile : float
			A tolerance used to ensure numerical stability in the dual solver. 
			Essentially, in a discretization, considers quantiles as small as
			tol from the estimated law of ydists.
		"""
		# Setup
		if verbose:
			print("Estimating optimal dual variables.")
		self.interp_fn = interp_fn
		if self.discrete:
			self.nvals = len(self.support)
		else:
			self.nvals = nvals
			# if self.nvals <= generic.MIN_NVALS

		####TODO: need to infer the ymin/ymaxes but that's okay for now
		if ymin is None:
			ymin = self.y.min() - (self.y.max() - self.y.min()) / 2
		if ymax is None:
			ymax = self.y.max() + (self.y.max() - self.y.min()) / 2

		# Discretize if necessary
		self._yvals = [[], []]
		self._yprobs = [[], []]
		for z in [0,1]:
			for w in [0,1]:
				yvals_zw, yprobs_zw = self._discretize(
					ydists[z][w], 
					nvals=self.nvals, 
					alpha=min_quantile,
					ymin=ymin,
					ymax=ymax,
					ninterp=ninterp,
				)
				# ensure everything is sorted across the nvals axis
				yvals_zw, yprobs_zw = utilities._sort_disc_dist(
					vals=yvals_zw, probs=yprobs_zw
				)
				self._yvals[z].append(yvals_zw)
				self._yprobs[z].append(yprobs_zw)

		##TODO: need to test that this concatenation is correct
		# Shape: (n, 2, 2, nvals) with the 2s corresponding to (z,w) in that order.
		self._yvals = np.stack(
			[np.stack(self._yvals[0], axis=1), np.stack(self._yvals[1], axis=1)],
			axis=1
		)
		self._yprobs = np.stack(
			[np.stack(self._yprobs[0], axis=1), np.stack(self._yprobs[1], axis=1)],
			axis=1
		)
		# joint law of (W, Y(W)) | Z
		self._jointprobs = self._yprobs.copy()
		self._jointprobs[:, :, 0] *= (1 - wprobs.reshape(self.n, 2, 1))
		self._jointprobs[:, :, 1] *= wprobs.reshape(self.n, 2, 1)

		## Initialize results
		# first dimension = [lower, upper]
		# second dimension = [z=0, z=1]
		# third dimension = [w=0, w=1]
		self.nus = np.zeros((self.n, 2, 2, 2, self.nvals))
		# realized dual variables
		self.hatnus = np.zeros((2, self.n))
		self.objvals = np.zeros((2, self.n))
		self.objdiffs = np.zeros((2, self.n))
		# estimated conditional means of nus given Z (NOT given W)
		# first dimension = [lower, upper]
		# second dimension = [z=0, z=1]
		self.cs = np.zeros((self.n, 2, 2)) 
		for i in utilities.vrange(self.n, verbose=verbose):
			# Compute optimal dual variable functions
			objvalsx, nusx, csx = self._solve_single_instance(
				yvals=self._yvals[i],
				probs=self._jointprobs[i],
				x=self.X[i],
				ymin=ymin,
				ymax=ymax,
			)
			self.objvals[:, i] = objvalsx
			self.nus[i] = nusx
			self.cs[i] = csx
			for lower in [0,1]:
				# Compute realized dual variables
				self._interpolate_and_ensure_feas(
					yi=self.y[i],
					zi=self.Z[i],
					wi=self.W[i],
					lower=lower,
					ymin=ymin,
					ymax=ymax,
					i=i,
				)

	def _compute_realized_dual_variables(self, y=None, Z=None, W=None, **kwargs):
		"""
		Helper function which applies interpolation 
		(for continuous Y) to compute realized dual variables.
		It also ensures feasibility through a 2D gridsearch.
		This is used primarily in unit tests.
		"""
		y = self.y if y is None else y
		W = self.W if W is None else W
		Z = self.Z if Z is None else Z

		# Create ylims
		ymin = kwargs.pop("ymin", y.min() - (y.max() - y.min()) / 2)
		ymax = kwargs.pop("ymax", y.max() + (y.max() - y.min()) / 2)

		### Compute realized hatnu1s/hatnu0s
		for i in range(self.n):
			for lower in [0, 1]:
				self._interpolate_and_ensure_feas(
					i=i,
					yi=y[i],
					zi=Z[i],
					wi=Z[i],
					lower=lower, 
					ymin=ymin,
					ymax=ymax,
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

		# Compute IPW summands and AIPW summands
		Z = self.Z.reshape(1, self.n)
		pis = self.pis.reshape(1, self.n)
		ipw_denom = Z / (pis) + (1 - Z) / (1 - pis) 
		self.ipw_summands = self.hatnus / ipw_denom
		# Compute AIPW summands
		mus = self.cs.sum(axis=2).T # (2, n)-shaped array
		deltas = self.cs[:, :, 1].T * Z + self.cs[:, :, 0].T * (1 - Z) # (2, n)-shaped array
		self.aipw_summands = (self.hatnus - deltas) / ipw_denom + mus

	def _compute_cond_means(self):
		raise NotImplementedError()

	def _compute_oos_resids(self):
		raise NotImplementedError()

	def fit(self):
		raise NotImplementedError()

	def summary(self):
		raise NotImplementedError()