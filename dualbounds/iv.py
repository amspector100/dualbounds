import copy
import warnings
import numpy as np
import ot
import cvxpy as cp
from scipy import stats
from . import utilities, generic, dist_reg, interpolation
from .utilities import BatchedCategorical
import pandas as pd
# typing
from scipy.stats._distn_infrastructure import rv_generic
import sklearn.base
from typing import Optional, Union

"""
Note to self: do we need to define a ConcatScipyDist wrapper...?
"""

def _clip_jointprobs(jointprobs):
	jointprobs = np.clip(jointprobs, 0, 1)
	for z in [0,1]:
		jointprobs[z] /= jointprobs[z].sum()
	return jointprobs

def _compute_max_dualval(logs):
	max_dualval = 0
	for lower in [0,1]:
		for key in ['u', 'v']:
			max_dualval = max(np.abs(logs[lower][key]).max(), max_dualval)
	return max_dualval


class DualIVBounds(generic.DualBounds):
	"""
	Beta version. Computes dual bounds on 
	:math:`E[f(W(0), W(1), Y(0),Y(1), X)].`
	in the instrumental variables context.

	Here, :math:`X` are covariates, :math:`Y(0), Y(1)` are 
	potential outcomes, and :math:`W(0), W(1)` are potential
	outcomes of a binary exposure/treatment.

	Parameters
	----------
	f : function
		Function which defines the partially identified estimand.
		Must be a function of three arguments: w0, w1, y0, y1, x 
		(in that order). E.g.,
		``f = lambda w0, w1, y0, y1, x : (y0 <= y1) * (w0 <= w1)``
	outcome : np.array | pd.Series
		n-length array of outcome measurements (Y).
	instrument : np.array | pd.Series
		n-length array of binary instrument (Z).
	exposure : np.array | pd.Series
		n-length array of binary exposure (W).
	covariates : np.array | pd.Series
		(n, p)-shaped array of covariates (X).
	propensities : np.array | pd.Series
		n-length array of propensity scores :math:`P(Z=1 | X)`. 
		If ``None``, will be estimated from the data.
	clusters : np.array | pd.Series
		Optional n-length array of clusters, so ``clusters[i] = j``
		indicates that observation i is in cluster j.
	outcome_model : str | dist_reg.DistReg | list
		The model for estimating the law of :math:`Y | X, W, Z`.
		Three options:

		- A str identifier, e.g., 'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'.
		- An object inheriting from ``dist_reg.DistReg``. 
		- A list of ``dist_reg.DistReg`` objects to automatically choose between.

		E.g., when ``outcome`` is continuous, the default is
		``outcome_model=dist_reg.CtsDistReg(model_type='ridge')``.
	exposure_model : str | dist_reg.DistReg
		The model for estimating the law of :math:`W | X, Z`.
		Two options:

		- A str identifier, e.g., 'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'.
		- An object inheriting from ``dist_reg.DistReg``.
		The default is
		``exposure_model=dist_reg.BinaryDistReg(model_type='ridge')``.

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
		Boolean-valued function of w0, w1, y0, y1, x where 
		``support_restriction(w0, w1, y0, y1, x) = False`` asserts that 
		w0, w1, y0, y1, x is not in the support of 
		:math:`W(0), W(1), Y(0), Y(1), X`.
		Defaults to ``None`` (no a-priori support restrictions).
		See the user guide for important usage tips.
	model_kwargs : dict
		Additional kwargs for the ``outcome_model``, e.g.,
		``feature_transform``. See 
		:class:`dualbounds.dist_reg.CtsDistReg` or 
		:class:`dualbounds.dist_reg.BinaryDistReg` for more kwargs.
	suppress_iv_warning : bool
		If True, suppresses the beta warning for DualIVBounds.

	Notes
	-----
	This is currently slower than the ``DualBounds`` class.
	"""

	def __init__(
		self,
		exposure: Union[np.array, pd.Series],
		instrument: Union[np.array, pd.Series],
		exposure_model: Union[str, dist_reg.DistReg]='ridge',
		suppress_iv_warning: bool=False,
		*args, 
		**kwargs,
	):
		if not suppress_iv_warning:
			print("Note: DualIVBounds are in beta. Feedback is welcome. To suppress this warning, set suppress_iv_warning=True.")
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
		self.exposure_model = exposure_model

	def _solve_single_instance_cp(
		self,
		probs0: np.array,
		probs1: np.array,
		fvals: np.array,
		not_binding: np.array,
		lower: np.array,
		objval_ot: float,
		solver='CLARABEL',
	):
		"""
		Fallback solver using CP to minimize a norm-penalized variant.

		Parameters
		----------
		probs0 : array
			nvals0-length array of marginal probabilities
		probs1 : array
			nvals1-length array of marginal probabilities
		fvals : array
			(nvals0, nvals1)-length array of constraint values
		lower : bool
			If True, solves the problem for the lower CI. Else
			solves the problem for the upper CI.
		tol : float
			Ensure we are within tol of the objective.
		not_binding : array
			(nvals0, nvals1)-length boolean array. True indicates
			that something is not binding, typically because it 
			violates an a-priori constraint or is a "padded" value.

		Returns
		-------
		objval : float
			Objective value
		nu0 : array
			nvals0-length array of dual variables
		nu1 : array
			nvals1-length array of dual variables

		Notes
		-----
		This should be adapted/moved to dualbounds.generic. 
		"""
		nvals0, nvals1 = len(probs0), len(probs1)
		nu0 = cp.Variable(nvals0)
		nu1 = cp.Variable(nvals1)
		objval_unreg = cp.Parameter()
		objval_unreg.value = objval_ot
		# Constraints
		nusum = cp.reshape(nu0, (-1, 1)) + cp.reshape(nu1, (1, -1))
		flags = (~not_binding).astype(int)
		if lower:
			constraints = [
				cp.multiply(nusum, flags) <= cp.multiply(fvals, flags)
			]
		else:
			constraints = [
				cp.multiply(nusum, flags) >= cp.multiply(fvals, flags)
			]

		# Stage 1: maximize linobj subject to constraints
		linobj = nu0 @ probs0 + nu1 @ probs1
		# if lower:
		# 	lp = cp.Problem(cp.Maximize(linobj), constraints=constraints)
		# else:
		# 	lp = cp.Problem(cp.Minimize(linobj), constraints=constraints)
		# lp.solve(solver=solver)
		# print(lp.value, lp.status)
		# objval_unreg.value = lp.value

		# Stage 2: find min norm optimal solution
		norm2 = cp.norm2(nu0) + cp.norm2(nu1)
		# Constraints on true objective value
		if lower:
			constraints.append(linobj >= objval_unreg)
		else:
			constraints.append(linobj <= objval_unreg)
		# Minimize norm subject to constraints
		problem = cp.Problem(cp.Minimize(norm2), constraints=constraints)
		# Back off until we find a feasible problem (this is for numerical stability)
		for tol in [0, 1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 0, 1, 3]:
			if lower:
				objval_unreg.value -= tol
			else:
				objval_unreg.value += tol
			try:
				problem.solve(solver=solver)
			except cp.SolverError as e:
				problem.solve(solver='ECOS')

			if problem.status in ['optimal', 'optimal_inaccurate']:
				break
		# Return values
		return nu0.value @ probs0 + nu1.value @ probs1, nu0.value, nu1.value

	def _solve_single_instance(
		self,
		yvals: np.array,
		probs: np.array,
		x: np.array,
		i: Optional[int]=None,
		**kwargs,
	):
		"""
		Generic instance solver.

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
		i : int
			Optional index specifying which observation this 
			problem corresponds to. Used only to save the 
			adjusted variant of probs if probs is primal infeasible.

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
		probs1 = np.concatenate([probs[1, 0], probs[1, 1]], axis=0)
		vals0 = np.concatenate([yvals[0, 0], yvals[0, 1]], axis=0)
		vals1 = np.concatenate([yvals[1, 0], yvals[1, 1]], axis=0)

		# Evaluate f(w0, w1, y0, y1, x).
		# This is not the final constraint matrix
		# and needs to be adjusted later, depending on the value of lower.
		nvals = yvals.shape[-1]
		fvals_raw = np.zeros((2*nvals, 2*nvals))
		not_in_support = np.zeros((2*nvals, 2*nvals)).astype(bool)
		for w0 in [0,1]:
			w0_vals = np.arange(nvals) + w0 * nvals
			for w1 in [0,1]:
				w1_vals = np.arange(nvals) + w1 * nvals
				block_inds = np.ix_(w0_vals, w1_vals)
				block_args=dict(
					w0=w0,
					w1=w1,
					y0=yvals[0, w0],
					y1=yvals[1, w1],
					x=x,
				)
				fvals_raw[block_inds] = self._apply_ot_fn(fn=self.f, **block_args)
				if self.support_restriction is not None:
					not_in_support[block_inds] = ~self._apply_ot_fn(
						fn=self.support_restriction, **block_args
					).astype(bool)

		# Signal that the off-diagonal entries of the
		# diagonal block matrices are not binding 
		# (these are essentially just padding)
		not_binding = not_in_support.copy()
		for w in [0,1]:
			inds = np.arange(nvals) + w * nvals
			block_inds = np.ix_(inds, inds)
			not_binding[block_inds] = ((not_binding[block_inds]) | ~(np.eye(nvals).astype(int))) 

		# Compute dual variables
		nu0s = np.zeros((2, 2, nvals))
		nu1s = np.zeros((2, 2, nvals))
		c0s = np.zeros(2)
		c1s = np.zeros(2)
		fvals_adj = dict()
		objvals = dict()
		for lower in [1, 0]:
			# Get rid of constraints which conflict with a-priori
			# support restrictions
			fvals = fvals_raw.copy()
			almost_inf = (2*lower - 1) * (np.abs(fvals).max() + 1) * 1e4
			fvals[not_in_support] = almost_inf
			# Adjust constraint matrix since when w0==w1,
			# we have a linear non-OT constraint. 
			# We use a hack to encode this within the OT framework
			# so that we can use the efficient OT functions.
			for w in [0,1]:
				winds = np.arange(nvals) + w * nvals
				block_inds = np.ix_(winds, winds)
				fblock = fvals[block_inds]
				if lower:
					fblock = fblock.min(axis=1-w)
				else:
					fblock = fblock.max(axis=1-w)
				# A hack to make the non-diagonal constraints unimportant
				fblock = np.ones((nvals, nvals)) * almost_inf + np.diag(fblock - almost_inf)
				fvals[block_inds] = fblock

			# Solve
			if not lower:
				fvals = -1 * fvals
			fvals_adj[lower] = fvals

		# First attempt. These dicts map lower --> output
		objvals = dict()
		logs = dict()
		for lower in [1, 0]:
			objvals[lower], logs[lower] = ot.lp.emd2(
				a=probs0,
				b=probs1,
				M=fvals_adj[lower],
				log=True,
			)

		# Check if we the "almost_inf" constraints are binding.
		# If so, the primal problem is likely infeasible and
		# the dual problem is likely unbounded. Then we resolve 
		# both the lower and upper problem.
		max_dualval = _compute_max_dualval(logs)
		if max_dualval > 0.1 * np.abs(almost_inf):
			# Ensure primal feasibility and save results
			probs = self._ensure_primal_feasibility(
				jointprobs=probs, yvals=yvals, x=x,
			)
			probs0 = np.concatenate([probs[0, 0], probs[0, 1]], axis=0)
			probs1 = np.concatenate([probs[1, 0], probs[1, 1]], axis=0)
			if i is not None:
				self._adj_jointprobs[i] = probs

			# Resolve
			for lower in [1, 0]:
				objvals[lower], logs[lower] = ot.lp.emd2(
					a=probs0,
					b=probs1,
					M=fvals_adj[lower],
					log=True,
				)

		# Multiply dual vars and objective val by -1 for upper CI
		objvals[False] *= -1
		logs[False]['u'] *= -1
		logs[False]['v'] *= -1
		# Reset fvals for  cvxpy
		fvals_adj[False] *= -1

		# If the dual variables are still too large, solve a norm penalized
		# variant with cvxpy. This is quite slow but ideally will only
		# need to be done for a few observations.
		max_dualval = _compute_max_dualval(logs)
		if max_dualval > 0.1 * np.abs(almost_inf):
			for lower in [1, 0]:
				# Resolve with CP and l2 norm
				objvals[lower], nu0, nu1 = self._solve_single_instance_cp(
					probs0=probs0,
					probs1=probs1,
					fvals=fvals_adj[lower],
					not_binding=not_binding,
					objval_ot=objvals[lower],
					lower=lower,
				)
				logs[lower]['u'] = nu0
				logs[lower]['v'] = nu1

		# Extract output
		objvals_array = np.zeros(2)
		for lower in [0,1]:
			# Extract objective value
			objvals_array[int(1-lower)] = objvals[lower]
			# Extract dual variables
			nu0s[int(1-lower)] = np.stack([logs[lower]['u'][0:nvals], logs[lower]['u'][nvals:]], axis=0)
			nu1s[int(1-lower)] = np.stack([logs[lower]['v'][0:nvals], logs[lower]['v'][nvals:]], axis=0)
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

		return objvals_array, nus, cs

	def _ensure_primal_feasibility(
		self,
		yvals,
		jointprobs,
		x,
		solver='CLARABEL',
	):
		"""
		Parameters
		----------
		jointprobs : np.array
			(2, 2, nvals)-shaped array where probs[z, w, y] =
			P(W(z) = w, Y(w) = yvals[z, w, y] | X = x).
		yvals : np.array
			(2, 2, nvals)-shaped array where
			yvals[z, w] is the support of Y(w) | W(z) = w
		x : np.array
			value of x

		Returns
		-------
		probs_adj : np.array
			(2, 2, nvals)-shaped array which is as close to
			probs as possible such that primal feasibility 
			is satisfied.

		Notes
		-----
		In this context, "primal feasibility" means that 
		there exists some law of W(0), W(1), Y(0), Y(1) | X
		which induces the implied marginals of 
		W(0), Y(W(0)) | X and W(1), Y(W(1)) | X as stored
		by probs and yvals.

		Todo
		----
		This will be a bottleneck, specifically the LP. We should
		check if there are ways to speed it up / dual formulations
		(e.g. ensure dual boundedness instead of primal feasibility.)
		"""
		jointprobs = _clip_jointprobs(jointprobs)
		## Step 1: Test that Y(0) has a common support regardless of the value of Z
		for w in [0,1]:
			np.testing.assert_array_almost_equal(
				yvals[0][w], 
				yvals[1][w],
				decimal=8,
				err_msg=f"yvals[0][{w}] != yvals[1][{w}], making the primal infeasible and the dual unbounded."
			)

		# These should be the same by default but this allows custom user inputs.
		nvals0 = len(yvals[0][0])
		nvals1 = len(yvals[1][1])

		## Step 2: set up the LP
		## Step 2(a): define core variables
		# w0yw0_probs[w][i] is the P(W(0) = w, Y(w) = yvals[0/1][w][i])
		w0yw0_probs = [cp.Variable(nvals0, pos=True), cp.Variable(nvals1, pos=True)]
		# w1yw1_probs[w][i] is the P(W(1) = w, Y(w) = yvals[0/1][w][i])
		w1yw1_probs = [cp.Variable(nvals0, pos=True), cp.Variable(nvals1, pos=True)]
		# jp_cp[w0][w1][i, j] is the joint PMF of W(0), W(1), Y(0) (value i), Y(1) (value j)
		jp_cp = [[], []]
		for w0 in [0,1]:
			for w1 in [0,1]:
				jp_cp[w0].append(cp.Variable((nvals0, nvals1), pos=True))
		# All distributions sum to 1
		constraints = [
			cp.sum(w0yw0_probs[0]) + cp.sum(w0yw0_probs[1]) == 1,
			cp.sum(w1yw1_probs[0]) + cp.sum(w1yw1_probs[1]) == 1,
			cp.sum(jp_cp[0][0] + jp_cp[0][1] + jp_cp[1][0] + jp_cp[1][1]) == 1,
		]
		## Step 2(b): support restrictions on the joint law
		if self.support_restriction is not None:
			for w0 in [0,1]:
				for w1 in [0,1]:
					block_args = dict(
						w0=w0, w1=w1, y0=yvals[0][w0], y1=yvals[1][w1], x=x,
					)
					# Account for support restrictions
					not_in_support = ~(self._apply_ot_fn(
						fn=self.support_restriction, **block_args
					).astype(bool))
					if np.any(not_in_support):
						constraints.append(jp_cp[w0][w1][not_in_support] == 0)

		### Step 2(c): marginal laws compatible with joint law
		# Note: this is not a typical OT problem! The axis choice
		# is subtle and important.
		constraints.extend([
			### 2(c)(i): the law of Y(W(0)), W(0) matches
			# W(0)=0 implies Y(W(0)) = Y(0), sum over W(1) and Y(1)
			cp.sum(jp_cp[0][0] + jp_cp[0][1], axis=1) == w0yw0_probs[0],
			# W(0) = 1 implies Y(W(0)) = Y(1), sum over W(1) and Y(0)
			cp.sum(jp_cp[1][0] + jp_cp[1][1], axis=0) == w0yw0_probs[1],
			### 2(c)(ii): the law of Y(W(1)), W(1) matches; repeat
			# W(1) = 0 implies Y(W(1)) = Y(0), sum over W(0) and Y(1)
			cp.sum(jp_cp[0][0] + jp_cp[1][0], axis=1) == w1yw1_probs[0],
			# W(1) = 1 implies Y(W(1)) = Y(1), sum over W(0) and Y(0)
			cp.sum(jp_cp[0][1] + jp_cp[1][1], axis=0) == w1yw1_probs[1],
		])

		### 3. Objective: 1-wasserstein distance over 
		# conditional laws of Y(w) | W(z) = w
		obj = 0
		for z, mprobs in zip([0,1], [w0yw0_probs, w1yw1_probs]):
			for w in [0,1]:
				### 3(a). Ensure P(W(z)  = w | Z = z) is preserved
				pw_given_z = jointprobs[z, w].sum()
				constraints.append(cp.sum(mprobs[w]) == pw_given_z)
				### 3(b). Compute Wasserstein distances
				cprobs_cp = cp.cumsum(mprobs[w])[:-1]
				cprobs_orig = np.cumsum(jointprobs[z][w])[:-1]
				ydiffs = yvals[z][w][1:] - yvals[z][w][:-1]
				obj += cp.sum(cp.multiply(ydiffs, cp.abs(cprobs_orig - cprobs_cp)))

		### 4. Create problem and solve
		problem = cp.Problem(cp.Minimize(obj), constraints)
		try:
			problem.solve(solver=solver)
		except cp.SolverError as e:
			problem.solve(solver='ECOS', verbose=True)

		### 5. Concatenate
		new_jointprobs = np.stack(
			[
				# np.stack([w0yw0_probs[0].value, w0yw0_probs[1].value], axis=0),
				# np.stack([w1yw1_probs[0].value, w1yw1_probs[1].value], axis=0),
				np.stack(
					[
						np.sum(jp_cp[0][0].value + jp_cp[0][1].value, axis=1),
						np.sum(jp_cp[1][0].value + jp_cp[1][1].value, axis=0)
					],
					axis=0
				),
				np.stack(
					[
						np.sum(jp_cp[0][0].value + jp_cp[1][0].value, axis=1),
						np.sum(jp_cp[0][1].value + jp_cp[1][1].value, axis=0)
					],
					axis=0
				),
			],
			axis=0
		)
		### 6. Clip and ensure sums to one
		# new_jointprobs = np.clip(new_jointprobs, 0, 1)
		# for z in [0,1]:
		# 	new_jointprobs[z] /= new_jointprobs[z].sum()
		return new_jointprobs

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
			new_fvals = [[], []]
			for w0 in [0,1]:
				for w1 in [0,1]:
					block_args = dict(
						w0=w0,
						w1=w1,
						y0=new_yvals[0][w0],
						y1=new_yvals[1][w1],
						x=self.X[i],
					)
					# fvals
					new_fvals[w0].append(self._apply_ot_fn(fn=self.f, **block_args))
					# Account for support restrictions
					if self.support_restriction is not None:
						not_in_support = ~(self._apply_ot_fn(
							fn=self.support_restriction, **block_args
						).astype(bool))
					else:
						not_in_support = np.zeros(new_fvals[w0][-1].shape).astype(bool)
					if lower:
						new_fvals[w0][-1][not_in_support] = np.inf
					else:
						new_fvals[w0][-1][not_in_support] = -np.inf

			# Interpolate nus
			new_nus = [[], []]
			for z in [0,1]:
				for w in [0,1]:
					new_nus[z].append(self.interp_fn(
						x=self._yvals[i, z, w], y=orig_nus[z, w], newx=new_yvals[z][w]
					))
			### Check feasibility. Two cases.
			### Case 1: w0 == w1. Here we have vanilla linear (non-OT) constraints
			for w in [0,1]:
				if lower:
					new_fvals_block = new_fvals[w][w].min(axis=1-w)
				else:
					new_fvals_block = new_fvals[w][w].max(axis=1-w)
				deltas = new_nus[0][w] + new_nus[1][w] - new_fvals_block
				if lower:
					deltas = np.maximum(deltas, 0)
					new_nus[0][w] -= deltas / 2
					new_nus[1][w] -= deltas / 2
				else:
					deltas = np.minimum(deltas, 0)
					new_nus[0][w] -= deltas / 2
					new_nus[1][w] -= deltas / 2

			### Case 2: w0 != w1. Here we have full OT constraints.
			for w0, w1 in zip([0,1], [1,0]):
				deltas = new_nus[0][w0].reshape(-1, 1) + new_nus[1][w1].reshape(1, -1)
				deltas = deltas - new_fvals[w0][w1]
				# Figure out which axis to adjust
				if lower:
					deltas = np.maximum(deltas, 0)
					deltas0 = deltas.max(axis=1)
					dx0 = np.mean(np.maximum(deltas0, 0))
					deltas1 = deltas.max(axis=0)
					dx1 = np.mean(np.maximum(deltas1, 0))
					adj_axis = 1 if dx1 <= dx0 else 0
				else:
					deltas = np.minimum(deltas, 0)
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
			).item()
		else:
			j = np.where(self._yvals[i, zi, wi] == yi)[0][0]
			self.hatnus[1-lower, i] = nusx[zi, wi, j]

	def compute_dual_variables(
		self,
		wprobs: np.array,
		ydists: list,
		ymin: Optional[float]=None,
		ymax: Optional[float]=None,
		min_quantile: Optional[float]=None,
		interp_fn: callable=interpolation.adaptive_interpolate,
		nvals: int=100,
		ninterp: Optional[int]=None,
		verbose: bool=True,
		**kwargs
	):
		"""
		Same signature as `generic.DualBounds.compute_dual_variables`
		with the following exceptions.

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
			Minimum quantile to consider when discretizing. 
			Defaults to 1 / (2*nvals).
		"""
		# Setup
		if verbose:
			print("Estimating optimal dual variables.")
		self.interp_fn = interp_fn
		if self.discrete:
			self.nvals = len(self.support)
		else:
			self.nvals = nvals
			##TODO?
			# if self.nvals <= generic.MIN_NVALS

		# Infer ymin/ymax
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
					min_quantile=min_quantile,
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

		# Ensure that the support of Y(w) | W(0) = w and Y(w) | W(1) = w 
		# is the same. This is necessary to ensure primal feasibility.
		concat_ys = False
		for w in [0,1]:
			if not np.all(self._yvals[0][w] == self._yvals[1][w]):
				concat_ys = True
		# Concatenate
		if concat_ys:
			for w in [0,1]:
				# zero padding for probabilities
				zero_pad = np.zeros((self.n, self.nvals))
				# Create the union of the supports
				yvals_w = np.concatenate([self._yvals[0][w], self._yvals[1][w]], axis=1)
				self._yvals[0][w] = yvals_w
				self._yvals[1][w] = yvals_w
				# Pad probabilities
				self._yprobs[0][w] = np.concatenate([self._yprobs[0][w], zero_pad], axis=1)
				self._yprobs[1][w] = np.concatenate([zero_pad, self._yprobs[1][w]], axis=1)
				# argsort
				for z in [0,1]:
					self._yvals[z][w], self._yprobs[z][w] = utilities._sort_disc_dist(
						vals=self._yvals[z][w], probs=self._yprobs[z][w]
					)

			# Adjust self.nvals
			self.nvals *= 2

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
		self._adj_jointprobs = self._jointprobs.copy()

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
				i=i,
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
					**kwargs,
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
					wi=W[i],
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
		ipw_denom = Z * (pis) + (1 - Z) * (1 - pis) 
		self.ipw_summands = self.hatnus / ipw_denom
		# Compute AIPW summands
		mus = self.cs.sum(axis=2).T # (2, n)-shaped array
		deltas = self.cs[:, :, 1].T * Z + self.cs[:, :, 0].T * (1 - Z) # (2, n)-shaped array
		self.aipw_summands = (self.hatnus - deltas) / ipw_denom + mus

	def _compute_cond_means(self):
		"""
		Computes self.mu0 = E[Y(W(0)) | Z=0]
		and self.mu1 = E[Y(W(1)) | Z = 1].
		"""
		ydist_means = np.zeros((2, 2, self.n))
		for z in [0,1]:
			for w in [0,1]:
				dists_zw = self.ydists[z][w]
				if not isinstance(dists_zw, list):
					dists_zw = [dists_zw]
				ydist_means[z, w] = np.concatenate([x.mean() for x in dists_zw])

		self.mu0 = self.wprobs[:, 0] * ydist_means[0, 1]
		self.mu0 += (1 - self.wprobs[:, 0]) * ydist_means[0][0]
		self.mu1 = self.wprobs[:, 1] * ydist_means[1][1]
		self.mu1 += (1 - self.wprobs[:, 1]) * ydist_means[1][0]

	def _compute_oos_resids(self):
		"""
		Computes out-of-sample predictions of Y | Z and residuals.
		"""
		self._compute_cond_means()
		self.oos_preds = self.mu0.copy()
		self.oos_preds[self.Z == 1] = self.mu1[self.Z == 1]
		# residuals and return
		self.oos_resids = self.y - self.oos_preds
		return self.oos_resids

	def cross_fit(
		self,
		nfolds: int=5,
		suppress_warning: bool=False,
		verbose: bool=True,
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

		Returns
		-------
		ydists : list
			for z, w in [0,1], ydists[z][w] is a list of batched scipy
			distributions whose shapes sum to n. the ith distribution 
			is the out-of-sample estimate of the conditional law
			of :math:`Y_i(w) | W(z) = w, X_i`.
		"""
		# if pis not supplied: will use cross-fitting
		if self.pis is None:
			self.fit_propensity_scores(nfolds=nfolds, verbose=verbose)

		# Fit exposure model
		if self.wprobs is None:
			self.exposure_model = generic.get_default_model(
				outcome_model=self.exposure_model, 
				# the following args are ignored if exposure_model
				# inherits from dist_reg.DistReg class
				support=set([0,1]), 
				discrete=True,
				monotonicity=True, 
				how_transform='intercept',
			)
			if verbose:
				print("Cross-fitting the exposure model.")
			wout = dist_reg.cross_fit_predictions(
				W=self.Z, X=self.X, y=self.W, 
				nfolds=nfolds, 
				model=self.exposure_model,
				verbose=verbose,
				probs_only=True,
			)
			wprobs, self.exposure_model_fits, _ = wout
			self.wprobs = np.stack(wprobs, axis=1)
		elif not suppress_warning:
			warnings.warn(generic.CROSSFIT_WARNING.replace("y0_dists/y1_dists", "wprobs"))

		# Fit outcome model
		if self.ydists is None:
			# Note: this returns the existing model
			# if an existing model is provided
			self.outcome_model = generic.get_default_model(
				discrete=self.discrete, 
				support=self.support,
				outcome_model=self.outcome_model,
				**self.model_kwargs
			)
			if verbose:
				print("Cross-fitting the outcome model.")
			y_out = dist_reg.cross_fit_predictions(
				W=self.W, X=self.X, y=self.y, Z=self.Z, 
				nfolds=nfolds, 
				model=self.outcome_model,
				verbose=verbose,
			)
			self.ydists, self.model_fits, self.oos_dist_preds = y_out
		elif not suppress_warning:
			warnings.warn(generic.CROSSFIT_WARNING)

		return self.ydists


	def fit(
		self,
		nfolds: int = 5,
		aipw: bool = True,
		alpha: float = 0.05,
		wprobs: Optional[list]=None,
		ydists: Optional[list]=None,
		verbose: bool = True,
		suppress_warning: bool = False,
		**solve_kwargs,
	):
		# Fit model of W | X and Y | X if not provided
		self.ydists, self.wprobs = ydists, wprobs
		self.cross_fit(
			nfolds=nfolds, suppress_warning=suppress_warning, verbose=verbose,
		)

		# compute dual variables
		self.compute_dual_variables(
			ydists=self.ydists,
			wprobs=self.wprobs,
			verbose=verbose,
			**solve_kwargs,
		)
		# compute dual bounds
		self.alpha = alpha
		self._compute_final_bounds(aipw=aipw, alpha=alpha)
		return self

	def eval_treatment_model():
		return dist_reg._evaluate_model_predictions(
			y=self.Z, haty=self.pis
		)

	def eval_exposure_model():
		"""
		Thinly wraps dist_reg._evaluate_model_predictions.

		Returns
		-------
		sumstats : pd.DataFrame
			DataFrame summarizing goodness-of-fit metrics for
			the cross-fit exposure scores.
		"""
		return dist_reg._evaluate_model_predictions(
			y=self.W, 
			haty=self.wprobs[(np.arange(self.n), self.Z.astype(int))]
		)


	def summary(self, minval=-np.inf, maxval=np.inf):
		print("___________________Inference_____________________")
		print(self.results(minval=minval, maxval=maxval))
		print()
		print("_________________Outcome model___________________")
		print(self.eval_outcome_model())
		print()
		print("_________________Exposure model__________________")
		print(self.eval_exposure_model())
		print()
		print("_________________Treatment model_________________")
		print(self.eval_treatment_model())
		print(sumstats)
		print()