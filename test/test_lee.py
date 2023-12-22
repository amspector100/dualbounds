import time
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest
import os
import sys
try:
	from . import context
	from .context import dualbounds as db
# For profiling
except ImportError:
	import context
	from context import dualbounds as db

from dualbounds import lee, utilities

def old_conditional_lee_bound(
	ps0,
	ps1,
	y0_vals,
	py0_given_s0,
	y1_vals,
	py1_given_s1,
	lower=True,
):
	"""
	Depreciated, used to test the new version.
	
	Parameters
	----------
	ps0 : np.array
		2-length array where ps0[i] = P(S(0) = i | X)
	ps1 : np.array
		2-length array where ps1[i] = P(S(1) = i | X)
	y0_vals : np.array
		nvals-length array of values y1 can take.
	py0_given_s0 : np.array
		(nvals)-length array where
		py0_given_s0[i] = P(Y(0) = yvals[i] | S(0) = 1)
	y1_vals : np.array
		nvals-length array of values y1 can take.
	py1_given_s1 : np.array
		(nvals)-length array where
		py1_given_s1[i] = P(Y(1) = yvals[i] | S(1) = 1)
	"""
	if np.any(y1_vals != np.sort(y1_vals)):
		raise ValueError(f"y1_vals must be sorted, not {y1_vals}")

	if not lower:
		return -1 * old_conditional_lee_bound(
			ps0=ps0, 
			ps1=ps1,
			y0_vals=-1 * y0_vals,
			py0_given_s0=py0_given_s0,
			y1_vals=np.flip(-1*y1_vals),
			py1_given_s1=np.flip(py1_given_s1),
			lower=True
		)

	# verify monotonicity
	if ps0[1] > ps1[1]: 
		raise ValueError(f"Monotonicity is violated, ps0={ps0}, ps1={ps1}") 

	# always takers share 
	p0 = ps0[1] / ps1[1]

	# compute E[Y(1) | Y(1) >= Q(p0), S(1)=1]
	# where Q is the quantile fn of Y(1) | S(1) = 1.
	cum_cond_probs = np.cumsum(py1_given_s1)
	cum_cond_probs /= py1_given_s1.sum() # just in case
	k = np.argmin(cum_cond_probs <= p0)
	# shave off probability from the last cell bc of discreteness
	gap = cum_cond_probs[k] / p0 - 1
	cond_probs = py1_given_s1[0:(k+1)] / p0
	cond_probs[-1] -= gap
	if np.abs(np.sum(cond_probs) - 1) > 1e-8: 
		raise ValueError(f"Cond probs sum to {np.sum(cond_probs)} != 1")
	term1 = y1_vals[0:(k+1)] @ cond_probs
	return term1 - y0_vals @ py0_given_s0

def slow_interpolation(x, y, newx):
	"""
	same as lee.interpolate but slower and more readable.
	"""
	if not utilities.haslength(newx):
		newx = np.array([newx])
	# compute differences
	n = len(x); m = len(newx)
	diffs = x.reshape(n, 1) - newx.reshape(1, m)
	# identify closest points to newx
	inds = np.argsort(np.abs(diffs), axis=0)[0:2]
	i1 = inds[0]; x1 = x[i1]; y1 = y[i1]
	i2 = inds[1]; x2 = x[i2]; y2 = y[i2]
	# interpolate
	dx = x2 - x1
	dy = y2 - y1
	haty = y1 + dy / dx * (newx - x1)
	return haty

def compute_cvar_samples(dists, n, alpha, lower=True, reps=100000):
	"""
	Batched computation of 
	E[Y | Y <= Q_{alpha}(Y)] from Y ~ dists if lower = True.
	Used to test that the other cvar implementation (which is more
	efficient) is accurate.
	"""
	if isinstance(alpha, float) or isinstance(alpha, int):
		alpha = alpha * np.ones(n)
	# sample
	samples = dists.rvs(size=(reps, n))
	# Compute estimates from samples
	cvar_est = np.zeros(n)
	for i in range(n):
		samplesi = samples[:, i]
		# estimated quantile
		hatq = np.quantile(samplesi, alpha[i])
		if lower:
			cvar_est[i] = samplesi[samplesi <= hatq].mean()
		else:
			cvar_est[i] = samplesi[samplesi >= hatq].mean()
	# Return
	return cvar_est

class TestHelpers(unittest.TestCase):
	"""
	tests
	"""
	def test_cvar(self):
		# create 
		np.random.seed(123)
		n = 20
		sigmas = np.random.uniform(size=n)
		mu = np.random.randn(n)
		ydists = stats.expon(loc=mu, scale=sigmas)
		# test
		for lower in [True, False]:
			avec = np.random.uniform(0.1, 0.9, size=n)
			for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, avec]:
				cvar_est = lee.compute_cvar(
					ydists, n=n, alpha=alpha, lower=lower
				)
				cvar_est_samp = compute_cvar_samples(
					ydists, n=n, alpha=alpha, lower=lower
				)
				err_ratio = np.sum((cvar_est - cvar_est_samp)**2) / np.sum(cvar_est**2)
				self.assertTrue(
					err_ratio  <= 1e-3,
					f"cvar_est={cvar_est} != {cvar_est_samp}."
				)

	def test_analytical_lee_bnds(self):
		# Create dgp
		n = 40
		nvals = 20
		s0_probs = np.random.uniform(0, 0.9, size=n)
		s1_probs = s0_probs + 0.1
		# create vals
		y0_vals = np.sort(np.random.randn(n, nvals), axis=1)
		y1_vals = np.sort(np.random.randn(n, nvals), axis=1)
		y0_probs = np.random.uniform(size=(n, nvals))
		y0_probs /= y0_probs.sum(axis=1).reshape(-1, 1)
		y1_probs = np.random.uniform(size=(n, nvals))
		y1_probs /= y1_probs.sum(axis=1).reshape(-1, 1)
		# compute lee bounds---old variant
		lbounds1 = np.zeros(n)
		ubounds1 = lbounds1.copy()
		for i in range(n):
			ps0 = np.array([1 - s0_probs[i], s0_probs[i]])
			ps1 = np.array([1 - s1_probs[i], s1_probs[i]])
			args = dict(
				ps0=ps0, ps1=ps1,
				y0_vals=y0_vals[i],
				y1_vals=y1_vals[i],
				py0_given_s0=y0_probs[i],
				py1_given_s1=y1_probs[i],
			)
			lbounds1[i] = old_conditional_lee_bound(
				**args, lower=True
			)
			ubounds1[i] = old_conditional_lee_bound(
				**args, lower=False
			)
		abounds1 = np.stack([lbounds1, ubounds1], axis=0)

		# new variant
		new_args = dict(
			s0_probs=s0_probs, s1_probs=s1_probs,
			y0_probs=y0_probs, y1_probs=y1_probs,
			y1_vals=y1_vals, y0_vals=y0_vals,
		)
		abounds = lee.compute_analytical_lee_bound(**new_args, m=10000)

		# assert equality
		np.testing.assert_array_almost_equal(
			abounds, abounds1, 
			decimal=3,
			err_msg="discrete analytical vs. cts analytical bounds do not match"
		)

	# def test_interpolation(self):
	# 	np.random.seed(1234)
	# 	n = 100
	# 	m = 30
	# 	x = np.sort(np.random.randn(n))
	# 	y = np.random.randn(n) + x
	# 	newx = np.random.randn(m)
	# 	expected = slow_interpolation(
	# 		x=x, y=y, newx=newx,
	# 	)
	# 	result = lee.interpolate(
	# 		x=x, y=y, newx=newx,
	# 	)
	# 	np.testing.assert_array_almost_equal(
	# 		expected, result, decimal=5, err_msg="Interp. fns do not agree"
	# 	)

class TestDualLeeBounds(unittest.TestCase):

	def test_dual_lee_lp_solver(self):
		np.random.seed(1234)
		# Create dgp
		n = 11
		nvals = 1000
		s0_probs = np.random.uniform(0, 0.9, size=n)
		s1_probs = s0_probs + 0.1
		# create dists
		y0_dists = stats.norm(
			loc=0.3*np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n),
		)
		y1_dists = stats.expon(
			loc=0.3*np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n),
		)
		y = y1_dists.rvs() # placeholder, not important
		# args
		args = dict(
			s0_probs=s0_probs, s1_probs=s1_probs, y1_dists=y1_dists,
		)
		# analytical solution
		abounds = lee.compute_analytical_lee_bound(**args, y0_dists=y0_dists)
		# analytical bounds based on linear programming
		ldb = lee.LeeDualBounds(
			y=y, X=None, W=None, S=np.random.binomial(1, 0.5, n),
		)
		ldb.compute_dual_variables(
			**args, ymin=-10, ymax=10, nvals=nvals
		)
		lp_bounds = ldb.objvals - ldb.dxs - y0_dists.mean() * s0_probs
		lp_bounds = lp_bounds / s0_probs
		expected = abounds
		np.testing.assert_array_almost_equal(
			lp_bounds,
			expected,
			decimal=2,
			err_msg="Lee LP bounds do not agree with analytical bounds",
		)
		# Test that dx == 0
		np.testing.assert_array_almost_equal(
			ldb.dxs,
			np.zeros(ldb.dxs.shape),
			decimal=5,
			err_msg=f"Lee bound dxs are too large"
		)

	def test_dual_lee_ipw_bounds(self):
		np.random.seed(1234)
		# Create dgp
		n = 50
		nvals = 100
		s0_probs = np.random.uniform(0, 0.9, size=n)
		s1_probs = s0_probs + 0.1
		# create dists
		for eps_dist in ['bernoulli', 'expon']:
			y0_dists = utilities.parse_dist(
				eps_dist, loc=np.random.randn(n),
				scale=np.random.uniform(0.1, 1, size=n),
			)
			y1_dists = utilities.parse_dist(
				eps_dist, loc=np.random.randn(n), 
				scale=np.random.uniform(0.1, 1, size=n),
			)
			if eps_dist == 'bernoulli':
				vals = np.zeros((n, 2)); vals[:, 1] += 1
				y1_probs = y1_dists.mean()
				y1_dists_input = utilities.BatchedCategorical(
					vals=vals, probs=np.stack([1-y1_probs, y1_probs], axis=1)
				)
			else:
				y1_dists_input = y1_dists

			args = dict(
				s0_probs=s0_probs, s1_probs=s1_probs, y1_dists=y1_dists_input,
			)
			# sample data. a trick to make the test run faster
			# is to sample many values of y per dual variable
			N = 2000 # num samples per value of x
			pis = np.ones(n) / 2
			W = np.random.binomial(1, pis, size=(N,n))
			Y0 = y0_dists.rvs(size=(N,n))
			Y1 = y1_dists.rvs(size=(N,n))
			Y = Y0.copy(); Y[W == 1] = Y1[W == 1]
			S0 = np.random.binomial(1, s0_probs, size=(N,n))
			S1 = np.random.binomial(1, s1_probs, size=(N,n))
			S = S0.copy(); S[W == 1] = S1[W == 1]

			# analytical solution
			abounds = lee.compute_analytical_lee_bound(**args, y0_dists=y0_dists)
			# convert to E[Y(1) S(0)]
			expected = np.mean(abounds * s0_probs + s0_probs * y0_dists.mean(), axis=1)
			# compute dual variables
			ldb = lee.LeeDualBounds(
				y=Y[0], S=S[0], W=W[0], X=None, pis=pis
			)
			ldb.y0_dists = y0_dists
			ldb.s0_probs = s0_probs
			ldb.compute_dual_variables(**args, ymin=-5, ymax=5)
			# compute IPW/AIPW summands
			ipws = []
			aipws = []
			for i in range(N):
				# use same dual variables, new data
				ldb.y = Y[i]
				ldb.S = S[i]
				ldb.W = W[i]
				ldb._compute_realized_dual_variables(y=Y[i], S=S[i])
				ldb.compute_ipw_summands()
				ipws.append(ldb.ipw_summands)
				aipws.append(ldb.aipw_summands)
			objval = ldb.objvals.mean(axis=1)
			ipw_ests = np.concatenate(ipws, axis=1).mean(axis=1)
			aipw_ests = np.concatenate(aipws, axis=1).mean(axis=1)
			for method, ests in zip(['ipw', 'aipw', 'LP'], [ipw_ests, aipw_ests, objval]):
				np.testing.assert_array_almost_equal(
					ests,
					expected,
					decimal=2,
					err_msg=f"{method} bounds do not agree with analytical bounds",
				)

if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestHelpers(), TestDualLeeBounds()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
