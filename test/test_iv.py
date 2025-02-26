import time
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import unittest
import pytest
import os
import sys
import sklearn.ensemble
from sklearn import linear_model as lm
try:
	from . import context
	from .context import dualbounds as db
# For profiling
except ImportError:
	import context
	from context import dualbounds as db

from dualbounds import generic, iv, utilities, gen_data
from dualbounds.utilities import parse_dist, BatchedCategorical


def _balke_pearl_bounds(wprobs, ydists):
	"""
	wprobs : np.array
		wprobs[i, z] = P(W_i(z) = 1)
	ydists : list
		ydists[z][w] is a n-shaped scipy distribution
		representing the distributions of Y_i(w) | W_i(z) = w, X_i.
		These must be Bernoulli distributions.
	"""
	# Convert to probabilities
	yprobs = [[], []]
	for z in [0,1]:
		for w in [0,1]:
			yprobs_zw = ydists[z][w].mean()
			if np.any(yprobs_zw < 0) or np.any(yprobs_zw > 1):
				raise ValueError("y must be binary to apply balke_pearl bounds")
			yprobs[z].append(yprobs_zw)

	def compute_pi(y, w, z):
		"""
		Computes vector of probabilities of P(Y=y, W=w | Z = z)
		This is pi in https://arxiv.org/pdf/2301.12106.
		"""

		## 1. vector of probabilities P(W = w) 
		wp = (1-w) * (1 - wprobs[:, z]) + w * wprobs[:, z]

		## 2. joint probabilities
		if y == 1:
			return wp * yprobs[z][w]
		else:
			return wp * (1 - yprobs[z][w])

	# Compute balke pearl bounds
	# This is ugly, but I'm not sure of a better solution than hard-coding
	### (a) minimum
	thetal1 = compute_pi(1, 1, 1) + compute_pi(0, 0, 0) - 1
	thetal2 = compute_pi(1, 1, 0) + compute_pi(0, 0, 1) - 1
	thetal3 = - compute_pi(y=0, w=1, z=1) - compute_pi(y=1, w=0, z=1)
	thetal4 = - compute_pi(y=0, w=1, z=0) - compute_pi(y=1, w=0, z=0)
	thetal5 = compute_pi(1, 1, 0) - compute_pi(1, 1, 1) - compute_pi(1, 0, 1) - compute_pi(0, 1, 0) - compute_pi(1, 0, 0)
	thetal6 = compute_pi(1, 1, 1) - compute_pi(1, 1, 0) - compute_pi(1, 0, 0) - compute_pi(0, 1, 1) - compute_pi(1, 0, 1)
	thetal7 = compute_pi(0, 0, 1) - compute_pi(0, 1, 1) - compute_pi(1, 0, 1) - compute_pi(0, 1, 0) - compute_pi(0, 0, 0)
	thetal8 = compute_pi(0, 0, 0) - compute_pi(0, 1, 0) - compute_pi(1, 0, 0) - compute_pi(0, 1, 1) - compute_pi(0, 0, 1)
	gammal = np.stack(
		[thetal1, thetal2, thetal3, thetal4, thetal5, thetal6, thetal7, thetal8],
		axis=0
	).max(axis=0)

	### (b) maximum
	thetau1 = 1 - compute_pi(0, 1, 1) - compute_pi(1, 0, 0)
	thetau2 = 1 - compute_pi(0, 1, 0) - compute_pi(1, 0, 1)
	thetau3 = compute_pi(1, 1, 1) + compute_pi(0, 0, 1)
	thetau4 = compute_pi(1, 1, 0) + compute_pi(0, 0, 0)
	thetau5 = - compute_pi(0, 1, 0) + compute_pi(0, 1, 1) + compute_pi(0, 0, 1) + compute_pi(1, 1, 0) + compute_pi(0, 0, 0)
	thetau6 = - compute_pi(0, 1, 1) + compute_pi(1, 1, 1) + compute_pi(0, 0, 1) + compute_pi(0, 1, 0) + compute_pi(0, 0, 0)
	thetau7 = - compute_pi(1, 0, 1) + compute_pi(1, 1, 1) + compute_pi(0, 0, 1) + compute_pi(1, 1, 0) + compute_pi(1, 0, 0)
	thetau8 = - compute_pi(1, 0, 0) + compute_pi(1, 1, 0) + compute_pi(0, 0, 0) + compute_pi(1, 1, 1) + compute_pi(1, 0, 1)
	gammau = np.stack(
		[thetau1, thetau2, thetau3, thetau4, thetau5, thetau6, thetau7, thetau8],
		axis=0
	).min(axis=0)
	return gammal, gammau

class TestDualIVBounds(unittest.TestCase):
	
	def test_balke_pearl_reduction(self):
		"""
		Test that dualIV bounds reduce to Balke-Pearl for binary y.
		""" 
		n = 100
		data = db.gen_data.gen_iv_data(
			n=n, p=2, tau=0.5, eps_dist='bernoulli', dgp_seed=1, sample_seed=1,
		)
		
		# Balke-Pearl bounds
		bp_lower, bp_upper = _balke_pearl_bounds(
			wprobs=data['wprobs'], ydists=data['ydists']
		)
		bp_bounds = np.stack([bp_lower, bp_upper], axis=0)
		
		# Plug-in bounds
		dbiv = iv.DualIVBounds(
			f=lambda w0, w1, y0, y1, x: y1 - y0,
			outcome=data['y'],
			exposure=data['W'],
			instrument=data['Z'],
			propensities=data['pis'],
			covariates=None,
			discrete=True,
			support=set([0,1]),
			suppress_iv_warning=True,
		)
		ydists_cat = [[np.nan, np.nan], [np.nan, np.nan]]
		for z in [0,1]:
			for w in [0,1]:
				ydists_cat[z][w] = BatchedCategorical.from_scipy(data['ydists'][z][w])

		dbiv.compute_dual_variables(
			wprobs=data['wprobs'],
			ydists=ydists_cat,
			verbose=True,
		)
		np.testing.assert_array_almost_equal(
			dbiv.objvals,
			bp_bounds,
			decimal=5,
			err_msg="For binary y, DualIVBounds objvals do not match Balke-Pearl bounds"
		)
		# Test relationship with CATE
		self.assertTrue(
			np.all(dbiv.objvals[0] <= data['cates']),
			"lower objvals are sometimes larger than the CATE"
		)
		self.assertTrue(
			np.all(dbiv.objvals[1] >= data['cates']),
			"upper objvals are sometimes smaller than the CATE"
		)

		# Test that the AIPW process works. First, sample new data
		reps = 2000
		newZ = np.random.binomial(1, data['pis'], size=(reps, n))
		newW = np.random.binomial(1, data['wprobs'][(np.arange(n), newZ)])
		newY = np.zeros((reps, n)) - 10
		for z in [0,1]:
			for w in [0,1]:
				ysamp = data['ydists'][z][w].rvs(size=(reps, n))
				newY[(newW == w) & (newZ == z)] = ysamp[(newW == w) & (newZ == z)]

		# Now resample 
		aipw_estimators = np.zeros((reps, 2))
		ipw_estimators = np.zeros((reps, 2))
		for r in range(reps):
			dbiv._compute_realized_dual_variables(
				y=newY[r], W=newW[r], Z=newZ[r],
			)
			dbiv.Z = newZ[r]
			dbiv.W = newW[r]
			dbiv.y = newY[r]
			dbiv._compute_ipw_summands()
			dbiv._compute_final_bounds(aipw=False)
			aipw_estimators[r] = dbiv.estimates
			dbiv._compute_final_bounds(aipw=True)
			ipw_estimators[r] = dbiv.estimates
		# Test equality
		for estimators, name in zip([ipw_estimators, aipw_estimators], ['ipw', 'aipw']):
			np.testing.assert_array_almost_equal(
				estimators.mean(axis=0),
				bp_bounds.mean(axis=1),
				decimal=2,
				err_msg=f"DualIVBounds {name} estimators do not match Balke-Pearl bounds"
			)

	def test_late_reduction(self):
		"""
		Check that DualIVBounds under monotonicity can estimate the LATE.
		"""
		n = 20
		data = db.gen_data.gen_iv_data(
			n=n, 
			p=2,
			dgp_seed=1,
			sample_seed=1,
		)
		# DBIV with monotonicity
		dbiv = iv.DualIVBounds(
			f=lambda w0, w1, y0, y1, x: (y1 - y0) * w1 * (1-w0),
			outcome=data['y'],
			exposure=data['W'],
			instrument=data['Z'],
			propensities=data['pis'],
			covariates=None,
			support_restriction=lambda w0, w1, y0, y1, x: w0 <= w1,
			suppress_iv_warning=True,
		)
		dbiv.compute_dual_variables(
			wprobs=data['wprobs'],
			ydists=data['ydists'],
			verbose=True,
		)
		# For simplicity, just check numerator (later will use DeltaDualIVBounds)
		late_numerators = 0
		for z in [0,1]:
			for w in [0,1]:
				wp = w * data['wprobs'][:, z] + (1 - w) * (1 - data['wprobs'][:, z])
				late_numerators += (2*z - 1) * data['ydists'][z][w].mean() * wp
		# Check that we get the right results
		for lower, name in zip([1, 0], ['lower', 'upper']):
			np.testing.assert_array_almost_equal(
				dbiv.objvals[1-lower],
				late_numerators,
				decimal=5,
				err_msg=f"{name} DualIV objective values do not match LATE numerator under monotonicity"
			)

	# def test_consistency(self):
	# 	"""
	# 	Check that the final bounds are similar to the oracle bounds.
	# 	"""
	# 	pass



if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestDualIVBounds()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
