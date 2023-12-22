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

from dualbounds import generic, utilities, gen_data
from dualbounds.utilities import parse_dist, _convert_to_cat


class TestGenericDualBounds(unittest.TestCase):

	def test_var_ite_oracle(self):
		# create dists
		np.random.seed(123)
		n = 1
		for dist0, dist1, discrete, support in zip(
			['gaussian', 'expon', 'tdist', 'bernoulli'],
			['gaussian', 'gaussian', 'expon', 'bernoulli'],
			[False, False, False, True],
			[None, None, None, np.array([0,1])],
		):
			y0_dists = parse_dist(
				dist0, loc=np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n)
			)
			y1_dists = parse_dist(
				dist1, loc=np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n)
			)
			y0_dists_input = _convert_to_cat(y0_dists, n=n) if discrete else y0_dists
			y1_dists_input = _convert_to_cat(y1_dists, n=n) if discrete else y1_dists
			# analytically compute lower/upper bounds on var(ITE)
			reps = 100000
			U = np.random.uniform(size=reps)
			y1 = y1_dists.ppf(U); y0l = y0_dists.ppf(U); y0u = y0_dists.ppf(1-U)
			lower = np.std(y1 - y0l)**2
			upper = np.std(y1 - y0u)**2
			mu2 = (y1_dists.mean() - y0_dists.mean())**2
			# create dualbound.
			# the input data is a dummy variable that does nothing
			nvals = 2000
			vdb = db.generic.DualBounds(
				f=lambda y0, y1, x: (y0-y1)**2,
				X=np.zeros(1),
				W=np.zeros(1),
				pis=np.ones(1) * 1/2,
				y=np.zeros(1),
				discrete=discrete,
				support=support,
			)
			vdb.compute_dual_variables(
				y0_dists=y0_dists_input, 
				y1_dists=y1_dists_input, 
				y0_min=np.min(y0l),
				y0_max=np.max(y0l),
				y1_min=np.min(y1), 
				y1_max=np.max(y1),
				nvals0=nvals,
				nvals1=nvals,
			)
			# check if objvals match with lower, upper
			np.testing.assert_array_almost_equal(
				np.array([lower, upper]),
				vdb.objvals[:, 0] - mu2,
				decimal=1.5,
				err_msg=f"LP bounds for {dist0, dist1} do not agree with analytical bounds on Var(ITE)",
			)

	def test_dx_size(self):
		# Create fake heavy-tailed data
		np.random.seed(123)
		n = 300; nvals = 50; grid_size = 3 * nvals
		y0_dists = parse_dist(
			'tdist', loc=np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n)
		)
		y1_dists = parse_dist(
			'expon', loc=np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n)
		)
		y0 = y0_dists.rvs()
		y1 = y1_dists.rvs()
		pis = np.ones(n) / 2
		W = np.random.binomial(1, pis, size=n)
		Y = y0.copy(); Y[W == 1] = y1[W == 1].copy()
		X = np.random.randn(n, 1)
		# Try with many different fs
		fs = [
			lambda y0, y1, x: (y0-y1)**2,
			lambda y0, y1, x: y0 <= y1,
			lambda y0, y1, x: (y0 <= 0) * (y1 <= 0),
		]
		# Test the size of dxs is small on average
		for f in fs:
			gdb = db.generic.DualBounds(
				f=f,
				X=X,
				W=W,
				pis=pis,
				y=Y,
			)
			gdb.compute_dual_variables(
				y0_dists=y0_dists, 
				y1_dists=y1_dists,
				nvals0=nvals,
				nvals1=nvals,
			)
			np.testing.assert_array_almost_equal(
				gdb.dxs.mean(axis=1),
				np.zeros(2),
				decimal=1.5,
				err_msg=f"For f={f}, dxs are too large."
			)

if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestGenericDualBounds()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
