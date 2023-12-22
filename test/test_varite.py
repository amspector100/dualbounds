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

from dualbounds import varite, utilities, gen_data


class TestVarITE(unittest.TestCase):

	def test_varite_consistency(self):
		""" This tests consistent Var(ITE) estimation. """
		n = 3000
		for tau, r2 in zip([0, 5, -2], [0.9, 0.0, 0.99]):
			data = gen_data.gen_regression_data(
				n=n, p=5, r2=r2, tau=tau, eps_dist='bernoulli', sample_seed=123,
			)
			# Compute true upper bound
			lower, upper = varite.compute_analytical_varite_bound(
				n=n, 
				y0_dists=data['y0_dists'], y1_dists=data['y1_dists'],
				reps=10,
			)

			# Compute vdb
			vdb = db.varite.VarITEDualBounds(
				X=data['X'], y=data['Y'], W=data['W'], pis=data['pis'],
			)
			ests, bounds = vdb.compute_dual_bounds(nfolds=3)

			# test accuracy
			np.testing.assert_array_almost_equal(
				np.array([lower, upper]),
				ests,
				decimal=1.5,
				err_msg=f"Var(ITE) bounds are not consistent, n={n}"
			)

	def test_varite_delta_method_se(self):
		""" Tests that we correctly estimate the SE. """
		context._run_se_computation_test(
			dim=3,
			f=lambda x, y, z: x - (y-z)**2,
			arg_names=['sbetas', 'skappa1s', 'skappa0s'],
			testname='Var(ITE)',
			se_function=db.varite.varite_delta_method_se,
		)

	def test_no_error(self):
		""" This tests if the code runs without error for continuous Y. """
		for eps_dist in ['gaussian', 'bernoulli']:
			data = gen_data.gen_regression_data(
				n=200, p=10, eps_dist=eps_dist, sample_seed=123
			)
			vdb = db.varite.VarITEDualBounds(
				X=data['X'], y=data['Y'], W=data['W'], pis=data['pis'],
			)
			ests, bounds = vdb.compute_dual_bounds(nfolds=3)

if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestVarITE()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()