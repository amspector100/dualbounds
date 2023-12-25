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
				X=data['X'], y=data['y'], W=data['W'], pis=data['pis'],
			)
			ests = vdb.compute_dual_bounds(nfolds=3)['estimates']

			# test accuracy
			np.testing.assert_array_almost_equal(
				np.array([lower, upper]),
				ests,
				decimal=1,
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
				X=data['X'], y=data['y'], W=data['W'], pis=data['pis'],
			)
			output = vdb.compute_dual_bounds(nfolds=3)

class TestVarCATE(unittest.TestCase):

	def test_varcate_delta_method_se(self):
		""" Tests that we correctly estimate the SE. """
		context._run_se_computation_test(
			dim=6,
			f=db.varcate._moments2varcate,
			arg_names=[
				'shxy1', 'shxy0', 'shx', 'sy1', 'sy0', 'shx2'
			],
			testname='Var(CATE)',
			se_function=db.varcate.varcate_delta_method_se,
		)

	def test_no_discrete_error(self):
		"""
		Just tests that the discrete case runs without errors.
		The LogisticCV solver is slow so we don't check consistency.
		"""
		data = db.gen_data.gen_lee_bound_data(
			n=200, p=5, eps_dist='bernoulli',
		)
		vdb = db.varcate.VarCATEDualBounds(
			X=data['X'], y=data['y'], W=data['W'], pis=data['pis'],
		)
		vdb.compute_dual_bounds(nfolds=3)


	def test_varcate_consistency(self):
		for eps_dist, r2 in zip(
			['gaussian', 'expon'], [0.9, 0.5, 0.0]
		):
			data = db.gen_data.gen_lee_bound_data(
				n=300000, p=5, r2=r2, tau=2, dgp_seed=1, sample_seed=1,
				eps_dist=eps_dist,
				interactions=True,
			)
			expected = data['cates'].std()**2
			## Oracle
			vdb_oracle = db.varcate.VarCATEDualBounds(
				X=data['X'], y=data['y'], W=data['W'], pis=data['pis'],
			)
			est_oracle = vdb_oracle.compute_dual_bounds(
				y0_dists=data['y0_dists'], y1_dists=data['y1_dists'], 
				suppress_warning=True,
			)['estimate']
			## Actual
			vdb = db.varcate.VarCATEDualBounds(
				X=data['X'], y=data['y'], W=data['W'], pis=data['pis'],
			)
			est_actual = vdb.compute_dual_bounds(nfolds=3)['estimate']
			for est, name in zip([est_oracle, est_actual], ['oracle', 'est']):
				np.testing.assert_array_almost_equal(
					est,
					expected,
					decimal=2,
					err_msg=f"VarCATE {name} is inaccurate (r2={r2}, eps_dist={eps_dist})"
				)


if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestVarITE(), TestVarCATE()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()