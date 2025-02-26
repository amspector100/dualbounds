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
				covariates=data['X'],
				outcome=data['y'],
				treatment=data['W'],
				propensities=data['pis'],
			)
			ests = vdb.fit(nfolds=3).estimates

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
				covariates=data['X'],
				outcome=data['y'],
				treatment=data['W'],
				propensities=data['pis'],
			)
			vdb.fit(nfolds=3).summary()


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
			covariates=data['X'], outcome=data['y'], treatment=data['W'], propensities=data['pis'],
		)
		vdb.fit(nfolds=3).summary()


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
				covariates=data['X'], outcome=data['y'], treatment=data['W'], propensities=data['pis'],
			)
			est_oracle = vdb_oracle.fit(
				y0_dists=data['y0_dists'], y1_dists=data['y1_dists'], 
				suppress_warning=True,
			).estimates[0]
			## Actual
			vdb = db.varcate.VarCATEDualBounds(
				covariates=data['X'], outcome=data['y'], treatment=data['W'], propensities=data['pis'],
				outcome_model=db.dist_reg.CtsDistReg(eps_dist='gaussian')
			)
			est_actual = vdb.fit(nfolds=3).estimates[0]
			for est, name in zip([est_oracle, est_actual], ['oracle', 'est']):
				np.testing.assert_array_almost_equal(
					est,
					expected,
					decimal=2,
					err_msg=f"VarCATE {name} is inaccurate (r2={r2}, eps_dist={eps_dist})"
				)

	def test_varcate_cluster_bootstrap(self):
		data = db.gen_data.gen_regression_data(n=300, p=10, sparsity=0.8, sample_seed=123)
		# Fit multiple varcate models
		vdbs = []
		outcome_models = ['lasso', db.dist_reg.CtsDistReg('knn', n_neighbors=50)]
		for outcome_model in outcome_models:
			vdb = db.varcate.VarCATEDualBounds(
				outcome=data['y'],
				treatment=data['W'],
				covariates=data['X'],
				propensities=data['pis'],
				outcome_model=outcome_model,
			).fit(verbose=False)
			vdbs.append(vdb)
		# Combine
		vdb_results = db.varcate.varcate_cluster_bootstrap(vdbs, verbose=False, alpha=0.05, B=1000)
		# Check estimate
		expected_estimate = np.max([x.estimates[0] for x in vdbs])
		estimate = vdb_results.loc['Estimate', 'Lower'].item()
		self.assertTrue(
			estimate==expected_estimate,
			msg=f"Combined estimate={estimate} is different than the max estimate {expected_estimate}"
		)
		# Check CI
		expected_ci = np.max([x.cis[0] for x in vdbs])
		ci = vdb_results.loc['Conf. Int.', 'Lower'].item()
		ci_tol = vdbs[np.argmin([x.cis[0] for x in vdbs])].ses[0]
		self.assertTrue(
			ci >= expected_ci - 2 * ci_tol,
			msg=f"Combined CI={ci} <= max lower_ci={expected_ci} plus tolerance {ci_tol}"
		)
		self.assertTrue(
			ci <= expected_ci,
			msg=f"Combined CI={ci} is somehow sharper than max lower_ci={expected_ci}"
		)

		# for expected, output, tol, decimal, name in zip(
		# 	[expected_estimate, expected_ci], 
		# 	[estimate, ci],
		# 	[0, ci_tol],
		# ):

	def test_calibrated_varcate_db(self):
		outcome_models = ['ridge', 'knn']
		for eps_dist, outcome_models in zip(
			['gaussian', 'bernoulli'],
			[['ridge', 'knn'], ['knn', db.dist_reg.BinaryDistReg('rf', min_samples_leaf=50, n_estimators=3)]],
		):
			data = db.gen_data.gen_regression_data(n=300, p=5, sample_seed=123, eps_dist=eps_dist)
			# fit
			vdb = db.varcate.CalibratedVarCATEDualBounds(
				outcome=data['y'],
				treatment=data['W'],
				covariates=data['X'],
				propensities=data['pis'],
				outcome_model=outcome_models,
			).fit(nfolds=3, verbose=False).summary()

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