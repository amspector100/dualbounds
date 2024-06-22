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

from dualbounds import bootstrap, utilities, gen_data

class TestMultiplierBootstrap(unittest.TestCase):
	"""
	tests multiplier bootstrap methods
	"""
	def test_multiplier_bootstrap_simple(self):
		# Parameters
		np.random.seed(1234)
		n, reps = 500, 200
		for d, eps_dist in zip(
			[1, 2, 5, 20],
			['laplace', 'gaussian', 'tdist', 'tdist']
		):
			alpha = 0.2
			mu = np.random.randn(d)
			sigmas = np.random.uniform(size=d)
			# Loop through and compute
			cis_upper = np.zeros(reps)
			cis_lower = np.zeros(reps)
			for r in range(reps):
				samples = utilities.parse_dist(
					eps_dist, mu=mu, sd=sigmas,
				).rvs(size=(n,d))
				# Test the ability to handle zero-variance columns
				if d > 1:
					samples[:, -1] = mu[-1]

				# Compute bounds
				_, ciu = bootstrap.multiplier_bootstrap(
					samples=samples, alpha=alpha, B=100
				)
				cis_upper[r] = ciu
				_, cil = bootstrap.multiplier_bootstrap(
					samples=samples, alpha=alpha, B=100, param='min'
				)
				cis_lower[r] = cil


			error_upper = np.mean(cis_upper > np.max(mu))
			error_lower = np.mean(cis_lower < np.min(mu))
			for error, name in zip([error_lower, error_upper], ['lower', 'upper']):
				self.assertTrue(
					error <= alpha + 3 * np.sqrt(alpha * (1-alpha) / reps),
					f"Multiplier bootstrap error={error} > alpha={alpha} for {name} CI with d={d}."
				)

	def test_db_multiplier_cluster_bootstrap_agree(self):
		# Fit two dualbounds objects
		data = db.gen_data.gen_regression_data(n=200, p=5, sample_seed=123)
		db_objects = []
		for Y_model in ['ridge', 'knn']:
			gdb = db.generic.DualBounds(
				outcome=data['y'], 
				covariates=data['X'], 
				treatment=data['W'], 
				propensities=data['pis'],
				outcome_model=Y_model,
				f=lambda y0, y1, x: (y1 < y0).astype(float)
			)
			gdb.fit(nfolds=2)
			db_objects.append(gdb)

		# Run mult bootstrap
		alpha = 0.1
		mbs_output = bootstrap.dualbound_multiplier_bootstrap(
			db_objects, alpha=alpha, B=10000
		)
		# Run cluster bootstrap
		cbs_output = bootstrap.dualbound_cluster_bootstrap(
			db_objects, alpha=alpha, B=10000
		)
		# Check that the results agree
		np.testing.assert_array_almost_equal(
			mbs_output.values,
			cbs_output.values,
			decimal=2,
			err_msg=f"Multiplier and cluster bootstrap do not agree."
		)