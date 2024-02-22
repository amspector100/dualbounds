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

from dualbounds import bootstrap, utilities

class TestMultiplierBootstrap(unittest.TestCase):
	"""
	tests parse_dist
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
			cis = np.zeros(reps)
			for r in range(reps):
				samples = utilities.parse_dist(
					eps_dist, mu=mu, sd=sigmas,
				).rvs(size=(n,d))
				_, ci = bootstrap.multiplier_bootstrap(
					samples=samples, alpha=alpha, B=200
				)
				cis[r] = ci

			error = np.mean(cis > np.max(mu))
			self.assertTrue(
				error <= alpha + 3 * np.sqrt(alpha * (1-alpha) / reps),
				f"Multiplier bootstrap error={error} > alpha={alpha} for d={d}."
			)
