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

from dualbounds import utilities

class TestUtils(unittest.TestCase):
	"""
	tests parse_dist
	"""
	def test_parse_dist_mu_sd(self):
		"""
		Ensures returns dists with correct mean/sd.
		"""
		n = 20
		mu = np.random.randn(n)
		sd = np.random.uniform(n)
		for dist, kwargs in zip(
			['gamma', 'invchi2', 'tdist', 'expon'],
			[dict(a=5), dict(df=3), dict(df=6), dict()],
		):
			out = utilities.parse_dist(
				dist, mu=mu, sd=sd, **kwargs
			)
			# assert equality
			np.testing.assert_array_almost_equal(
				out.mean(), mu, 
				decimal=3,
				err_msg=f"parse_dist mean is wrong for dist={dist}, {kwargs}"
			)
			# assert equality
			np.testing.assert_array_almost_equal(
				out.std(), sd, 
				decimal=3,
				err_msg=f"parse_dist sd is wrong for dist={dist}, {kwargs}"
			)

		# test uniform
		mu = 0
		sd = np.sqrt(4 / 12)
		out = utilities.parse_dist('uniform', mu=0, sd=sd)
		np.testing.assert_array_almost_equal(
			np.array(out.support()),
			np.array([-1, 1]),
			decimal=3,
			err_msg=f"parse_dist returns wrong support for uniform with mu=0, sd={sd}"
		)