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
	def test_adjust_support_size(self):
		# A test where we know the result
		vals = np.array([0, 1, 2, 3, 3.1, 4.01, 5.05])
		probs = np.array([5., 1, 1, 1, 1, 1, 1])
		probs /= probs.sum()
		newvals, newprobs = utilities._adjust_support_size_unbatched(
			vals, probs, new_nvals=5, ymin=None, ymax=None
		)
		evals = np.array([0, 1.5, 3.05, 4.01, 5.05])
		eprobs = np.array([5., 2, 2, 1, 1])
		eprobs /= eprobs.sum()
		for out, expected in zip([newvals, newprobs], [evals, eprobs]):
			np.testing.assert_array_almost_equal(
				out, expected, decimal=4, 
				err_msg="greedy _adjust_support_size gives incorrect result"
			)

		# Test results where we don't know the answer
		n = 10
		new_nvals = 5
		for nvals in [new_nvals - 1, new_nvals, new_nvals + 20]:
			vals = np.sort(np.random.randn(n, nvals), axis=1)
			probs = np.random.uniform(size=(n, nvals))
			probs /= probs.sum(axis=1).reshape(-1, 1)
			newvals, newprobs = utilities.adjust_support_size(
				vals, probs, new_nvals=new_nvals, ymin=-1, ymax=1
			)
			self.assertTrue(
				newvals.shape == (n, new_nvals),
				f"Adjusted dist has wrong shape: expected n={n}, nvals={new_nvals}"
			)
			np.testing.assert_array_almost_equal(
				np.sum(vals * probs, axis=1),
				np.sum(newvals * newprobs, axis=1),
				decimal=6,
				err_msg="Mean of original/adjusted distributions are different"
			)


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


	def test_weighted_quantile(self):
		np.random.seed(123)
		n = 101
		# Data and desired quantiles
		x = np.random.randn(n)
		probs = np.ones(n) / n
		qs = np.linspace(0, 1, n-1)
		# Check that weighted_quantile agrees with np.quantile
		out = utilities.weighted_quantile(x, probs, qs)
		expected = np.quantile(x, qs)
		np.testing.assert_array_almost_equal(
			out,
			expected,
			decimal=5,
			err_msg=f"weighted_quantile with equal weights differs from np.quantile"
		)
		# Check weighted variant
		n = 1001
		x = np.linspace(0, 1, n)
		probs = x**2 / np.sum(x**2)
		out = utilities.weighted_quantile(x, probs, 0.5)
		expected = np.median(
			np.random.choice(x, p=probs, size=100000, replace=True)
		)
		np.testing.assert_array_almost_equal(
			out,
			expected,
			decimal=2,
			err_msg=f"weighted_quantile median from sampling approach"
		)