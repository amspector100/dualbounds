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

from dualbounds.interpolation import adaptive_interpolate

class TestInterpolation(unittest.TestCase):
	"""
	tests parse_dist
	"""
	def test_interpolation(self):
		# A test which should use linear interpolation
		n = 5
		x = np.arange(n)
		y = np.arange(n)
		newx = np.arange(20) - 10
		expected = newx.copy()
		out = adaptive_interpolate(x, y, newx)
		np.testing.assert_array_almost_equal(
			expected, out, decimal=5, err_msg="adaptive_interpolate fails for a linear function"
		)
		# A test which should use nearest-neighbor interpolation
		x = np.arange(n)
		y = x >= 2
		newx = np.array([-20, 1.4, 2.1, 20])
		expected = newx >= 2
		out = adaptive_interpolate(x, y, newx)
		np.testing.assert_array_almost_equal(
			expected, out, decimal=5, err_msg="adaptive_interpolate fails for indicator function"
		)