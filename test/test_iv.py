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
from dualbounds.utilities import parse_dist, _convert_to_cat

class TestDualIVBounds(unittest.TestCase):
	
	def test_balke_pearl_reduction(self):
		"""
		Test that dualIV bounds reduce to Balke-Pearl.
		""" 
		


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
