import time
import numpy as np
import pandas as pd
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

from dualbounds import varite, gen_data, delta


class TestDeltaDualBounds(unittest.TestCase):

	def test_deltadb_and_varite(self):
		""" This tests that DeltaDualBounds and VarITEDualBounds give the same answer. """
		data = gen_data.gen_regression_data(n=1000, p=5, dgp_seed=1, sample_seed=1)
		# Bootstrap variant
		deltadb  = delta.DeltaDualBounds(
			h=lambda f, z1, z0: f - (z1-z0)**2,
			z1=lambda y1, x: y1,
			z0=lambda y0, x: y0,
			f=lambda y0, y1, x: (y0-y1)**2,
			covariates=data['X'],
			outcome=data['y'],
			treatment=data['W'],
			propensities=data['pis'],
		)
		deltadb.fit().summary()
		# Analytical variant
		vdb = db.varite.VarITEDualBounds(
			covariates=data['X'],
			outcome=data['y'],
			treatment=data['W'],
			propensities=data['pis'],
		)
		vdb.fit(
			y0_dists=deltadb.y0_dists,
			y1_dists=deltadb.y1_dists,
			suppress_warning=True,
		).summary()
		for out, expected, name in zip(
			[deltadb.estimates, deltadb.cis],
			[vdb.estimates, vdb.cis],
			['estimates', 'cis'] 
		):
			np.testing.assert_array_almost_equal(
				out,
				expected,
				decimal=1,
				err_msg=f"VarITEDualBounds and DeltaDualBounds give different {name}"
			)

	def test_delta_from_pd(self):
		data = gen_data.gen_regression_data(n=50, p=5, eps_dist='bernoulli', r2=0)
		df = pd.DataFrame(data['X'])
		df['outcome'] = data['y']
		df['pis'] = data['pis']
		df['treatment'] = data['W']

		# Method 1
		fn_args = dict(
			f=lambda y0, y1, x: y1 - y0,
			h=lambda fval, z1, z0: fval,
			z0=lambda y0, x: y0,
			z1=lambda y1, x: y1, 
		)
		ddb1 = db.delta.DeltaDualBounds(
			**fn_args,
			covariates=pd.DataFrame(data['X']),
			outcome=pd.DataFrame(data['y']),
			treatment=pd.DataFrame(data['W']),
			propensities=pd.DataFrame(data['pis']),
		)
		# Method 2
		ddb2 = db.delta.DeltaDualBounds(
			**fn_args,
			covariates=data['X'],
			outcome=data['y'],
			treatment=data['W'],
			propensities=data['pis'],
		)

		# Test equality
		self.assertTrue(
			ddb1.X.shape == ddb2.X.shape,
			"initialization using pandas changes the shape of the covariates"
		)
		for expected, out, name in zip(
			[ddb2.y, ddb2.W, ddb2.pis],
			[ddb1.y, ddb1.W, ddb1.pis],
			['y', 'W', 'pis'],
		):
			np.testing.assert_array_almost_equal(
				expected,
				out,
				decimal=8,
				err_msg=f"DeltaDualBounds pandas init. changes {name} values."
			)




if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestDeltaDualBounds()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()