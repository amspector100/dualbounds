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

from dualbounds import varite, gen_data, delta


class TestDeltaDualBounds(unittest.TestCase):

	def test_deltadb_and_varite(self):
		""" This tests that DeltaDualBounds and VarITEDualBounds give the same answer. """
		data = gen_data.gen_regression_data(n=500, p=5, dgp_seed=1, sample_seed=1)
		# Bootstrap variant
		deltadb  = delta.DeltaDualBounds(
			h=lambda f, z1, z0: f - (z1-z0)**2,
			z1=lambda y1, x: y1,
			z0=lambda y0, x: y0,
			f=lambda y0, y1, x: (y0-y1)**2,
			X=data['X'],
			y=data['y'],
			W=data['W'],
			pis=data['pis'],
		)
		deltadb.compute_dual_bounds()
		# Analytical variant
		vdb = db.varite.VarITEDualBounds(
			X=data['X'],
			y=data['y'],
			W=data['W'],
			pis=data['pis'],
		)
		vdb.compute_dual_bounds(
			y0_dists=deltadb.y0_dists,
			y1_dists=deltadb.y1_dists,
			suppress_warning=True,
		)
		for out, expected, name in zip(
			[deltadb.estimates, deltadb.cis],
			[vdb.estimates, vdb.cis],
			['estimates', 'cis'] 
		):
			np.testing.assert_array_almost_equal(
				3*out,
				3*expected,
				decimal=1,
				err_msg=f"VarITEDualBounds and DeltaDualBounds give different {name}"
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