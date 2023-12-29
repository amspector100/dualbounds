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

from dualbounds import dist_reg

def convert_norm_dists_to_array(dists):
	mus = []; sigmas = []
	for dist in dists:
		mu, sigma = dist.stats()
		mus.append(mu); sigmas.append(sigma)
	return np.concatenate(mus, axis=0), np.concatenate(sigmas, axis=0)

class TestCrossFit(unittest.TestCase):
	"""
	tests
	"""
	def test_crossfit_linreg(self):
		# create model 
		np.random.seed(123)
		n = 200; p = 30; nfolds = 3; cv = 3
		X = np.random.randn(n, p)
		W = np.random.binomial(1, 0.5, size=n)
		beta = np.random.randn(p)
		Y = X @ beta + np.random.randn(n)
		# cross-fitting approach one
		rdr = dist_reg.CtsDistReg(
			model_type='ridge', cv=cv, how_transform='identity'
		)
		preds0, preds1, _ = dist_reg.cross_fit_predictions(
			W=W, y=Y, X=X, nfolds=nfolds, model=rdr,
		)
		mu0s, sigma0s = convert_norm_dists_to_array(preds0)
		mu1s, sigma1s = convert_norm_dists_to_array(preds1)

		# manually perform cross-fitting in this case
		starts, ends = dist_reg.create_folds(n=n, nfolds=nfolds)
		p0s = []; p1s = []
		for start, end in zip(starts, ends):
			not_in_fold = [i for i in range(n) if i < start or i >= end]
			model = dist_reg.CtsDistReg(
				model_type='ridge', cv=cv, how_transform='identity'
			)
			model.fit(W=W[not_in_fold], X=X[not_in_fold], y=Y[not_in_fold])
			pr0s, pr1s = model.predict(X=X[start:end])
			p0s.append(pr0s); p1s.append(pr1s)
		mu0s_exp, sigma0s_exp = convert_norm_dists_to_array(p0s)
		mu1s_exp, sigma1s_exp = convert_norm_dists_to_array(p1s)

		# check everything is the same
		for actual, expected, name in zip(
			[mu0s, sigma0s, mu1s, sigma1s],
			[mu0s_exp, sigma0s_exp, mu1s_exp, sigma1s_exp],
			['mu0', 'mu1', 'sigma0', 'sigma1']
		):
			np.testing.assert_array_almost_equal(
				actual, expected, decimal=3, err_msg=f"{name} from cross_fit_predictions did not match manual method"
			)

if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestCrossFit()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
