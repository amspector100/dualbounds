import time
import numpy as np
import scipy as sp
from scipy import stats
from sklearn import linear_model as lm
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

class TestDistReg(unittest.TestCase):
	"""
	tests
	"""
	def test_oos_ridge(self):
		# create data
		np.random.seed(123)
		n, p = 100, 50
		X = np.random.randn(n, p)
		beta = np.random.randn(p) * 0.2
		y = np.random.randn(n) + X @ beta + 10
		# Fit various ridge models and oos residuals
		for fit_intercept in [True, False]:
			for alpha in [0.1, 9]:
				# Fit ridge
				kwargs = dict(fit_intercept=fit_intercept)
				ridge_nocv = lm.Ridge(alpha=alpha, solver='cholesky', **kwargs)
				ridge_nocv.fit(X, y)
				# Fit ridge CV
				ridge_cv = lm.RidgeCV(**kwargs)
				ridge_cv.fit(X, y)
				for ridge, cv in zip([ridge_nocv, ridge_cv], [False, True]):
					# Fetch out of sample
					loo_resids = dist_reg.ridge_loo_resids(
						features=X, y=y, ridge_cv_model=ridge,
					)
					# Fit ridge leaving one ind out
					ind = 3
					negind = np.array([i for i in range(n) if i != ind])
					if isinstance(ridge, lm.RidgeCV):
						ridge_negind = lm.Ridge(alpha=ridge.alpha_, **kwargs)
					else:
						ridge_negind = lm.Ridge(alpha=alpha, **kwargs)
					ridge_negind.fit(X[negind], y[negind])
					expected = y[[ind]] - ridge_negind.predict(X[[ind]])
					# test equality
					np.testing.assert_array_almost_equal(
						expected,
						loo_resids[[ind]],
						decimal=3 if not fit_intercept else 1,
						err_msg=f"Analytical ridge LOO comp failed for kwargs={kwargs}, cv={cv}"
					)


	def test_crossfit_linreg(self):
		# create model 
		np.random.seed(123)
		n = 200; p = 30; nfolds = 3; cv = 3
		X = np.random.randn(n, p)
		W = np.random.binomial(1, 0.5, size=n)
		beta = np.random.randn(p)
		Y = X @ beta + np.random.randn(n)
		# cross-fitting approach one
		model_kwargs = dict(model_type='ridge', cv=cv, how_transform='identity', eps_dist='gaussian')
		rdr = dist_reg.CtsDistReg(**model_kwargs)
		(preds0, preds1), _, _ = dist_reg.cross_fit_predictions(
			W=W, y=Y, X=X, nfolds=nfolds, model=rdr,
		)
		mu0s, sigma0s = convert_norm_dists_to_array(preds0)
		mu1s, sigma1s = convert_norm_dists_to_array(preds1)

		# manually perform cross-fitting in this case
		starts, ends = dist_reg.create_folds(n=n, nfolds=nfolds)
		p0s = []; p1s = []
		for start, end in zip(starts, ends):
			not_in_fold = [i for i in range(n) if i < start or i >= end]
			model = dist_reg.CtsDistReg(**model_kwargs)
			model.fit(W=W[not_in_fold], X=X[not_in_fold], y=Y[not_in_fold])
			pr0s = model.predict(X=X[start:end], W=np.zeros(end-start))
			pr1s = model.predict(X=X[start:end], W=np.ones(end-start))
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
		context.run_all_tests([TestDistReg()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
