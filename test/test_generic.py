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

from dualbounds import generic, utilities, gen_data
from dualbounds.utilities import parse_dist, _convert_to_cat


class TestGenericDualBounds(unittest.TestCase):

	def test_var_ite_oracle(self):
		# create dists
		np.random.seed(123)
		n = 1
		for dist0, dist1, discrete, support in zip(
			['gaussian', 'expon', 'tdist', 'bernoulli'],
			['gaussian', 'gaussian', 'expon', 'bernoulli'],
			[False, False, False, True],
			[None, None, None, np.array([0,1])],
		):
			y0_dists = parse_dist(
				dist0, loc=np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n)
			)
			y1_dists = parse_dist(
				dist1, loc=np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n)
			)
			y0_dists_input = _convert_to_cat(y0_dists, n=n) if discrete else y0_dists
			y1_dists_input = _convert_to_cat(y1_dists, n=n) if discrete else y1_dists
			# analytically compute lower/upper bounds on var(ITE)
			reps = 100000
			U = np.random.uniform(size=reps)
			y1 = y1_dists.ppf(U); y0l = y0_dists.ppf(U); y0u = y0_dists.ppf(1-U)
			lower = np.std(y1 - y0l)**2
			upper = np.std(y1 - y0u)**2
			mu2 = (y1_dists.mean() - y0_dists.mean())**2
			# create dualbound.
			# the input data is a dummy variable that does nothing
			nvals = 2000
			vdb = db.generic.DualBounds(
				f=lambda y0, y1, x: (y0-y1)**2,
				covariates=np.zeros(1),
				treatment=np.zeros(1),
				propensities=np.ones(1) * 1/2,
				outcome=np.zeros(1),
				discrete=discrete,
				support=support,
			)
			vdb.compute_dual_variables(
				y0_dists=y0_dists_input, 
				y1_dists=y1_dists_input, 
				y0_min=np.min(y0l),
				y0_max=np.max(y0l),
				y1_min=np.min(y1), 
				y1_max=np.max(y1),
				nvals0=nvals,
				nvals1=nvals,
			)
			# check if objvals match with lower, upper
			np.testing.assert_array_almost_equal(
				np.array([lower, upper]),
				vdb.objvals[:, 0] - mu2,
				decimal=1.5,
				err_msg=f"LP bounds for {dist0, dist1} do not agree with analytical bounds on Var(ITE)",
			)

	def test_gridsearch_loss(self):
		# Create fake heavy-tailed data
		np.random.seed(123)
		n = 300; nvals = 50; grid_size = 3 * nvals
		y0_dists = parse_dist(
			'tdist', loc=np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n)
		)
		y1_dists = parse_dist(
			'expon', loc=np.random.randn(n), scale=np.random.uniform(0.1, 1, size=n)
		)
		y0 = y0_dists.rvs()
		y1 = y1_dists.rvs()
		pis = np.ones(n) / 2
		W = np.random.binomial(1, pis, size=n)
		Y = y0.copy(); Y[W == 1] = y1[W == 1].copy()
		X = np.random.randn(n, 1)
		# Try with many different fs
		fs = [
			lambda y0, y1, x: (y0-y1)**2,
			lambda y0, y1, x: y0 <= y1,
			lambda y0, y1, x: (y0 <= 0) * (y1 <= 0),
		]
		# Test the size of objdiffs is small on average
		for f in fs:
			gdb = db.generic.DualBounds(
				f=f,
				covariates=X,
				treatment=W,
				propensities=pis,
				outcome=Y,
			)
			gdb.compute_dual_variables(
				y0_dists=y0_dists, 
				y1_dists=y1_dists,
				nvals0=nvals,
				nvals1=nvals,
			)
			np.testing.assert_array_almost_equal(
				gdb.objdiffs.mean(axis=1),
				np.zeros(2),
				decimal=1.5,
				err_msg=f"For f={f}, objdiffs are too large."
			)

	def test_base_models_cts(self):
		"""
		Ensure the built-in model types work.
		"""
		# Generate data
		data = gen_data.gen_regression_data(n=50, p=5)
		f = lambda y0, y1, x: y0 <= y1
		## 1. Model types to test
		## 1a. Stringkwargs
		model_types = ['ridge', 'lasso', 'elasticnet', 'rf', 'knn']
		# expected types
		mt_exps = [
			db.dist_reg.parse_model_type(mt, discrete=False) for mt in model_types
		]
		## 1b. Other inputs
		model_types.append(db.dist_reg.CtsDistReg(model_type='ridge'))
		model_types.append(db.dist_reg.QuantileDistReg(nquantiles=10, alphas=[0, 0.1]))
		mt_exps.extend([lm.RidgeCV, dict])

		## 2. sigma_models to test
		sigma_model_types = ['none', 'lasso', 'ols', 'None', None, None, None]
		sigma_model_exps = []
		for m2t in sigma_model_types:
			if m2t is None:
				sigma_model_exps.append(None)
			elif m2t.lower() == 'none':
				sigma_model_exps.append(None)
			else:
				sigma_model_exps.append(db.dist_reg.parse_model_type(m2t, discrete=False))


		## 3. Epsilon dists to test
		eps_dists = ['empirical', 'laplace', 'empirical', 'expon', 'tdist']
		eps_dists.extend(['these_args', 'should_be_ignored_and_not_raise_error'])
		for model, mt_exp, sigma_model, sigma_model_exp, eps_dist in zip(
			model_types,
			mt_exps,
			sigma_model_types,
			sigma_model_exps,
			eps_dists
		):
			gdb = db.generic.DualBounds(
				f=f, 
				covariates=data['X'], 
				treatment=data['W'], 
				outcome=data['y'],
				propensities=data['pis'], 
				outcome_model=model,
				heterosked_model=sigma_model,
				eps_dist=eps_dist,
			)
			gdb.fit(nfolds=3, alpha=0.05).summary()
			# Check the correct class for the main model
			Ym = gdb.model_fits[0].model
			self.assertTrue(
				isinstance(Ym, mt_exp),
				f"fit outcome_model {Ym} with propensity_model={model} is not of type {mt_exp}"
			)
			# Check correct class for the heteroskedastic model
			if sigma_model_exp is not None:
				Ym2 = gdb.model_fits[0].sigma2_model
				self.assertTrue(
					isinstance(Ym2, sigma_model_exp),
					f"sigma_model {Ym2} with heterosked_model={sigma_model} is not of type {sigma_model_exp}"
				)
			if sigma_model_exp is None:
				# In this case heterosked_model='none', so there should be no sigma_model
				with np.testing.assert_raises(AttributeError):
					x = gdb.model_fits[0].sigma2_model

	def test_base_models_discrete(self):
		"""
		Same as above but for discrete models.
		"""
		data = gen_data.gen_regression_data(n=300, p=3, eps_dist='bernoulli', r2=0)
		f = lambda y0, y1, x: y0 <= y1
		model_types = ['ridge', 'lasso', 'elasticnet', 'rf', 'knn']
		expected_mts = [
			db.dist_reg.parse_model_type(mt, discrete=True) for mt in model_types
		]
		model_types.append(db.dist_reg.BinaryDistReg(model_type='ridge'))
		expected_mts.append(lm.LogisticRegressionCV)
		for model_type, mt_exp in zip(model_types, expected_mts):
			gdb = db.generic.DualBounds(
				f=f, 
				covariates=data['X'], 
				treatment=data['W'], 
				outcome=data['y'],
				propensities=data['pis'], 
				outcome_model=model_type,
			)
			gdb.fit(nfolds=3, alpha=0.05).summary()
			# Check the correct model type
			Ym = gdb.model_fits[0].model
			self.assertTrue(
				isinstance(Ym, mt_exp),
				f"fit model {mt_exp} with outcome_model={model_type} is not of type {mt_exp}"
			)


	def test_fit_propensity_scores(self):
		"""
		Same as above but for propensity_model
		"""
		data = gen_data.gen_regression_data(n=300, p=3, eps_dist='gaussian', r2=0)
		f = lambda y0, y1, x : y0 <= y1
		propensity_models = ['ridge', 'lasso', 'knn', 'rf']
		expecteds = [
			db.dist_reg.parse_model_type(wm, discrete=True) for wm in propensity_models
		]
		propensity_models.append(sklearn.ensemble.AdaBoostClassifier(algorithm='SAMME'))
		expecteds.append(sklearn.ensemble.AdaBoostClassifier)
		for propensity_model, expected in zip(propensity_models, expecteds):
			gdb = db.generic.DualBounds(
				f=f,
				covariates=data['X'], treatment=data['W'], outcome=data['y'],
				propensities=None, # to be estimated 
				outcome_model='ridge',
				propensity_model=propensity_model,
			)
			gdb.fit(nfolds=3).summary()
			# Check the correct class
			Wm = gdb.propensity_model_fits[0]
			self.assertTrue(
				isinstance(Wm, expected),
				f"fit propensity_model {Wm} with propensity_model={propensity_model} is not of type {expected}"
			)

	def test_from_pd(self):
		# Sample data
		for eps_dist in ['gaussian', 'bernoulli']:
			data = gen_data.gen_regression_data(n=50, p=5, eps_dist='bernoulli', r2=0)

			# Method 1
			f = lambda y0, y1, x: y1 - y0
			gdb1 = db.generic.DualBounds(
				f=f,
				outcome=pd.Series(data['y']),
				treatment=pd.Series(data['W']),
				covariates=pd.DataFrame(data['X']),
				propensities=pd.Series(data['pis']),
			).fit()

			# Method 2
			gdb2 = db.generic.DualBounds(
				f=f,
				covariates=data['X'],
				outcome=data['y'],
				treatment=data['W'],
				propensities=data['pis'],
			)

			# Test equality
			self.assertTrue(
				gdb1.X.shape == gdb2.X.shape,
				"using pandas changes the shape of the covariates"
			)
			for expected, out, name in zip(
				[gdb2.y, gdb2.W, gdb2.pis],
				[gdb1.y, gdb1.W, gdb1.pis],
				['y', 'W', 'pis'],
			):
				np.testing.assert_array_almost_equal(
					expected,
					out,
					decimal=8,
					err_msg=f"Using pandas initialization changes {name} values."
				)


	def test_plug_in_no_covariates(self):
		np.random.seed(123)
		## DGP
		n, reps = 300, 100
		mu0 = np.zeros(n)
		mu1 = np.random.randn(n)
		y0_dists = stats.norm(loc=mu0)
		y1_dists = stats.norm(loc=mu1)
		pis = np.clip(np.exp(2*mu1) / (1 + np.exp(2*mu1)), 0.25, 0.75)
		## Unobserved Y(1), Y(0)
		y1 = y1_dists.rvs()
		y0 = y0_dists.rvs()
		## Target
		f = lambda y0, y1, x: np.maximum(0, y1-y0)
		## Initialize outputs
		naive = np.zeros((reps, 2))
		ests = np.zeros((reps, 2))
		oracles = np.zeros((reps, 2))
		for r in range(reps):
			# Observed Y(1), Y()
			W = np.random.binomial(1, pis, size=n)
			y = y1.copy(); y[W == 0] = y0[W==0]
			# Naive estimator
			naive[r] = db.generic.plug_in_no_covariates(
				outcome=y, treatment=W, f=f, propensities=None, max_nvals=n
			)['estimates']
			# Oracle estimator
			oracles[r] = db.generic.plug_in_no_covariates(
				outcome=np.concatenate([y1, y0]), 
				treatment=np.concatenate([np.ones(n), np.zeros(n)]), 
				f=f,
				propensities=None,
				max_nvals=2*n,
			)['estimates']
			# Real estimator
			ests[r] = db.generic.plug_in_no_covariates(
				outcome=y, treatment=W, f=f, propensities=pis, max_nvals=n
			)['estimates']

		# Take averages
		naive_mu = naive.mean(axis=0)
		est_mu = ests.mean(axis=0)
		oracle_mu = oracles.mean(axis=0)

		# Test inaccuracy of naive method
		msg = "Nonconstant prop scores should invalidate plug_in_no_covariates"
		for lower in [1, 0]:
			self.assertTrue(
				np.abs(naive_mu[1-lower] - oracle_mu[1-lower]) > 0.1,
				msg + f": naive_est={naive_mu}, correct={oracle_mu}" 
			)

		# Test accuracy of correct method
		np.testing.assert_array_almost_equal(
			3*est_mu,
			3*oracle_mu,
			decimal=1,
			err_msg="plug_in_no_covariates fails to adjust for prop scores"
		)

if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestGenericDualBounds()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
