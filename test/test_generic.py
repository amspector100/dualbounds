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
from dualbounds.utilities import parse_dist, BatchedCategorical


class TestGenericDualBounds(context.DBTest):

	def test_var_ite_oracle(self):
		"""
		Tests that Var(ITE) objvals are accurate.
		"""
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
			y0_dists_input = BatchedCategorical.from_scipy(y0_dists) if discrete else y0_dists
			y1_dists_input = BatchedCategorical.from_scipy(y1_dists) if discrete else y1_dists
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


	def test_optimal_dualvars_consistency(self):
		"""
		Tests that if we evaluate the optimal dual variables
		we get the same value as the objvals in the perfectly
		well-specified case.
		"""
		np.random.seed(1)
		n, nvals, reps = 11, 5, 2000
		# propensity scores
		pis = np.random.uniform(0.3, 0.7, size=n)
		# Create distributions satisfying monotonicity
		y0_vals = np.random.randn(n, nvals)
		#y0_vals = np.stack([y0_vals for _ in range(n)], axis=0)
		probs = np.random.uniform(size=(n, nvals))
		probs /= probs.sum(axis=-1).reshape(n, 1)
		y1_vals = y0_vals + 1
		# create distributions
		y0_dists = BatchedCategorical(vals=y0_vals, probs=probs)
		y1_dists = BatchedCategorical(vals=y1_vals, probs=probs)
		ydists_input = dict(
			y0_dists=y0_dists, y1_dists=y1_dists, suppress_warning=True,
		)
		# Sample values of Y and W
		W = np.random.binomial(1, pis, size=(reps, n)).astype(int)
		Y = np.zeros((reps, n))
		for r in range(reps):
			y = y0_dists.rvs()
			y[W[r] == 1] = y1_dists.rvs()[W[r] == 1]
			Y[r] = y
		# Define estimand
		f = lambda y0, y1, x : y1 - y0 >= 0.5
		# Loop through different support restrictions
		for srname, support_restriction in zip(
			['none', 'monotonicity'], 
			[None, lambda y0, y1, x: y0 <= y1]
		):
			for dual_strategy in ['ot', 'lp', 'qp', 'se']:
				# Fit original dual bounds object
				gdb = db.generic.DualBounds(
					f=f,
					outcome=Y[0],
					treatment=W[0],
					propensities=pis,
					discrete=True,
					support_restriction=support_restriction,
				)
				gdb.fit(
					dual_strategy=dual_strategy, **ydists_input
				)#.summary()
				# Now, evaluate many dual variables
				aipw_estimates = np.zeros((reps, 2))
				ipw_estimates = np.zeros((reps, 2))
				for r in range(reps):
					# Collect W and Y
					# Compute new (A)IPW summands 
					gdb.W = W[r]
					gdb._compute_realized_dual_variables(y=Y[r], grid_size=0)
					gdb._compute_final_bounds(aipw=True)
					aipw_estimates[r] = gdb.estimates
					gdb._compute_final_bounds(aipw=False)
					ipw_estimates[r] = gdb.estimates
				# Test
				for method, estimates in zip(
					['aipw', 'ipw '], [aipw_estimates, ipw_estimates]
				):
					np.testing.assert_array_almost_equal(
						estimates.mean(axis=0),
						gdb.objvals.mean(axis=1),
						decimal=2 if method == 'aipw' else 1,
						err_msg=f"Oracle {method} estimates are not unbiased for objective value, dual_strategy={dual_strategy}"
					)

	def test_base_models_cts(self):
		"""
		Ensure the built-in model types work.
		"""
		# Generate data
		data = gen_data.gen_regression_data(n=50, p=5, heterosked='exp_linear')
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
			gdb.fit(nfolds=2, alpha=0.05).summary()
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
		data = gen_data.gen_regression_data(n=300, p=3, eps_dist='gaussian', r2=0, heterosked='invnorm')
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
		# Sample data. The different heterosked kwargs are just to increase coverage
		for eps_dist, heterosked in zip(
			['gaussian', 'bernoulli'], ['linear', 'norm']
		):
			data = gen_data.gen_regression_data(
				n=100, p=5, eps_dist=eps_dist, r2=0, interactions=False, heterosked=heterosked
			)

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

		# Test handling of strings
		W = pd.Series(
			np.array(['c', 't'])[data['W'].astype(int)]
		).astype(str)
		# Create some nans to test nan handling
		data['X'][:, 1:][data['X'][:, 1:] < -2] = np.nan
		# Create some discrete features
		X = pd.DataFrame(data['X'])
		X[0] = X[0].apply(lambda x: 'a' if x > 0 else 'b')
		gdb = db.generic.DualBounds(
			f=f,
			outcome=data['y'],
			treatment=W,
			covariates=X,
			propensities=data['pis'],
		).fit(nfolds=2).summary()

	def test_generic_model_selection_no_error(self):
		# Make sure that using multiple outcome models doesn't lead to an error
		data = db.gen_data.gen_regression_data(n=100, p=1, sample_seed=123)
		gdb = db.generic.DualBounds(
			f=lambda y0, y1, x: np.maximum(y1-y0,0),
			outcome=data['y'],
			covariates=data['X'],
			treatment=data['W'],
			propensities=data['pis'],
			# outcome models with and without interactions
			outcome_model=['ridge', db.dist_reg.CtsDistReg('ridge', how_transform='identity')],
		).fit(nfolds=3)

	@pytest.mark.slow
	def test_support_restriction_consistency(self):
		tau = 1
		n = 11
		data = db.gen_data.gen_regression_data(
			n=n, p=1, r2=0.5, dgp_seed=1, sample_seed=1, eps_dist='gaussian',
			interactions=False, tau=tau,
		)
		# Train on augmented data to get consistent estimates
		def gen_new_data(reps):		
			W = np.random.binomial(1, data['pis'], size=(reps, n))
			Y0 = data['y0_dists'].rvs(size=(reps, n))
			Y1 = data['y1_dists'].rvs(size=(reps, n))
			Y = Y0.copy(); Y[W==1] = Y1[W==1].copy()
			return Y, W

		# Fit ridge
		t0 = time.time()
		reps = 5000
		Y, W = gen_new_data(reps=reps)
		ridge = db.dist_reg.CtsDistReg(model_type='ridge', how_transform='identity')
		ridge.fit(
			y=np.concatenate([Y[r] for r in range(reps)], axis=0),
			W=np.concatenate([W[r] for r in range(reps)], axis=0),
			X=np.concatenate([data['X'] for r in range(reps)], axis=0),
		)
		hat_y0_dists, hat_y1_dists = ridge.predict_counterfactuals(data['X'])


		# Fit oracle and empirical dual variables
		gdb_args = dict(
			f=lambda y0, y1, x: (y1-y0)**2,
			outcome=data['y'],
			treatment=data['W'],
			propensities=data['pis'],
			covariates=data['X'],
			support_restriction=lambda y0, y1, x: y0 <= y1,
		)
		oracle_gdb = db.generic.DualBounds(**gdb_args)
		oracle_gdb.fit(
			y0_dists=data['y0_dists'], y1_dists=data['y1_dists'], suppress_warning=True,
		).summary()
		empirical_gdb = db.generic.DualBounds(**gdb_args)
		empirical_gdb.fit(
			y0_dists=hat_y0_dists, y1_dists=hat_y1_dists, suppress_warning=True,
		).summary()

		# Evaluate many times
		reps = 500
		Y, W = gen_new_data(reps=reps)
		oracle_ests = np.zeros((reps, 2))
		empirical_ests = np.zeros((reps, 2))
		for r in range(reps):
			for ests, gdb in zip([oracle_ests, empirical_ests], [oracle_gdb, empirical_gdb]):
				gdb.W = W[r]
				gdb._compute_realized_dual_variables(y=Y[r], grid_size=0)
				gdb._compute_final_bounds()
				ests[r] = gdb.estimates
		# Check validity
		o_ests = oracle_ests.mean(axis=0)
		e_ests = empirical_ests.mean(axis=0)
		o_objs = oracle_gdb.objvals.mean(axis=1)
		e_objs = empirical_gdb.objvals.mean(axis=1)
		err_msg = f"DualBounds w/ monotonicity is not consistent: empirical objvals={e_objs}"
		err_msg += f", oracle objvals={o_objs}, empirical est={e_ests}, oracle est={o_ests}"
		for obj1, obj2 in zip(
			[e_objs, e_ests, e_ests],
			[o_objs, o_ests, o_objs],
		):
			np.testing.assert_allclose(
				obj1, 
				obj2,
				rtol=1e-1,
				err_msg=err_msg
			)

	def test_generic_dualbound_clustered_ses(self):
		data = db.gen_data.gen_regression_data(
			n=100, p=1, r2=0, dgp_seed=1, sample_seed=1, eps_dist='gaussian',
		)
		f = lambda y0, y1, x: np.abs(y1 - y0)
		# Stack data
		data = np.stack(
			[data['y'], data['W'], data['pis'], data['X'][:, 0]],
			axis=1
		)
		def gdb_se_function(data, clusters):
			gdb = generic.DualBounds(
				f=f,
				outcome=data[:, 0],
				treatment=data[:, 1],
				propensities=data[:, 2],
				covariates=data[:, 3].reshape(-1, 1),
				clusters=clusters,
			)
			gdb.fit().summary()
			return gdb.ses
		self.check_clustered_ses(func=gdb_se_function, data=data, msg_context='DualBounds')

class TestPluginBounds(unittest.TestCase):

	def test_plug_in_no_covariates(self):
		np.random.seed(123)
		## DGP
		n, reps, max_nvals = 350, 100, 150
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
				outcome=y, 
				treatment=W, 
				f=f, 
				propensities=None, 
				max_nvals=max_nvals,
				B=0
			)['estimates']
			# Oracle estimator
			oracles[r] = db.generic.plug_in_no_covariates(
				outcome=np.concatenate([y1, y0]), 
				treatment=np.concatenate([np.ones(n), np.zeros(n)]), 
				f=f,
				propensities=None,
				max_nvals=2*max_nvals,
				B=0,
			)['estimates']
			# Real estimator
			ests[r] = db.generic.plug_in_no_covariates(
				outcome=y, 
				treatment=W,
				f=f,
				propensities=pis,
				max_nvals=max_nvals,
				B=0
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

	def test_plug_in_ses(self):
		np.random.seed(123)
		f = lambda y0, y1, x: np.abs(y0 - y1)
		sample_kwargs = dict(n=200, p=1, dgp_seed=123)
		# Compute SEs on one run
		data = gen_data.gen_regression_data(**sample_kwargs, sample_seed=1)
		output = generic.plug_in_no_covariates(
			f=f,
			outcome=data['y'],
			treatment=data['W'],
			propensities=data['pis'],
			clusters=None,
			B=200,
		)
		# Compute oracle SE on many runs
		reps = 200
		ests = np.zeros((reps, 2))
		for r in range(reps):
			data = gen_data.gen_regression_data(**sample_kwargs, sample_seed=r+1)
			ests[r] = generic.plug_in_no_covariates(
				f=f,
				outcome=data['y'],
				treatment=data['W'],
				propensities=data['pis'],
				clusters=None,
				B=0,
			)['estimates']
		# Compare
		ses = ests.std(axis=0),
		hatses = output['ses']
		for obj1, obj2 in zip([ses, hatses], [hatses, ses]):
			self.assertTrue(
				np.all(obj1 / obj2 <= 1.2),
				msg=f"plug_in_no_covariates estimated se={hatses}, truth={ses}; these are not the same"
			)





if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	basename = os.path.basename(os.path.abspath(__file__))
	if sys.argv[0] == f'test/{basename}':
		time0 = time.time()
		context.run_all_tests([TestPluginBounds(), TestGenericDualBounds()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()
