import copy
import warnings
import numpy as np
import pandas as pd
import ot
from scipy import stats
from . import utilities
from .generic import DualBounds

def _delta_bootstrap_ses(
	h: callable,
	summands: np.array,
	z1summands: np.array,
	z0summands: np.array,
	B: int,
	alpha: float,
):
	"""
	Uses the bootstrap to compute SES and CIs for 

	:math:`h(E[f(Y(0), Y(1), X)], E[z_1(Y(1), X)], E[z_0(Y(0), X)])`.
	
	Parameters
	----------
	h : function
		real-valued function of fval, z0, z1, e.g.,
		``h = lambda fval, z0, z1 : fval / z0 + z1``.
	summands : np.array
		(2,n)-shaped array where summands[0].mean()
		is a lower bound on E[f(Y(0), Y(1), X)] and
		summands[1].mean() is an upper bound.
	z1summands : np.array
		(n,d)-length array where z1summands.mean(axis=0)
		estimates E[z_1(Y(1), X)].
	z0summands : np.array
		(n,d)-length array where z0summands.mean(axis=0) 
		estimates E[z_0(Y(0), X)].
	B : int
		Number of bootstrap replications.
	alpha : float
		Nominal level.
	"""
	d0 = z0summands.shape[1]
	# Stack everything in the appropriate format
	combined_summands = np.stack([
		np.concatenate([summands[[k]], z0summands.T, z1summands.T], axis=0) 
		for k in [0,1]
	], axis=0)
	return utilities.compute_est_bounds(
		summands=combined_summands,
		func=lambda x: h(
			fval=x[0], z0=x[1:(1+d0)], z1=x[(1+d0):]
		),
		B=B,
		#clusters=clusters,
		alpha=alpha,
	)
	# n = len(z1summands)
	# estimates = np.zeros(2)
	# bootstrap_ests = np.zeros((B, 2))
	# for lower in [1, 0]:
	# 	sbetas = summands[1-lower]
	# 	est = h(
	# 		sbetas.mean(), 
	# 		z0=z0summands.mean(axis=0),
	# 		z1=z1summands.mean(axis=0),
	# 	).item()
	# 	estimates[1-lower] = est
	# 	for b in range(B):
	# 		inds = np.random.choice(np.arange(n), size=n, replace=True)
	# 		bootstrap_ests[b, 1-lower] = h(
	# 			sbetas[inds].mean(),
	# 			z0=z0summands[inds].mean(axis=0),
	# 			z1=z1summands[inds].mean(axis=0),
	# 		).item()

	# # Standard errors 
	# scale = stats.norm.ppf(1-alpha/2)
	# ses = bootstrap_ests.std(axis=0)
	# cis = np.array([
	# 	estimates[0] - ses[0] * scale,
	# 	estimates[1] + ses[1] * scale
	# ])
	# return dict(
	# 	estimates=estimates,
	# 	ses=ses,
	# 	cis=cis,
	# 	bootstrap_ests=bootstrap_ests,
	# )


class DeltaDualBounds(DualBounds):
	"""
	Computes generalized dual bounds via the delta method.

	The estimand is

	:math:`h(E[f(Y(0), Y(1), X)], E[z_1(Y(1), X)], E[z_0(Y(0), X)])`

	where h must be monotone increasing in its first argument. 
	
	Parameters
	----------
	h : function
		real-valued function of fval, z0, z1, e.g.,
		``h = lambda fval, z0, z1 : fval / z0 + z1``.
	z0 : function
		vector-valued function of y0, x.
	z1 : function
		vector-valued function of y1, x.
	f : function
		Function which defines the partially identified estimand.
		Must be a function of three arguments: y0, y1, x 
		(in that order). E.g.,
		``f = lambda y0, y1, x : y0 <= y1``
	outcome : np.array | pd.Series
		n-length array of outcome measurements (Y).
	treatment : np.array | pd.Series
		n-length array of binary treatment (W).
	covariates : np.array | pd.Series
		(n, p)-shaped array of covariates (X).
	propensities : np.array | pd.Series
		n-length array of propensity scores :math:`P(W=1 | X)`. 
		If ``None``, will be estimated from the data.
	outcome_model : str | dist_reg.DistReg
		The model for estimating the law of :math:`Y \mid X, W`.
		Two options:

		- A str identifier, e.g., 'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'.
		- An object inheriting from ``dist_reg.DistReg``. 

		E.g., when ``outcome`` is continuous, the default is
		``outcome_model=dist_reg.CtsDistReg(model_type='ridge')``.
	propensity_model : str | sklearn classifier
		How to estimate the propensity scores if they are not provided.
		Two options:

		- A str identifier, e.g., 'ridge', 'lasso', 'elasticnet', 'randomforest', 'knn'.
		- An sklearn classifier, e.g., ``sklearn.linear_model.LogisticRegressionCV()``.
	discrete : bool
		If True, treats the outcome as a discrete variable. 
		Defaults to ``None`` (inferred from the data).
	support : np.array
		Optional support of the outcome, if known and discrete.
		Defaults to ``None`` (inferred from the data).
	model_kwargs : dict
		Additional kwargs for the ``outcome_model``, e.g.,
		``feature_transform``. See 
		:class:`dualbounds.dist_reg.CtsDistReg` or 
		:class:`dualbounds.dist_reg.BinaryDistReg` for more kwargs.
	"""
	def __init__(
		self, 
		h: callable, 
		z1: callable, 
		z0: callable,
		*args,
		**kwargs,
	) -> None:
		self.h = h
		self.z1 = z1
		self.z0 = z0
		#self.h_grad = h_grad
		super().__init__(*args, **kwargs)

	def _plug_in_results(self, B: int=1000):
		ests, ses, cis = _delta_bootstrap_ses(
			h=self.h,
			summands=self.objvals,
			z1summands=self.z1summands,
			z0summands=self.z0summands,
			alpha=self.alpha,
			B=B,
		)
		return pd.DataFrame(
			np.stack(
				[ests, ses, cis], 
				axis=0
			),
			index=['Estimate', 'SE', 'Conf. Int.'],
			columns=['Lower', 'Upper']
		)

	def _compute_final_bounds(
		self, 
		aipw: bool=True,
		alpha: float=0.05,
		B: int=1000,
	):
		"""
		Computes final bounds based on (A)IPW summands,
		using the delta method or the bootstrap.

		Parameters
		----------
		aipw : bool
			If True, uses AIPW estimation. 
		alpha : float
			Nominal level.
		B : int
			Number of bootstrap replications. Default: 1000.
		"""
		# Compute summands for AIPW
		self._compute_ipw_summands()
		summands = self.aipw_summands if aipw else self.ipw_summands
		# Compute summands for z1 and z0
		self.z1_vals = np.stack(
			[self.z1(self.y[i], self.X[i]) for i in range(self.n)], axis=0
		).reshape(self.n, -1)
		d1 = self.z1_vals.shape[-1] # dimension of z1
		self.z0_vals = np.stack(
			[self.z0(self.y[i], self.X[i]) for i in range(self.n)], axis=0
		).reshape(self.n, -1)
		d0 = self.z0_vals.shape[-1] # dimension of z0

		### Step 1: use AIPW ideas to center/scale z1_values z0_vals
		if aipw:
			self.z1_mus = np.zeros((self.n, d1))
			self.z0_mus = np.zeros((self.n, d0))
			for i in range(self.n):
				nvals1 = self.y1_vals.shape[1]
				for j in range(nvals1):
					self.z1_mus[i] += self.z1(self.y1_vals[i, j], self.X[i]) * self.y1_probs[i, j]
				nvals0 = self.y0_vals.shape[1]
				for j in range(nvals0):
					self.z0_mus[i] += self.z0(self.y0_vals[i, j], self.X[i]) * self.y0_probs[i, j]
		else:
			self.z1_mus = 0
			self.z0_mus = 0

		Wr = self.W.reshape(-1, 1)
		pisr = self.pis.reshape(-1, 1)
		self.z1summands = (self.z1_vals - self.z1_mus) * Wr / pisr + self.z1_mus
		self.z0summands = (self.z0_vals - self.z0_mus) * (1 - Wr) / (1 - pisr) 
		self.z0summands += self.z0_mus

		## sample means and bootstrap
		# Bootstrap is valid because the analytical delta method is valid
		self.alpha = alpha
		self.estimates, self.ses, self.cis = _delta_bootstrap_ses(
			h=self.h,
			summands=summands,
			z1summands=self.z1summands,
			z0summands=self.z0summands,
			B=B,
			alpha=self.alpha,
		)
		print(self.estimates.shape, self.ses.shape, self.cis.shape)
		# self.estimates = bootstrap_out['estimates']
		# self.ses = bootstrap_out['ses']
		# self.cis = bootstrap_out['cis']
		# self.bootstrap_ests = bootstrap_out['bootstrap_ests']
		# Return
		return dict(
			estimates=self.estimates,
			ses=self.ses,
			cis=self.cis,
		)
