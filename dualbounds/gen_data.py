"""
Generate synthetic data for tests and illustrations.
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from .utilities import parse_dist, BatchedCategorical
from typing import Optional, Union

def heteroskedastic_scale(X, heterosked='constant'):
	n, p = X.shape
	if heterosked == 'constant' or heterosked == 'none':
		scale = np.ones(n)
	elif heterosked == 'linear':
		scale = np.abs(X[:, 0])
	elif heterosked == 'norm':
		scale = np.power(X, 2).sum(axis=-1)
	elif heterosked == 'invnorm':
		scale = 1 / (np.power(X, 2).sum(axis=-1))
	elif heterosked == 'exp_linear':
		scale = np.sqrt(np.exp(X[:, 0] + X[:, 1]))
	else:
		raise ValueError(f"Unrecognized heterosked={heterosked}")

	# normalize to ensure Var(epsilon) = 1 marginally
	scale /= np.sqrt(np.power(scale, 2).sum() / n)
	return scale

def create_cov(p, covmethod='identity'):
	covmethod = str(covmethod).lower()
	if covmethod == 'identity':
		return np.eye(p)
	else:
		raise ValueError(f"Unrecognized covmethod={covmethod}")

def _sample_norm_vector(dim, norm, sparsity=0):
	""" samples x ~ N(0, dim) and then normalizes so |x|_2 = norm """
	if norm == 0:
		return np.zeros(dim)
	# Possibly select sparse subset
	if sparsity > 0:
		x = np.zeros(dim)
		n_nonzero = int(max(1, np.ceil((1-sparsity) * dim)))
		x[np.random.choice(dim, n_nonzero, replace=False)] = _sample_norm_vector(
			dim=n_nonzero, norm=norm, sparsity=0,
		)
		return x
	# Else create a dense vector
	else:
		x = np.random.randn(dim)
		return x / np.sqrt(np.power(x, 2).sum() / norm)

def gen_regression_data(
	n: int,
	p: int,
	lmda_dist: str='constant',
	eps_dist: str='gaussian',
	heterosked: str='constant',
	tauv: float=1, # Var(Y(1) | X ) / Var(Y(0) | X)
	r2: float=0.95,
	sparsity: float=0,
	interactions: bool=True,
	tau: float=3,
	betaW_norm: float=0,
	covmethod: str='identity',
	dgp_seed: int=1,
	sample_seed: Optional[int]=None,
):
	"""
	Samples a synthetic regression dataset.

	Parameters
	----------
	n : int
		Number of observations.
	p : int
		Number of covariates
	lmda_dist : str
		str specifying the distribution of lmdai, 
		where Xi = lmdai * N(0, Sigma), so the 
		covariates are elliptically distributed. 
	eps_dist : str
		str specifying the distribution of the 
		residuals. See ``utilities.parse_dist``
		for the list of options.
	heterosked : str
		str specifying the type of heteroskedasticity.
		Defaults to ``constant``.
	tauv : float
		Ratio of Var(Y(1) | X) / Var(Y(0) | X)
	r2 : float
		Population r^2 of 1 - E[Var(Y | X)] / Var(Y).
	sparsity : float
		Proportion of covariates with zero coefficients.
		Defaults to zero (no sparsity).
	interactions : bool	
		If True (default), 
		Y = X beta + W * X * beta_int + epsilon.
		Else, the interactions between the treatment
		and the covariates are ommitted.
	tau : float
		Average treatment effect.
	betaW_norm : float
		E[W | X] = logistic(X @ betaW). This parameter
		controls the norm of betaW and thus the 
		variance of the propensity scores.
	covmethod : str
		str identifier for how to generate the covariance matrix.
	dgp_seed : int
		Random seed for the data-generating parameters.
	sample_seed : int
		Random seed for the randomness from sampling.

	Returns
	-------
	data : dict
		Dictionary with keys ``X`` (covariates),
		``y`` (response), ``W`` (treatment), 
		``pis`` (true propensity scores), 
		``beta``, ``beta_int``, ``betaW``, 
		and more.
	"""
	# create parameters
	np.random.seed(dgp_seed)
	Sigma = create_cov(p=p, covmethod=covmethod)
	# Create beta
	beta_norm = r2 / (1 - r2) if r2 > 0 else 0
	# No interactions with treatment
	if not interactions:
		beta = _sample_norm_vector(p, beta_norm, sparsity=sparsity)
		beta_int = np.zeros(p)
	# interactions with treatment
	else:
		beta = _sample_norm_vector(2*p, beta_norm, sparsity=sparsity)
		beta_int = beta[p:].copy()
		beta = beta[0:p].copy()

	# Create beta_W (for W | X)
	betaW = _sample_norm_vector(p, betaW_norm, sparsity=sparsity)

	# sample X
	np.random.seed(sample_seed)
	X = np.random.randn(n, p)
	L = np.linalg.cholesky(Sigma)
	X = X @ L.T
	lmdas = parse_dist(lmda_dist).rvs(size=n)
	lmdas /= np.sqrt(np.power(lmdas, 2).mean())
	X = X * lmdas.reshape(-1, 1)

	# sample W
	muW = X @ betaW
	pis = np.exp(muW)
	pis = pis / (1 + pis)
	# clip in truth
	pis = np.maximum(np.minimum(pis, 1-1e-3), 1e-3)
	W = np.random.binomial(1, pis)

	# conditional mean of Y
	mu = X @ beta
	# conditional variance of Y
	sigmas = heteroskedastic_scale(X, heterosked=heterosked)
	# allow for sigmas to depend on W
	sigmas0 = sigmas.copy()
	sigmas1 = tauv * sigmas.copy()
	denom = np.sqrt((np.mean(sigmas0**2) + np.mean(sigmas1**2)) / 2)
	sigmas0 /= denom; sigmas1 /= denom
	sigmas0 = np.maximum(1e-5, sigmas0)
	sigmas1 = np.maximum(1e-5, sigmas1)
	# Sample Y
	y0_dists = parse_dist(
		eps_dist, loc=mu, scale=sigmas0
	)
	cates = X @ beta_int + tau
	y1_dists = parse_dist(
		eps_dist, loc=mu+cates, scale=sigmas1
	)
	Y0 = y0_dists.rvs(); Y1 = y1_dists.rvs()
	Y = Y0.copy(); Y[W == 1] = Y1[W == 1]
	# return everything
	out = dict(
		X=X,
		W=W,
		y=Y,
		y0_dists=y0_dists,
		y1_dists=y1_dists,
		pis=pis,
		Sigma=Sigma,
		beta=beta,
		beta_int=beta_int,
		cates=y1_dists.mean() - y0_dists.mean(),
		betaW=betaW,
	)
	# # for convenience
	# if eps_dist == 'bernoulli':
	# 	out.update(dict(
	# 		_y0_dists_4input=BatchedCategorical.from_scipy(y0_dists),
	# 		_y1_dists_4input=BatchedCategorical.from_scipy(y1_dists),
	# 	))
	# else:
	# 	out.update(dict(
	# 		_y0_dists_4input=y0_dists,
	# 		_y1_dists_4input=y1_dists,
	# 	))
	return out

def gen_lee_bound_data(
	stau: float=1, betaS_norm: float=1, **kwargs
):
	"""
	Generates synthetic datasets with selection bias.

	Parameters
	----------
	stau : float
		In the logistic regression of S on W and X,
		stau is the coefficient on W.
	betaS_norm : float
		norm of coefficients in logistic regression of S 
		on W and X.
	"""
	# Generate regression data
	output = gen_regression_data(**kwargs)
	X, W = output['X'], output['W']
	n, p = X.shape
	# create DGP for S | X
	np.random.seed(kwargs.get("dgp_seed", 1))
	betaS = _sample_norm_vector(p, norm=betaS_norm)
	# sample S | X
	np.random.seed(kwargs.get("sample_seed", None))
	muS0 = X @ betaS
	muS1 = muS0 + stau
	s0_probs = np.exp(muS0) / (1 + np.exp(muS0))
	s1_probs = np.exp(muS1) / (1 + np.exp(muS1))
	S0 = np.random.binomial(1, s0_probs)
	S1 = np.random.binomial(1, s1_probs)
	S = S0.copy(); S[W == 1] = S1[W == 1]
	# save and return
	for key, val in zip(
		['s0_probs', 's1_probs', 'S'],
		[s0_probs, s1_probs, S]
	):
		output[key] = val
	return output

def gen_iv_data(
	n: int,
	p: int,
	betaZ_norm: float=0,
	betaW_norm: float=0.5,
	r2: float=0.95,
	sparsity: float=0,
	eps_dist: str='gaussian',
	lmda_dist: str='constant',
	interactions: bool=True,
	tau: float=3,
	tauv: float=1,
	tauZ: float=1,
	tau_conf: float=0,
	heterosked: str='constant',
	dgp_seed: int=1,
	sample_seed: Optional[int]=None,
):
	"""
	Generates IV data.
	
	Parameters
	----------
	n : int
		Number of observations.
	p : int
		Number of covariates
	lmda_dist : str
		str specifying the distribution of lmdai, 
		where Xi = lmdai * N(0, Sigma), so the 
		covariates are elliptically distributed. 
	eps_dist : str
		str specifying the distribution of the 
		residuals. See ``utilities.parse_dist``
		for the list of options.
	heterosked : str
		str specifying the type of heteroskedasticity.
		Defaults to ``constant``.
	tau : float
		Average treatment effect.
	tauv : float
		Ratio of Var(Y(1) | X) / Var(Y(0) | X)
	r2 : float
		Population r^2 of 1 - E[Var(Y | X)] / Var(Y).
	sparsity : float
		Proportion of covariates with zero coefficients.
		Defaults to zero (no sparsity).
	interactions : bool
		If True, there are interactions in the regression
		of Y on X.
	betaZ_norm: float
		Norm of beta in logistic regression of Z on X.
	betaW_norm : float
		Norm of beta in logistic regression of W on X, Z.
	tauZ : float
		Coefficient of Z in logistic regression of W on X, Z.
	dgp_seed : int
		Random seed for the data-generating parameters.
	sample_seed : int
		Random seed for the randomness from sampling.

	Returns
	-------
	data_dictionary : dict
		Dictionary of various data-generating parameters plus raw data.
	"""
	# create parameters
	np.random.seed(dgp_seed)
	# Create beta
	beta_norm = r2 / (1 - r2) if r2 > 0 else 0
	# No interactions with treatment
	if not interactions:
		beta = _sample_norm_vector(p, beta_norm, sparsity=sparsity)
		beta_int = np.zeros(p)
	# interactions with treatment
	else:
		beta = _sample_norm_vector(2*p, beta_norm, sparsity=sparsity)
		beta_int = beta[p:].copy()
		beta = beta[0:p].copy()

	# Create beta_W (for W | X)
	betaW = _sample_norm_vector(p, betaW_norm, sparsity=sparsity)
	betaZ = _sample_norm_vector(p, betaZ_norm, sparsity=sparsity)

	# sample X
	np.random.seed(sample_seed)
	X = np.random.randn(n, p)
	lmdas = parse_dist(lmda_dist).rvs(size=n)
	lmdas /= np.sqrt(np.power(lmdas, 2).mean())
	X = X * lmdas.reshape(-1, 1)

	### Sample Z | X
	muZ = X @ betaZ
	pis = np.exp(muZ)
	pis = pis / (1 + pis)
	# clip in truth
	pis = np.maximum(np.minimum(pis, 1-1e-3), 1e-3)
	Z = np.random.binomial(1, pis).astype(int)

	### Sample W | Z, X. Here U are meant to be unmeasured.
	wprobs = np.stack([X @ betaW, X @ betaW + tauZ], axis=1)
	wprobs = np.exp(wprobs) / (1 + np.exp(wprobs))
	U = np.random.uniform(size=n)
	W01 = (U.reshape(n, 1) <= wprobs).astype(int)
	W = W01[(np.arange(n), Z)]

	### Sample Y | Z, W, X
	## (a) conditional variance of Y
	sigmas = heteroskedastic_scale(X, heterosked=heterosked)
	# allow for sigmas to depend on W
	sigmas0 = sigmas.copy()
	sigmas1 = tauv * sigmas.copy()
	denom = np.sqrt((np.mean(sigmas0**2) + np.mean(sigmas1**2)) / 2)
	sigmas0 /= denom; sigmas1 /= denom
	## (b) conditional mean of Y | X, W U
	mu0 = X @ beta + tau_conf * np.sign(U - 1/2)
	_y0_dists = parse_dist(
		eps_dist, mu=mu0, sd=sigmas0
	)
	cates = X @ beta_int + tau
	mu1 = mu0 + cates
	_y1_dists = parse_dist(
		eps_dist, mu=mu1, sd=sigmas1
	)
	Y0 = _y0_dists.rvs(); Y1 = _y1_dists.rvs()
	Y = Y0.copy(); Y[W == 1] = Y1[W == 1]
	## (c) ydists for external use. 
	# ydists[z][w] are the laws of Y(w) | W(z) = w
	ydists = [[], []]
	if tau_conf != 0:
		warnings.warn("Oracle ydists not available when tau_conf != 0; need to implement ScipyMixture.")
	else:
		for z in [0,1]:
			for w, muY, sigmaY in zip([0,1], [mu0, mu1], [sigmas0, sigmas1]):
				ydists[z].append(parse_dist(eps_dist, mu=muY, sd=sigmaY))

	return dict(
		y=Y,
		W=W,
		Z=Z,
		X=X,
		pis=pis,
		cates=_y1_dists.mean() - _y0_dists.mean(),
		# betas
		beta=beta,
		beta_int=beta_int,
		betaZ=betaZ,
		betaW=betaW,
		# distributions
		wprobs=wprobs,
		ydists=ydists,
	)


# class ScipyMixture:
# 	"""
# 	A finite mixture of scipy distributions.

	
# 	Parameters
# 	---------
# 	distributions : rv_generic
# 		A list of scipy distributions of shape (n,).
# 	proportions : np.array
# 		The mixture proportions. Must sum to 1.

# 	Notes
# 	-----
# 	This is only used within the ``gen_iv_data``
# 	function to calculate the law of Y(w) | W(z) = w
# 	when tau_conf != 0.
# 	"""
# 	def __init__(
# 		self, distributions, proportions
# 	):
# 		self.dists = distributions
# 		self.props = proportions
# 		if not np.allclose(np.sum(self.props), 1):
# 			raise ValueError("proportions must sum to 1")

# 	def mean(self):
# 		"""
# 		Returns
# 		-------
# 		mu : np.array
# 			n-shaped array of means of mixture distribution.
# 		"""
# 		mus = np.stack([d.mean() * p for d, p in zip(self.dists, self.props)], axis=0)
# 		return np.sum(mus, axis=0)

# 	def cdf(self, x):
# 		"""
# 		Parameters
# 		----------
# 		x : np.array
# 			Inputs to CDF.

# 		Returns
# 		-------
# 		cdf : np.array
# 			CDF realized values.
# 		"""
# 		cdfs = np.stack([d.cdf() * p for d, p in zip(self.dists, self.props)], axis=0)
# 		return np.sum(cdfs, axis=0)

# 	def ppf(self, qs):
# 		"""
# 		Parameters
# 		----------
# 		qs : np.array
# 			Inputs to quantile function.

# 		Returns
# 		-------
# 		vals : np.array
# 			Quantile function evaluated at qs.
# 		"""
# 		pass