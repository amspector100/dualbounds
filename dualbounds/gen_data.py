"""
Generate synthetic data for tests and illustrations.
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from .utilities import parse_dist, _convert_to_cat
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

def _sample_norm_vector(dim, norm):
	""" samples x ~ N(0, dim) and then normalizes so |x|_2 = norm """
	if norm == 0:
		return np.zeros(dim)
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
		beta = _sample_norm_vector(p, beta_norm)
		beta_int = np.zeros(p)
	# interactions with treatment
	else:
		beta = _sample_norm_vector(2*p, beta_norm)
		beta_int = beta[p:].copy()
		beta = beta[0:p].copy()

	# Create beta_W (for W | X)
	betaW = _sample_norm_vector(p, betaW_norm)

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
	# for convenience
	if eps_dist == 'bernoulli':
		out.update(dict(
			_y0_dists_4input=_convert_to_cat(y0_dists, n=n),
			_y1_dists_4input=_convert_to_cat(y1_dists, n=n)
		))
	else:
		out.update(dict(
			_y0_dists_4input=y0_dists,
			_y1_dists_4input=y1_dists,
		))
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