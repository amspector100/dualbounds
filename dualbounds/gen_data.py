"""
Generate synthetic data for tests and illustrations.
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
#from knockpy import dgp
from .utilities import parse_dist, _convert_to_cat

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
	# elif covmethod == 'ar1':
	# 	return dgp.AR1(p=p)
	# elif covmethod == 'factor':
	# 	return dgp.FactorModel(p=p)
	else:
		raise ValueError(f"Unrecognized covmethod={covmethod}")

def _sample_norm_vector(dim, norm):
	""" samples x ~ N(0, dim) and then normalizes so |x|_2 = norm """
	if norm == 0:
		return np.zeros(dim)
	x = np.random.randn(dim)
	return x / np.sqrt(np.power(x, 2).sum() / norm)

def gen_regression_data(
	n,
	p,
	lmda_dist='constant',
	eps_dist='gaussian',
	heterosked='constant',
	tauv=1, # Var(Y(1) | X ) / Var(Y(0) | X)
	r2=0,
	interactions=True,
	tau=0,
	betaW_norm=0,
	covmethod='identity',
	dgp_seed=1,
	sample_seed=None,
):
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
	stau=1, betaS_norm=1, **kwargs
):
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


# def gen_trial_data(
# 	tau=0,
# 	**kwargs,
# ):
# 	X, y, beta, Sigma = gen_regression_data(**kwargs)
# 	n, p = X.shape
# 	# create treatment
# 	W = np.arange(n) % 2
# 	# create treatment effect
# 	y += tau * W
# 	return X, W, y, beta, Sigma

# def compute_yprobs(
# 	yvals,
# 	mu,
# 	scales,
# 	eps_dist,
# ):
# 	"""
# 	Suppose y ~ mu + scales * eps_dist
# 	but restricted to the support specified by yvals.

# 	Returns
# 	-------
# 	probs : np.array
# 		n x nvals array.
# 		probs[i, j] = P(y(i) = yvals[j])
# 	"""
# 	log_probs = parse_dist(
# 		eps_dist, loc=mu, scale=scales, 
# 	).logpdf(yvals.reshape(-1, 1)).T
# 	log_probs = np.maximum(-500, log_probs)
# 	log_probs -= logsumexp(log_probs, axis=1).reshape(-1, 1)
# 	return np.exp(log_probs)

# def sample_disc_y(
# 	yvals,
# 	mu,
# 	scales,
# 	eps_dist,
# ):
# 	"""
# 	Suppose y ~ mu + scales * eps_dist
# 	but restricted to the support specified by yvals.

# 	Returns
# 	-------
# 	y : np.array
# 		n shaped array of y values sampled according to this
# 		model.
# 	"""
# 	n = mu.shape[0]
# 	probs = compute_yprobs(yvals, mu, scales, eps_dist)
# 	U = np.random.uniform(size=(n,1))
# 	inds = np.argmax(
# 		U <= np.cumsum(probs, axis=-1), axis=1
# 	)
# 	return yvals[inds]


# def gen_discrete_trial_data(
# 	n,
# 	p,
# 	lmda_dist='constant',
# 	eps_dist='gaussian',
# 	heterosked='constant',
# 	r2=0,
# 	tau=0,
# 	covmethod='identity',
# 	dgp_seed=1,
# 	sample_seed=None,
# 	Sigma=None,
# 	beta=None,
# 	nvals=21,
# 	yrange=20,
# ):
# 	# create parameters
# 	np.random.seed(dgp_seed)
# 	if Sigma is None:
# 		Sigma = create_cov(p=p, covmethod=covmethod)
# 	if beta is None:
# 		if r2 > 0:
# 			beta = np.random.randn(p)
# 			target = r2 / (1 - r2)
# 			beta /= np.sqrt(np.power(beta, 2).sum() / target)
# 		else:
# 			beta = np.zeros(p)

# 	# create y-values
# 	yvals = np.linspace(-yrange/2, yrange/2, nvals)

# 	# sample X
# 	np.random.seed(sample_seed)
# 	X = np.random.randn(n, p)
# 	L = np.linalg.cholesky(Sigma)
# 	X = X @ L.T
# 	lmdas = parse_dist(lmda_dist).rvs(size=n)
# 	lmdas /= np.sqrt(np.power(lmdas, 2).mean())
# 	X = X * lmdas.reshape(-1, 1)

# 	# create treatment
# 	W = np.arange(n) % 2

# 	# create final probabilities
# 	mu = X @ beta + W * tau
# 	scales = heteroskedastic_scale(X, W, heterosked=heterosked)
# 	y = sample_disc_y(
# 		yvals=yvals, mu=mu, scales=scales, eps_dist=eps_dist
# 	)

# 	# sample Y
# 	return X, W, y, beta, Sigma, yvals

# def compute_lee_logistic_probs(X, W, beta, gamma, stau):
# 	denom = np.sqrt(np.power(beta, 2).sum())
# 	if denom == 0:
# 		denom = 1
# 	beta = gamma * beta / denom
# 	Smu = stau * W + X @ beta
# 	return np.exp(Smu) / (1 + np.exp(Smu))

# def gen_lee_bound_data(stau, gamma, **kwargs):
# 	X, W, y, beta, Sigma, yvals = gen_discrete_trial_data(**kwargs)
# 	# sample S
# 	Sprobs = compute_lee_logistic_probs(X=X, W=W, beta=beta, gamma=gamma, stau=stau)
# 	S = np.random.binomial(1, Sprobs)
# 	return X, W, S, y, beta, Sigma, yvals 