import numpy as np
from scipy import stats
from .utilities import vrange

def multiplier_bootstrap(
	samples, 
	alpha,
	B=1000, 
	maxarrsize=int(1e10),
	param='max',
	verbose=False,
):
	"""
	Computes a lower confidence bound on the max(mu),
	where mu is the vector of means of ``samples``.

	Parameters
	----------
	samples : np.array
		(n,d)-shaped array where samples[i]
		is i.i.d. with mean mui.
	alpha : float
		Nominal error control level.
	B : int
		Number of bootstrap replications
	maxarrsize : float
		Maximum size of an array; used to
		save memory.
	param : str
		If param='max', computes a lower CB on max(mu).
		Else, computes an upper CB on min(mu).
	verbose : bool
		If True, shows a progress bar. Only useful
		if ``samples`` is a very large matrix.

	Returns
	-------
	estimate : float
		Estimate of max(mu1, ..., mud).
	ci : float
		Lower confidence bound on max(mu1, ..., mud).
	"""
	if param != 'max':
		estimate, ci = multiplier_bootstrap(
			samples=-1 * samples, 
			alpha=alpha, 
			B=B, 
			maxarrsize=maxarrsize,
			param='max',
		)
		return -estimate, -ci

	hatmu = samples.mean(axis=0)
	hatsigma = samples.std(axis=0)
	# This is important: if hatsigma \approx 0,
	# treat it as zero to ensure numerical stability when dividing.
	# Otherwise this should have zero effect.
	hatsigma = np.around(hatsigma, 10) 
	if np.all(hatsigma == 0):
		return np.max(hatmu), np.max(hatmu)
	if np.any(hatsigma == 0):
		# use noise-free columns as a deterministic lower bound
		min_val = np.max(hatmu[hatsigma == 0])
		samples = samples[:, hatsigma != 0]
		hatmu = hatmu[hatsigma != 0]
		hatsigma = hatsigma[hatsigma != 0]
	else:
		min_val = - np.inf

	# Centered statistics
	Tbs = []

	# Determine batch size
	n, d = samples.shape
	batchsize = min(B, max(1, int(maxarrsize / (n * d))))
	n_batches = int(np.ceil(B / batchsize))
	# Loop and compute bootstrap
	for b in vrange(n_batches, verbose=verbose):
		W = np.random.randn(n, batchsize, 1)
		sw = np.sum(
			W * (samples - hatmu).reshape(n, 1, d),
			axis=0
		) / np.sqrt(n)
		Tbs.append(np.max(sw / hatsigma, axis=1))
	Tbs = np.concatenate(Tbs)
	# Compute quantile and upper CI
	quantile = np.quantile(Tbs, 1-alpha)
	estimate = max(np.max(hatmu), min_val)
	ci = max(
		np.max(hatmu - quantile * hatsigma / np.sqrt(n)), 
		min_val
	)
	return estimate, ci

def dualbound_multiplier_bootstrap(
	db_objects, aipw=True, alpha=0.1, **kwargs	
):
	"""
	Combines evidence across multiple DualBounds classes
	using the multiplier bootstrap.

	Parameters
	----------
	db_objects : list
		A list of fit DualBounds classes.
	aipw : bool
		If True, uses AIPW estimators to reduce variance
		(highly recommended).
	alpha : float
		Nominal level, between 0 and 1.
	kwargs : dict
		kwargs for dualbounds.bootstrap.multiplier_bootstrap.
	Returns
	-------
	Returns a dictionary with the following attributes:

	estimates : np.array
		array of lower and upper estimates. These in 
		general may be biased.
	cis : np.array
		1 - alpha confidence lower/upper bounds on 
		the partial identification bounds.
	"""
	# Fetch summands
	if aipw:
		summands = [x.aipw_summands for x in db_objects]
	else:
		summands = [x.ipw_summands for x in db_objects]

	# Separate lower/upper summands
	lower_summands = np.stack([x[0, :] for x in summands], axis=1)
	upper_summands = np.stack([x[1, :] for x in summands], axis=1)

	# Compute lower/upper CI
	lower_est, lower_ci = multiplier_bootstrap(
		samples=lower_summands,
		param='max',
		alpha=alpha/2,
		**kwargs
	)
	upper_est, upper_ci = multiplier_bootstrap(
		samples=upper_summands,
		param='min',
		alpha=alpha/2,
		**kwargs
	)
	# Return
	estimates = np.array([lower_est, upper_est])
	cis = np.array([lower_ci, upper_ci])
	return dict(
		estimates=estimates,
		cis=cis,
	)