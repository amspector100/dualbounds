import numpy as np
from scipy import stats

def multiplier_bootstrap(
	samples, alpha, B=1000, _maxarrsize=int(1e10)
):
	"""
	Parameters
	----------
	samples : np.array
		(n,d)-shaped array where samples[i]
		is i.i.d. with mean mui.
	alpha : float
		Nominal error control level.
	B : int
		Number of bootstrap replications
	_maxarrsize : float
		Maximum size of an array; used to
		save memory.


	Returns
	-------
	estimate : float
		Estimate of max(mu1, ..., mud).
	ci : float
		Lower confidence bound on max(mu1, ..., mud).
	"""
	n, d = samples.shape
	hatmu = samples.mean(axis=0)
	hatsigma = samples.std(axis=0)
	if np.any(hatsigma == 0):
		raise NotImplementedError()
	# Centered statistics
	Tbs = []

	# Determine batch size
	batchsize = min(B, max(1, int(_maxarrsize / (n * d))))
	n_batches = int(np.ceil(B / batchsize))
	# Loop and compute bootstrap
	for b in range(n_batches):
		W = np.random.randn(n, batchsize, 1)
		sw = np.sum(
			W * (samples - hatmu).reshape(n, 1, d),
			axis=0
		) / np.sqrt(n)
		Tbs.append(np.max(sw / hatsigma, axis=1))
	Tbs = np.concatenate(Tbs)
	# Compute quantile and upper CI
	quantile = np.quantile(Tbs, 1-alpha)
	estimate = np.max(hatmu)
	ci = np.max(hatmu - quantile * hatsigma / np.sqrt(n))
	return estimate, ci
