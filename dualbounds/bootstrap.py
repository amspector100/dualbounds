import numpy as np
import pandas as pd
from scipy import stats
from .utilities import vrange, cluster_bootstrap_se, itemize
from .generic import DualBounds
from .delta import DeltaDualBounds

def multiplier_bootstrap(
	samples: np.array, 
	alpha: float,
	B: int=1000, 
	maxarrsize: int=int(1e10),
	param: str='max',
	verbose: int=False,
):
	"""
	Computes multiplier bootstrap lower confidence bounds. 

	Precisely, computes a lower confidence bound on
	:math:`\max(\mu_1, \dots, \mu_d)`,
	where :math:`\mu_i` is the mean of ``samples[i]``.

	Parameters
	----------
	samples : np.array
		(n,d)-shaped array where samples[i]
		is i.i.d. with mean :math:`\mu_i`.
	alpha : float
		Nominal error control level.
	B : int
		Number of bootstrap replications
	maxarrsize : float
		Maximum size of an array; used to save memory.
	param : str
		
		- If param='max', computes a lower confidence bound on :math:`\max(\mu_1, \dots, \mu_d)`.
		- Else, computes an upper confidence bound on :math:`\min(\mu_1, \dots, \mu_d)`,

	verbose : bool
		If True, shows a progress bar. Only useful
		if ``samples`` is a very large matrix.

	Returns
	-------
	estimate : float
		Estimate of :math:`\max(\mu_1, \dots, \mu_d)`.
	ci : float
		Lower confidence bound on :math:`\max(\mu_1, \dots, \mu_d)`.
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

def dualbound_cluster_bootstrap(
	db_objects: list[DualBounds],
	aipw: bool=True,
	alpha: float=0.05,
	B: int=1000,
):
	"""
	Combines evidence across multiple DualBounds classes
	using a (clustered) bootstrap.

	Parameters
	----------
	db_objects : list
		A list of fit DualBounds classes.
	aipw : bool
		If True, uses AIPW estimators to reduce variance
		(highly recommended).
	alpha : float
		Nominal level, between 0 and 1.
	B : int
		Number of bootstrap replications.
	"""
	# Learn dimensionality
	K = len(db_objects)
	n = db_objects[0].aipw_summands.shape[1]
	# Original estimates and SEs
	estimates = np.stack([x.estimates for x in db_objects], axis=-1) # 2 X K
	ses = np.stack([x.ses for x in db_objects], axis=-1) # 2 X K
	# Combine all of the summands appropriately
	if aipw:
		summands = np.stack([x.aipw_summands for x in db_objects], axis=-1) # 2 X n x K
	else:
		summands = np.stack([x.ipw_summands for x in db_objects], axis=-1)
	if isinstance(db_objects[0], DeltaDualBounds):
		z0summands = np.stack([x.z0summands for x in db_objects], axis=-1) # n x d0 x K
		z1summands = np.stack([x.z1summands for x in db_objects], axis=-1) # n x d1 x K
		# Extract functions
		h = db_objects[0].h
	else:
		z0summands = np.zeros((n, 1, K))
		z1summands = np.zeros((n, 1, K))
		# Dummy versions of the DeltaDualBounds functions
		h = lambda fval, z0, z1: fval

	d0 = z0summands.shape[1]
	def compute_estimate(data):
		hatmu = data.mean(axis=0) # (1 + d0 + d1) x K
		return np.array([
			itemize(h(fval=hatmu[0, k], z0=hatmu[1:(d0+1), k], z1=hatmu[(d0+1):, k]))
			for k in range(K)
		])
		
	# Bootstrap
	combined_estimates = np.zeros(2)
	cis = np.zeros(2)
	#abe = np.zeros((2, B, K))
	#cbe = np.zeros((2, B, K))
	for lower in [0, 1]:
		samples = np.concatenate(
			[summands[1-lower].reshape(n, 1, K), z0summands, z1summands], axis=1
		) # n x (1 + d0 + d1) x K
		# Bootstrapped estimators
		_, bs_estimators = cluster_bootstrap_se(
			data=samples, 
			clusters=db_objects[0].clusters,
			func=compute_estimate,
			B=B,
			verbose=False,
		) # B x K
		# Centerd and standardized variant
		centered = (bs_estimators - estimates[1-lower]) / ses[1-lower]
		# Compute final estimates and bounds
		if lower == 1:
			hatq = np.quantile(centered.max(axis=-1), 1-alpha/2)
			combined_estimates[1-lower] = np.max(estimates[1-lower])
			cis[1-lower] = np.max(estimates[1-lower] - hatq * ses[1-lower])
		else:
			hatq = np.quantile(centered.min(axis=-1), alpha/2)
			combined_estimates[1-lower] = np.min(estimates[1-lower])
			cis[1-lower] = np.min(estimates[1-lower] - hatq * ses[1-lower])
	# Return
	return pd.DataFrame(
		np.stack([combined_estimates, cis], axis=0),
		index=['Estimate', 'Conf. Int.'],
		columns=['Lower', 'Upper'],
	)



def dualbound_multiplier_bootstrap(
	db_objects: list[DualBounds], 
	aipw: bool=True,
	alpha: float=0.05,
	**kwargs,
) -> pd.DataFrame:
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
	result : pd.DataFrame
		dataframe of results.
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
	return pd.DataFrame(
		np.stack([estimates, cis], axis=0),
		index=['Estimate', 'Conf. Int.'],
		columns=['Lower', 'Upper'],
	)