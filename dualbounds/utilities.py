import copy
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool
from functools import partial
from itertools import product
from tqdm.auto import tqdm
from typing import Optional, Union

def elapsed(t0):
	return np.around(time.time() - t0, 2)

def vrange(n, verbose=False):
	if not verbose:
		return range(n)
	else:
		return tqdm(list(range(n)))

def haslength(x):
	try:
		len(x)
		return True
	except:
		return False

def floatable(x):
	try:
		float(x)
		return True
	except:
		return False

def itemize(x):
	try:
		return x.item()
	except AttributeError:
		return x

### For processing data
def _binarize_variable(x: np.array, var_name: str):
	"""
	Converts x to a binary r.v. and raises an error
	if it contains more than 2 values.
	"""
	# Check if nans
	if x.dtype in [int, np.int32, np.int64, np.float32, np.float64]:
		if np.any(np.isnan(x)):
			raise ValueError(f"{var_name} has missing values.")

	# Check if binary
	vals = set(list(np.unique(x)))
	if len(vals) > 2:
		raise ValueError(f"{var_name} is not binary and takes {len(vals)} values.")
	elif len(vals - set([0,1])) == 0:
		return x.astype(int)
	else:
		vals = sorted(list(vals))
		print(f"For {var_name}, replacing {vals[0]} with 0 and {vals[1]} with 1.")
		return (x == list(vals)[1]).astype(int)

def process_covariates(data: pd.DataFrame):
	"""
	Performs rudimentary pre-processing of covariates.

	Parameters
	----------
	data : pd.DataFrame
		Dataframe of covariates.

	Returns
	-------
	processed : pd.DataFrame
		DataFrame of preprocessed covariates.
	"""
	cts_covs = []
	discrete_covs = []
	for c in data.columns:
		if np.all(data[c].apply(floatable)):
			cts_covs.append(c)
		else:
			discrete_covs.append(c)

	# Get dummies and return
	return pd.get_dummies(
		data[discrete_covs + cts_covs],
		columns=discrete_covs,
		dummy_na=True,
		drop_first=True,
	).astype(float)

def preprocess_clusters(clusters):
	"""
	Parameters
	----------
	clusters : np.array
		n-length array of cluster indicators.

	Returns
	-------
	new_clusters : np.array
		n-length array of clusters which takes values
		from 0 to len(np.unique(clusters))-1.
	"""
	new_clusters = np.zeros(clusters.shape)
	for i, x in enumerate(np.sort(np.unique(clusters))):
		new_clusters[clusters == x] = i
	return new_clusters.astype(int)

def cluster_bootstrap_se(
	data: np.array,
	clusters: Optional[callable]=None,
	func: Optional[callable]=None,
	B: int=1000,
	verbose: bool=False,
):
	"""
	Computes clustered bootstrap on func(data).

	Parameters
	----------
	data : np.array
		array whose zeroth axis reflects the number of observations n.
		Typically a n-shaped or (n,d)-shaped array.
	func : callable
		A callable that maps an np.array to a scalar or array.
		Defaults to ``func=lambda x: x.mean()``.
	clusters : np.array
		n-shaped array where clusters[i] = j means that observation
		i is in the jth cluster.
	B : int
		Number of bootstrap draws to use. Ignored unless func or
		clusters are provided.
	verbose : bool
		If True, provides a progress bar.

	Returns
	-------
	se : float | np.array
		standard error of the estimator.
	bs_ests : np.array
		Array of bootstrapped estimators.
	"""
	# parse function
	n = len(data)
	if func is None:
		func = lambda x: x.mean()
	# Non-clustered case
	bs_ests = []
	if clusters is None:
		for b in vrange(B, verbose=verbose):
			inds = np.random.choice(n, n, replace=True)
			bs_ests.append(func(data[inds]))
	# Clustered case
	else:
		# Preprocess clusters
		clusters = preprocess_clusters(clusters).astype(int)
		n_clusters = len(np.unique(clusters))
		# data_clusters[j] = observations from cluster j
		data_clusters = [
			data[clusters==i] for i in range(n_clusters)
		]
		# Resample clusters uniformly at random
		for b in vrange(B, verbose=verbose):
			bs_clusters = np.random.choice(clusters, n_clusters, replace=True)
			new_data = np.concatenate(
				[data_clusters[i] for i in bs_clusters], axis=0
			)
			bs_ests.append(func(new_data))
	bs_ests = np.stack(bs_ests, axis=0)
	return bs_ests.std(axis=0), bs_ests

def compute_est_bounds(
	summands: np.array,
	clusters: Optional[callable]=None,
	func: Optional[callable]=None,
	B: int=1000,
	alpha: float=0.05,
):
	"""
	Helper to computes confidence intervals.

	Specifically, computes intervals for 

		``func(summands[k].mean(axis=-1))``

	Provides lower CI for k=0 and upper CI for k=1.

	Parameters
	----------
	summands : np.array
		(2, n)-shaped array or (2, n, d)-shaped array
	func : callable
		A callable that maps a 1D np.array to a scalar.
		Defaults to ``func=lambda x: x.mean()``.
	clusters : np.array
		n-shaped array where clusters[i] = j means that observation
		i is in the jth cluster.
	B : int
		Number of bootstrap draws to use. Ignored unless func or
		clusters are provided.
	alpha : float
		Nominal level.

	Returns
	-------
	ests : np.array
		2-shaped array of lower and upper estimators (sample mean).
	ses : np.array
		2-shaped array of standard errors.
	bounds : np.array
		2-shaped array of lower/upper confidence bounds.

	Notes
	-----
	This is meant primarily for internal use in the DualBounds class.
	"""
	scale = stats.norm.ppf(1-alpha/2)
	n = summands.shape[1]
	# In the simplest case, use analytical bounds
	if func is None and clusters is None:
		ests = summands.mean(axis=1)
		ses = summands.std(axis=1) / np.sqrt(n)
	# else, use the bootstrap/delta method
	else:
		if func is None:
			func = lambda x: x.mean()
		new_func = lambda data: itemize(func(data.mean(axis=0)))
		# Estimators
		ests = np.array([new_func(summands[k]) for k in [0,1]])
		ses = np.zeros(2)
		for k in [0,1]:
			ses[k] = itemize(cluster_bootstrap_se(
				data=summands[k],
				clusters=clusters,
				func=new_func,
				B=B,
			)[0])

	return ests, ses, np.array([
		ests[0] - scale * ses[0], ests[1] + scale * ses[1]
	])

def _sort_disc_dist(vals, probs):
	"""
	Parameters
	----------
	vals : (n, nvals)-shaped array.
	probs : (n, nvals)-shaped array. probs.sum(axis=1) == 1

	Returns
	-------
	vals_new : (n, nvals) array
		``vals`` sorted so that vals[i] is in increasing order.
	probs_new : (n, nvals) array
		``probs`` sorted in the corresponding order as vals.
	"""
	inds = np.argsort(vals, axis=1)
	new_vals = np.take_along_axis(vals, inds, axis=1)
	new_probs = np.take_along_axis(probs, inds, axis=1)
	return new_vals, new_probs

class ConstantDist:

	def __init__(self, loc=0, scale=1):
		self.loc = loc
		self.scale = scale

	def rvs(self, size):
		return self.loc + self.scale * np.ones(size)

class BatchedCategorical:
	"""
	Batched discrete (categorical) distribution.

	This class supports the small set of operations
	needed by the ``DualBounds`` class.

	Parameters
	----------
	vals : np.array
		(n, nvals)-shaped array of supports.
	probs : np.array
		(n, nvals)-shaped array of probabilities.
	"""
	def __init__(
		self, vals, probs
	):
		# sort
		self.n, self.nvals = vals.shape
		self.vals, self.probs = _sort_disc_dist(vals=vals, probs=probs)
		self.cumprobs = np.cumsum(self.probs, axis=1)
		# validate args
		if np.any(self.probs < 0):
			raise ValueError("probs must be nonnegative")
		if np.any(np.abs(self.probs.sum(axis=1)-1) > 1e-5):
			raise ValueError("probs.sum(axis=1) must equal 1")

	@classmethod
	def from_scipy(
		cls, scipy_bernoulli
	):
		"""
		Instantiates distribution from a 1D scipy.stats.bernoulli 
		object.

		Parameters
		----------
		scipy_dist : scipy.stats.bernoulli
		"""
		# Probabilities
		probs = scipy_bernoulli.mean()
		# Values 
		n = len(probs)
		vals = np.zeros((n, 2)); vals[:, 1] += 1
		# Return
		return cls(
			vals=vals, probs=np.stack([1-probs, probs], axis=1)
		)


	@classmethod
	def from_binary_probs(
		cls, probs
	):
		"""
		Instantiates distribution for a binary-valued variable.

		Parameters
		----------
		probs : np.array
			n-length array of P(Yi = 1)
		"""
		n = len(probs)
		probs_stacked = np.stack([1-probs, probs], axis=1)
		vals = np.zeros((n, 2))
		vals[:, 1] += 1
		return cls(vals=vals, probs=probs_stacked)
	
	def mean(self):
		"""
		Returns
		-------
		mu : np.array
			n-length mean of discrete distributions.
		"""
		return np.sum(self.vals * self.probs, axis=1)
	
	def ppf(self, q):
		"""
		Parameters
		----------
		q : np.array
			(m,n)-shaped array of desired quantiles, between 0 and 1.

		Returns
		-------
		quantile_values : np.array
			The desired quantile values of the categorical distributions.
		"""
		m = q.shape[0]
		# To be compatible with scipy, if q is (m, 1)
		# change it to (m,)
		if len(q.shape) == 2:
			if q.shape[1] == 1:
				q = q[:, 0]
		# Vectorize
		if len(q.shape) == 1:
			q = np.stack([q for _ in range(self.n)], axis=1)
		# use a for loop to save memory
		qvals = np.zeros((m, self.n))
		for i in range(self.n):
			flags = self.cumprobs[i].reshape(self.nvals, 1) >= q[:, i]
			qvals[:, i] = self.vals[i][np.argmax(flags, axis=0)]
		return qvals

	def rvs(self):
		"""
		Returns
		-------
		y : array
			n-length array of a single RV drawn from each categorical.
		"""
		u = np.random.uniform(size=(self.n, 1))
		inds = (u < self.cumprobs).argmax(axis=1)
		return self.vals[(np.arange(self.n), inds)]

def _adjust_support_size_unbatched(
	vals, probs, new_nvals, ymin, ymax
):
	"""
	Adjust categorical distribution to have a support of size new_nvals.
	
	Parameters
	----------
	vals : np.array
		nvals-length array of support. must be sorted.
	probs : np.array
		nvals-length array of probabilities
	new_nvals : int
		Desired size of support.
	ymin : float
		Minimum value for support
	ymax : float
		Maximum value for support
	"""
	nvals = len(vals)
	vals = vals.copy(); probs = probs.copy()
	if np.any(vals != np.sort(vals)):
		raise ValueError("vals must be sorted")
	# Case 0: no changes
	if nvals == new_nvals:
		return vals, probs
	# Case 1: just add padding
	elif nvals < new_nvals:
		pad = np.random.uniform(
			ymin, ymax, size=new_nvals - nvals
		)
		vals = np.concatenate([vals, pad], axis=0)
		probs = np.concatenate([probs, np.zeros(new_nvals - nvals)], axis=0)
		return vals, probs
	# Case 2: greedily merge support points
	else:
		probs = np.maximum(probs, 1e-8)
		probs /= probs.sum()
		while len(vals) > new_nvals:
			# Probability of equaling the left merge
			mixtures = probs[:-1] / (probs[:-1] + probs[1:])
			# candidate values for the post-merge support point
			candvals = vals[1:] * (1 - mixtures) + vals[:-1] * mixtures
			# E[|X - Y|] under best coupling
			costs = (candvals - vals[1:])**2 * probs[1:] 
			costs += (candvals - vals[:-1])**2 * probs[:-1]
			# pick the best variant
			istar = np.argmin(costs)
			vals[istar] = np.nan
			vals[istar + 1] = candvals[istar]
			probs[istar + 1] += probs[istar]
			probs[istar] = np.nan
			# drop na vals
			vals = vals[~np.isnan(vals)]
			probs = probs[~np.isnan(probs)]
		return vals, probs
	
def adjust_support_size(
	vals, probs, new_nvals, ymin, ymax
):
	"""
	Adjust categorical distribution to have support of size ``new_nvals``.
	
	Parameters
	----------
	vals : np.array
		(n, nvals)-length array of support. vals[i] must be sorted for each i.
	probs : np.array
		(n, nvals)-length array of probabilities of original distributions.
	new_nvals : int
		Desired size of support.
	ymin : float
		Minimum value for support.
	ymax : float
		Maximum value for support.

	Returns
	-------
	new_vals : np.array
		(n, new_nvals)-array of supports.
	new_probs : np.array
		(n, new_nvals)-array of probabilities.
	"""
	n, nvals = vals.shape
	new_vals = np.zeros((n, new_nvals))
	new_probs = np.zeros((n, new_nvals))
	for i in range(n):
		new_vals[i], new_probs[i] = _adjust_support_size_unbatched(
			vals[i], probs[i], new_nvals, ymin=ymin, ymax=ymax,
		)
	return new_vals, new_probs

def weighted_quantile(values, weights, quantiles, _old_style=True):
	"""
	Very close to numpy.percentile, but supports weights.

	Parameters
	----------
	values : np.array
		n-length array of data
	weights : np.array
		n-length array of sample weights
	quantiles : array-like
		d-length array of desired quantiles

	Returns
	-------
	results : np.array
		d-length array of weighted quantiles.
	"""
	if np.any(quantiles < 0) or np.any(quantiles > 1):
		raise ValueError("Quantiles must be in [0,1]")

	# Sort values
	sorter = np.argsort(values)
	values = values[sorter]
	weights = weights[sorter]
	# Compute quantiles
	cdf = np.cumsum(weights) - 0.5 * weights
	if _old_style:
		cdf -= cdf[0]
		cdf /= cdf[-1]
	else:
		cdf /= np.sum(weights)
	# Interp + return
	return np.interp(quantiles, cdf, values)

def parse_dist(
	dist, loc=0, scale=1, mu=None, sd=None, **kwargs
):
	## variant 1 based on mu/sd
	dist = dist.lower()
	if mu is not None and sd is not None and dist != 'bernoulli':

		# get SD correct
		temp = parse_dist(dist, loc=0, scale=sd, **kwargs)
		scale = sd / temp.std()
		temp = parse_dist(dist, loc=0, scale=sd*scale, **kwargs)
		shift = mu - temp.mean()
		return parse_dist(dist, loc=shift, scale=sd*scale, **kwargs)
	elif mu is not None and sd is not None and dist == 'bernoulli':
		# Note: Cannot enforce mean/sd for Bernoulli dist.
		# So we just use loc/scale.
		loc = copy.copy(mu)
		scale = copy.copy(sd)
		mu = None
		sd = None
	elif mu is not None or sd is not None:
		raise ValueError("Must provide both of mu and sd, not just one.")

	## variant 2 based on location/scale
	# sometimes return a regular dist object
	if not isinstance(dist, str):
		return dist

	# Parse
	if dist == 'constant':
		return ConstantDist(loc=loc, scale=scale)
	if dist == 'bernoulli':
		prob = np.exp(loc / scale)
		prob = prob / (1 + prob)
		return stats.bernoulli(p=prob)
	if dist in ['gaussian', 'normal']:
		return stats.norm(loc=loc, scale=scale)
	if dist == 'invchi2':
		df = kwargs.pop("df", 4)
		return stats.chi2(df=df, loc=loc, scale=scale/df)
	if dist == 't' or dist == 'tdist':
		df = kwargs.pop("df", 4)
		return stats.t(loc=loc, scale=scale, df=df)
	if dist == 'cauchy':
		return stats.cauchy(loc=loc, scale=scale)
	if dist == 'laplace':
		return stats.laplace(loc=loc, scale=scale)
	if dist in ['uniform', 'unif']:
		return stats.uniform(loc=loc-scale/2, scale=scale)
	if dist in ['expo', 'expon', 'exponential']:
		return stats.expon(loc=loc-scale, scale=scale)
	if dist == 'gamma':
		a = kwargs.pop("a", 5)
		return stats.gamma(loc=loc-a*scale, scale=scale, a=a)
	if dist == 'skewnorm':
		a = kwargs.pop("a", 5)
		return stats.skewnorm(loc=loc, scale=scale, a=a)
	if dist == 'powerlaw':
		a = kwargs.pop("a", 5)
		return stats.powerlaw(loc=loc, scale=scale, a=a)
	else:
		raise ValueError(f"Dist {dist} is unrecognized.")