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

### For processing data
def _binarize_variable(x: np.array, var_name: str):
	"""
	Converts x to a binary r.v. and raises an error
	if it contains more than 2 values.
	"""
	# Check if nans
	if np.any(np.isnan(x)):
		raise ValueError(f"{var_name} has missing values.")

	# Check if binary
	vals = set(list(np.unique(x)))
	if len(vals) > 2:
		raise ValueError(f"{var_name} is not binary and takes {len(vals)} values.")
	elif len(vals - set([0,1])) == 0:
		return x.astype(int)
	else:
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

### Multiprocessing helper
def _one_arg_function(list_of_inputs, args, func, kwargs):
	"""
	Globally-defined helper function for pickling in multiprocessing.
	:param list of inputs: List of inputs to a function
	:param args: Names/args for those inputs
	:param func: A function
	:param kwargs: Other kwargs to pass to the function. 
	"""
	new_kwargs = {}
	for i, inp in enumerate(list_of_inputs):
		new_kwargs[args[i]] = inp
	return func(**new_kwargs, **kwargs)

def apply_pool_factorial(
	func, 
	constant_inputs={}, 
	num_processes=1, 
	**kwargs
):
	"""
	Spawns num_processes processes to apply func to many different arguments.
	This wraps the multiprocessing.pool object plus the functools partial function. 
	
	Parameters
	----------
	func : function
		An arbitrary function
	constant_inputs : dictionary
		A dictionary of arguments to func which do not change in each
		of the processes spawned, defaults to {}.
	num_processes : int
		The maximum number of processes spawned, defaults to 1.
	kwargs : dict
		Each key should correspond to an argument to func and should
		map to a list of different arguments.

	Returns
	-------
	outputs : list
		List of outputs for each input, in the order of the inputs.
	
	Examples
	--------
	If we are varying inputs 'a' and 'b', we might have
		>> apply_pool_factorial(func=my_func, a=[1,2], b=[5])

	which would return ``[my_func(a=1, b=5), my_func(a=2,b=5)]``.
	"""
	# Construct input sequence 
	args = sorted(kwargs.keys())
	kwarg_prod = list(product(*[kwargs[x] for x in args]))
	# Prepare to send this to apply pool
	final_kwargs = {}
	for i, arg in enumerate(args):
		final_kwargs[arg] = [k[i] for k in kwarg_prod]
	return apply_pool(
		func=func, 
		constant_inputs=constant_inputs,
		num_processes=num_processes,
		**final_kwargs
	)


def apply_pool(func, constant_inputs={}, num_processes=1, **kwargs):
	"""
	Spawns num_processes processes to apply func to many different arguments.
	This wraps the multiprocessing.pool object plus the functools partial function. 
	
	Parameters
	----------
	func : function
		An arbitrary function
	constant_inputs : dictionary
		A dictionary of arguments to func which do not change in each
		of the processes spawned, defaults to {}.
	num_processes : int
		The maximum number of processes spawned, defaults to 1.
	kwargs : dict
		Each key should correspond to an argument to func and should
		map to a list of different arguments.
	Returns
	-------
	outputs : list
		List of outputs for each input, in the order of the inputs.
	Examples
	--------
	If we are varying inputs 'a' and 'b', we might have
		>> apply_pool(func=my_func, a=[1,3,5], b=[2,4,6])

	which returns ``[my_func(a=1, b=2), my_func(a=3,b=4), my_func(a=5,b=6)]``.
	"""

	# Construct input sequence
	args = sorted(kwargs.keys())
	num_inputs = len(kwargs[args[0]])
	for arg in args:
		if len(kwargs[arg]) != num_inputs:
			raise ValueError(f"Number of inputs differs for {args[0]} and {arg}")
	inputs = [[] for _ in range(num_inputs)]
	for arg in args:
		for j in range(num_inputs):
			inputs[j].append(kwargs[arg][j])

	# Construct partial function
	partial_func = partial(
		_one_arg_function, args=args, func=func, kwargs=constant_inputs,
	)

	# Don't use the pool object if num_processes=1
	num_processes = min(num_processes, len(inputs))
	if num_processes == 1:
		all_outputs = []
		for inp in inputs:
			all_outputs.append(partial_func(inp))
	else:
		with Pool(num_processes) as thepool:
			all_outputs = thepool.map(partial_func, inputs)

	return all_outputs

def compute_est_bounds(summands, alpha=0.05):
	"""
	Helper function to compute confidence intervals.

	Parameters
	----------
	summands : np.array
		(2, n)-shaped array

	Returns
	-------
	ests : np.array
		2-shaped array of lower and upper estimators (sample mean).
	ses : np.array
		2-shaped array of standard errors.
	bounds : np.array
		2-shaped array of lower/upper confidence bounds.
	"""
	ests = summands.mean(axis=1)
	ses = summands.std(axis=1) / np.sqrt(summands.shape[1])
	scale = stats.norm.ppf(1-alpha/2)
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


def _convert_to_cat(bern_dist, n):
	"""
	Convert bernoulli dist. object to BatchedCategorical
	"""
	vals = np.zeros((n, 2)); vals[:, 1] += 1
	probs = bern_dist.mean()
	return BatchedCategorical(
		vals=vals, probs=np.stack([1-probs, probs], axis=1)
	)

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
	if mu is not None and sd is not None:
		if dist == 'bernoulli':
			raise ValueError("Cannot enforce mean/sd for Bernoulli dist.")
		# get SD correct
		temp = parse_dist(dist, loc=0, scale=sd, **kwargs)
		scale = sd / temp.std()
		temp = parse_dist(dist, loc=0, scale=sd*scale, **kwargs)
		shift = mu - temp.mean()
		return parse_dist(dist, loc=shift, scale=sd*scale, **kwargs)

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