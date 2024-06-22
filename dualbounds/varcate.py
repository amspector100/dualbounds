"""
Methods for bounding Var(E[Y(1) - Y(0) | X]).
"""
import warnings
import numpy as np
from scipy import stats
from . import generic, utilities
from .generic import infer_discrete, get_default_model
import pandas as pd

def _moments2varcate(
	hxy1, hxy0, hx, y1, y0, shx2,
):
	"""
	Computes bound on the variance of the
	CATE from a set of marginal moments
	on E[h(X) Y(1)], E[h(X) Y(0)], E[h(X)],
	etc.
	"""
	return 2 * (
		hxy1 - hxy0 - hx * y1 + hx * y0 
	) - shx2 + hx**2

def varcate_delta_method_se(
	shxy1, shxy0, shx, sy1, sy0, shx2,
):
	"""
	Estimates and provides SE for 
	2 * Cov(h(X), Y(1) - Y(0)) - var(h(X)).

	Parameters
	----------
	shxy1 : np.array
		n-length array of AIPW summands for E[Y(1) h(X)].
	shxy0 : np.aray
		n-length array of AIPW summands for E[Y(0) h(X)].
	shx : np.array
		n-length array of AIPW summands for E[h(X)].
	sy1 : np.array
		n-length array of AIPW summands for E[Y(1)].
	sy0 : np.array
		n-length array of AIPW summands for E[Y(1)].
	shx2 : np.array
		n-length array of AIPW summands for E[h(X)^2].

	Returns
	-------
	estimate : float
		Plug-in estimate
	se : float
		Standard error
	"""
	# concatenate and estimate
	summands = np.stack(
		[shxy1, shxy0, shx, sy1, sy0, shx2], axis=1
	)
	mus = summands.mean(axis=0)
	hattheta = _moments2varcate(
		*tuple(list(mus))
	)
	hatSigma = np.cov(summands.T)
	grad = np.array([
		2,
		-2, 
		2 * (mus[4] - mus[3] + mus[2]),
		- 2 * mus[2],
		2 * mus[2],
		-1,
	])
	# estimate
	se = np.sqrt(grad @ hatSigma @ grad / len(shx))
	return hattheta, se

class VarCATEDualBounds(generic.DualBounds):
	"""
	Computes lower bounds on :math:`Var(E[Y(1) - Y(0) | X]).`

	This class has the same signature as ``generic.DualBounds``
	except it only provides lower bounds and the input ``f``
	will be ignored.
	"""

	def __init__(self, *args, **kwargs):
		# Initialize with no f function
		kwargs['f'] = None
		super().__init__(*args, **kwargs)

	def _ensure_feasibility(self):
		raise NotImplementedError()

	def _discretize(self):
		raise NotImplementedError()

	def _solve_single_instance(self):
		raise NotImplementedError()

	def _compute_realized_dual_variables(self):
		raise NotImplementedError()

	def compute_dual_variables(self, *args, **kwargs):
		""" 
		In this case, the optimal dual variables are simply 
		the estimated CATE, so this function does nothing.
		"""
		pass

	def _compute_ipw_summands(self):
		pass

	def diagnostics(self, plot=True, aipw=True):
		return None

	def results(self, minval: float=0, maxval: float=np.inf):
		# minval is always zero
		return super().results(minval=0, maxval=maxval)

	def _plug_in_results(self, B: int=1000):
		# Estimates
		pests = np.array([self.cates.std()**2, np.nan])
		# SEs via bootstrap
		inds = np.random.choice(self.n, size=(self.n, B), replace=True)
		bootstrap_varcates = self.cates[inds].std(axis=0)**2
		pses = np.array([bootstrap_varcates.std(), np.nan])
		pcis = pests - pses * stats.norm.ppf(1-self.alpha) 
		return pd.DataFrame(
			np.stack(
				[pests, pses, pcis], 
				axis=0
			),
			index=['Estimate', 'SE', 'Conf. Int.'],
			columns=['Lower', 'Upper']
		)

	def _compute_final_bounds(self, aipw=True, alpha=0.05):
		self._compute_cond_means()
		self.cates = self.mu1 - self.mu0 
		#### We have to use the 6-d delta method
		# The notation uses h(X) = hat E[Y(1) - Y(0) | X]
		# 1. AIPW terms for h(X) * Y(1)
		self.shxy1 = self.W * self.cates * (self.y - self.mu1)
		self.shxy1 = self.shxy1 / self.pis + self.cates * self.mu1
		# 2. AIPW terms for h(X) * Y(0)
		self.shxy0 = (1 - self.W) * self.cates * (self.y - self.mu0)
		self.shxy0 = self.shxy0 / (1 - self.pis) + self.cates * self.mu0
		# 3. AIPW terms for h(X)
		self.shx = self.cates 
		# 4. AIPW terms for Y(1)
		self.sy1 = self.W * (self.y - self.mu1) / self.pis + self.mu1
		# 5. AIPW terms for Y(0)
		self.sy0 = (1 - self.W ) * (self.y - self.mu0)
		self.sy0 = self.sy0 / (1 - self.pis) + self.mu0
		# 6. AIPW terms for h(X)^2
		self.shx2 = self.cates**2
		estimate, se = varcate_delta_method_se(
			shxy1=self.shxy1, 
			shxy0=self.shxy0, 
			shx=self.shx, 
			sy1=self.sy1, 
			sy0=self.sy0, 
			shx2=self.shx2,
		)
		scale = stats.norm.ppf(1-alpha)
		lower_ci = estimate - scale * se
		self.estimates = np.array([estimate, np.nan])
		self.ses = np.array([se, np.nan])
		self.cis = np.array([lower_ci, np.nan])

def varcate_cluster_bootstrap(
	varcate_objects: list[VarCATEDualBounds],
	aipw: bool=True,
	alpha: float=0.05,
	B: int=1000,
	verbose: bool=False,
):
	"""
	Combines evidence across multiple VarCATEDualBounds classes
	using a (clustered) bootstrap.

	Parameters
	----------
	varcate_objects : list
		A list of fit VarCATEDualBounds objects.
	alpha : float
		Nominal level, between 0 and 1.
	B : int
		Number of bootstrap replications.
	verbose : bool
		If True, prints a progress report.
	"""
	K = len(varcate_objects)
	# Initial estimates and ses
	estimates = np.stack([x.estimates for x in varcate_objects], axis=-1)[0] # K-length array
	ses = np.stack([x.ses for x in varcate_objects], axis=-1)[0] # K-length array
	# Create data: n x 2 x K
	data = np.stack(
		[np.stack([vdb.cates, vdb.sy1 - vdb.sy0], axis=1) for vdb in varcate_objects],
		axis=2
	)
	# Function which maps data to maximum VarCATE estimate
	def compute_estimates(data):
		K = data.shape[-1]
		ests = np.zeros(K)
		for k in range(K):
			cov_truecate = np.cov(data[:, 0, k], data[:, 1, k])[0,1]
			var_estcate = np.std(data[:, 0, k])**2
			ests[k] = 2 * cov_truecate - var_estcate
		return ests
	# Cluster bootstrap
	_, bs_estimators = utilities.cluster_bootstrap_se(
		data=data,
		clusters=varcate_objects[0].clusters,
		func=compute_estimates,
		B=B,
		verbose=verbose,
	) # B x K
	# Centered BS estimators and quantile
	centered = (bs_estimators - estimates) / ses
	hatq = np.quantile(centered.max(axis=-1), 1-alpha)
	# Take maximums
	combined_estimates = np.maximum(np.array([np.max(estimates), np.nan]), 0)
	cis = np.maximum(np.array([np.max(estimates - hatq * ses), np.nan]), 0)
	# Return
	return pd.DataFrame(
		np.stack([combined_estimates, cis], axis=0),
		index=['Estimate', 'Conf. Int.'],
		columns=['Lower', 'Upper'],
	)
