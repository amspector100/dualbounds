"""
Methods for bounding Var(E[Y(1) - Y(0) | X]).
"""
import warnings
import numpy as np
from scipy import stats
from . import generic, utilities
from .generic import infer_discrete
import pandas as pd
import copy
from typing import Optional

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
		# change default eps_dist for cts to Gaussian
		# (eps_dist has zero impact on the final results here 
		# and Gaussian is the most computationally efficient)
		kwargs['eps_dist'] = 'gaussian'
		# Initialize
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
	optimal_index = np.argmax(estimates - hatq * ses)
	combined_estimates = np.maximum(np.array([estimates[optimal_index], np.nan]), 0)
	final_ses = np.array([ses[optimal_index], np.nan])
	cis = np.maximum(np.array([(estimates - hatq * ses)[optimal_index], np.nan]), 0)
	# Return
	return pd.DataFrame(
		np.stack([combined_estimates, final_ses, cis], axis=0),
		index=['Estimate', 'SE', 'Conf. Int.'],
		columns=['Lower', 'Upper'],
	)

DEFAULT_SHRINKAGE_VALUES = np.array([0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.5, 2])

class CalibratedVarCATEDualBounds(VarCATEDualBounds):
	"""
	Improved lower bounds on :math:`Var(E[Y(1) - Y(0) | X])`.

	This has the same signature as ``varcate.VarCATEDualBounds`` 
	except it takes one additional argument. 

	Parameters
	----------
	shrinkages : np.array
		Array of shrinkage values s, where the CATEs are replaced with
		(1-s) * CATE + s * ATE. 

	Notes
	-----
	This class is identical to ``varcate.VarCATEDualBounds``
	except it selects a shrinkage parameter which shrinks
	estimated CATEs towards the ATE. Also, if ``outcome_model`` is a list,
	it uses the multiplier bootstrap to perform model selection. 

	This class should not be used for observational studies.
	"""
	def __init__(self, *args, shrinkages=DEFAULT_SHRINKAGE_VALUES, **kwargs):
		self.shrinkages = shrinkages
		# initialize outcome models
		self.outcome_models = kwargs.pop("outcome_model", ['ridge'])
		if not isinstance(self.outcome_models, list):
			self.outcome_models = [self.outcome_models]
		# create vdbs
		self.vdbs = []
		for outcome_model in self.outcome_models:
			self.vdbs.append(
				VarCATEDualBounds(*args, **kwargs, outcome_model=outcome_model)
			)

	def fit(self, B: int=1000, **kwargs):
		# Fit all models
		verbose = kwargs.get("verbose", True)
		counter = 0
		for outcome_model, vdb in zip(self.outcome_models, self.vdbs):
			if verbose and len(self.vdbs) > 1:
				print(f"Fitting model={outcome_model}, num. {counter} out of {len(self.vdbs)}.")
				counter += 1
			vdb.fit(**kwargs)
		
		# Run multiplier bootstrap at various shrinkages
		self.shrunk_vdb_objects = []
		for vdb in self.vdbs:	
			hatate = np.mean(
				vdb.mu1 - vdb.mu0
				+ vdb.W*(vdb.y - vdb.mu1)/vdb.pis
				- (1-vdb.W) * (vdb.y - vdb.mu0)/ (1-vdb.pis)
			)
			mu0_homogeneous = (vdb.mu0 + vdb.mu1) / 2 - hatate/2
			mu1_homogeneous = (vdb.mu0 + vdb.mu1) / 2 + hatate/2
			aipw = vdb.sy1 - vdb.sy0
			for s in self.shrinkages:
				new_mu0 = vdb.mu0 * s + (1 - s) * mu0_homogeneous
				new_mu1 = vdb.mu1 * s + (1 - s) * mu1_homogeneous
				new_vdb = VarCATEDualBounds(
					outcome=vdb.y,
					treatment=vdb.W,
					propensities=vdb.pis,
				).fit(
					y0_dists=stats.norm(loc=new_mu0),
					y1_dists=stats.norm(loc=new_mu1),
					suppress_warning=True,
				)
				self.shrunk_vdb_objects.append(new_vdb)

		# Cluster bootstrap
		if verbose:
			print("Fitting cluster bootstrap to aggregate results.")
		results = varcate_cluster_bootstrap(
			self.shrunk_vdb_objects, B=B, verbose=verbose, alpha=self.vdbs[0].alpha
		)
		self.estimates = results.loc['Estimate'].values
		self.ses = results.loc['SE'].values
		self.cis = results.loc['Conf. Int.'].values
		return self


	def summary(self, minval: Optional[float]=0, maxval: Optional[float]=np.inf):
		print("___________________Inference_____________________")
		print(self.results(minval=minval, maxval=maxval))
		print()
		print("_________________Outcome models__________________")
		# Loop through and print model diagnostics
		for outcome_model, vdb in zip(self.outcome_models, self.vdbs):
			print(f"Outcome model: {outcome_model}")
			print(vdb.eval_outcome_model())

