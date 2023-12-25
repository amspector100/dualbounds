"""
Methods for bounding Var(E[Y(1) - Y(0) | X]).
"""
import warnings
import numpy as np
from scipy import stats
from . import generic
from .generic import infer_discrete, get_default_model

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
	Class for computing lower bounds on 
	
	Var(E[Y(1) - Y(0) | X]).

	This class has the same signature as 
	``generic.DualBounds`` except it only
	provides lower bounds and the input
	``f`` will be ignored.
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
		the estimated CATE.
		"""
		pass

	def _compute_ipw_summands(self):
		pass

	def compute_final_bounds(self, aipw=True, alpha=0.05):
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
		self.estimate, self.se = varcate_delta_method_se(
			shxy1=self.shxy1, 
			shxy0=self.shxy0, 
			shx=self.shx, 
			sy1=self.sy1, 
			sy0=self.sy0, 
			shx2=self.shx2,
		)
		scale = stats.norm.ppf(1-alpha)
		self.lower_ci = self.estimate - scale * self.se
		return dict(
			estimate=self.estimate,
			se=self.se,
			lower_ci=self.lower_ci
		)