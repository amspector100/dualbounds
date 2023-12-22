import numpy as np
#import cvxpy as cp
from scipy import stats
from .generic import DualBounds

def compute_analytical_varite_bound(
	n,
	y0_dists,
	y1_dists,
	reps=100,
):
	"""
	Parameters
	----------
	n : int
		Number of observations.
	y0_dists : np.array
		batched scipy distribution of shape (n,) where the ith
		distribution is the conditional law of Yi(0) | Xi
	y1_dists : np.array
		batched scipy distribution of shape (n,) where the ith
		distribution is the conditional law of Yi(1) | Xi
	reps : int
		Number of samples to take from each distribution.

	Returns
	-------
	lower : float
		Lower bound on Var(Y(1) - Y(0))
	upper : float
		Upper bound on Var(Y(1) - Y(0))
	"""
	# Sample coupled r.v.s
	U = np.random.uniform(size=(reps, n))
	y1 = y1_dists.ppf(U)
	y0l = y0_dists.ppf(U)
	y0u = y0_dists.ppf(1-U)
	# Evaluate bounds
	lower = np.std(y1-y0l)**2
	upper = np.std(y1 - y0u)**2
	return lower, upper

class VarITEDualBounds(DualBounds):
	"""
	Computes dual bounsd on Var(Y(1) - Y(0)).

	The input/output signature of this class is identical
	to the ``DualBounds`` class. 
	However, the input ``f`` will be ignored.
	"""

	def __init__(self, *args, **kwargs):
		# Initialize with correct f function
		kwargs['f'] = lambda y0, y1, x: (y0-y1)**2
		super().__init__(*args, **kwargs)

	def compute_final_bounds(self, aipw=True, alpha=0.05):
		"""
		Computes final bounds based in (A)IPW summands,
		using the delta method for Var(Y(1) - Y(0)).
		"""
		self.compute_ipw_summands()
		summands = self.aipw_summands if aipw else self.ipw_summands
		self._compute_cond_means()
		# Note: the notation follows Appendix A.2 of 
		# https://arxiv.org/pdf/2310.08115.pdf (version 1)
		ests = []
		ses = []
		bounds = []
		scale = stats.norm.ppf(1-alpha/2)
		for lower in [1, 0]:
			# part. identifiable component
			sbetas = summands[1-lower]
			hat_beta = sbetas.mean()
			# kappa1 = E[Y(1)], kappa0 = E[Y(0)] are ident components
			skappa1s = self.W * (self.y - self.mu1) / self.pis + self.mu1
			hat_kappa1 = skappa1s.mean()
			skappa0s = (1-self.W) * (self.y - self.mu0) / (1-self.pis) + self.mu0
			hat_kappa0 = skappa0s.mean()
			ate = hat_kappa1 - hat_kappa0 # average treatment effect
			# estimate
			hattheta = hat_beta - ate**2; ests.append(hattheta)
			# standard error
			hatSigma = np.cov(
				np.stack([sbetas, skappa1s, skappa0s], axis=0)
			) # 3 x 3 cov matrix
			grad = np.array(
				[1, - 2 * ate, 2 * ate]
			)
			# estimate
			se = np.sqrt(grad @ hatSigma @ grad / len(sbetas))
			ses.append(se)
			if lower:
				bounds.append(hattheta - scale * se)
			else:
				bounds.append(hattheta + scale * se)

		self.ests = np.maximum(np.array(ests), 0)
		self.ses = np.array(ses)
		self.bounds = np.maximum(np.array(bounds), 0)
		return self.ests, self.bounds