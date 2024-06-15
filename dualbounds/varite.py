"""
Methods for bounding Var(E[Y(1) - Y(0)]), the
variance of the individual treatment effect (ITE).
"""
import numpy as np
from scipy import stats
from .generic import DualBounds

def compute_analytical_varite_bound(
	n,
	y0_dists,
	y1_dists,
	reps=100,
):
	"""
	Semi-analytical bounds on :math:`Var(Y(0) - Y(1))`.

	Unlike dual bounds, this function is not
	robust to model misspecification.
	
	Parameters
	----------
	n : int
		Number of observations.
	y0_dists : stats.rv_continuous / stats.rv_discrete
		batched scipy distribution of shape (n,) where the ith
		distribution is the conditional law of Yi(0) | Xi
	y1_dists : stats.rv_continuous / stats.rv_discrete
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

def varite_delta_method_se(
	sbetas, skappa1s, skappa0s
):
	# estimate
	hat_beta = sbetas.mean()
	hat_kappa1 = skappa1s.mean()
	hat_kappa0 = skappa0s.mean()
	ate = hat_kappa1 - hat_kappa0 # average treatment effect
	hattheta = hat_beta - ate**2
	# standard error
	hatSigma = np.cov(
		np.stack([sbetas, skappa1s, skappa0s], axis=0)
	) # 3 x 3 cov matrix
	grad = np.array(
		[1, - 2 * ate, 2 * ate]
	)
	# estimate
	se = np.sqrt(grad @ hatSigma @ grad / len(sbetas))
	return hattheta, se

class VarITEDualBounds(DualBounds):
	"""
	Computes dual bounds on :math:`Var(Y(1) - Y(0)).`

	The signature of this class is identical to 
	the ``generic.DualBounds`` class.  However, 
	the input ``f`` will be ignored.
	"""

	def __init__(self, *args, **kwargs):
		# Initialize with correct f function
		kwargs['f'] = lambda y0, y1, x: (y0-y1)**2
		super().__init__(*args, **kwargs)

	def results(self, minval: float=0, maxval: float=np.inf):
		# minval is always zero
		return super().results(minval=0, maxval=maxval)

	def _compute_final_bounds(self, aipw=True, alpha=0.05):
		"""
		Computes final bounds based in (A)IPW summands,
		using the delta method for Var(Y(1) - Y(0)).
		"""
		self._compute_ipw_summands()
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
			# kappa1 = E[Y(1)], kappa0 = E[Y(0)] are ident components
			skappa1s = self.W * (self.y - self.mu1) / self.pis + self.mu1
			skappa0s = (1-self.W) * (self.y - self.mu0) / (1-self.pis) + self.mu0
			# estimate
			hattheta, se = varite_delta_method_se(
				sbetas=sbetas, skappa1s=skappa1s, skappa0s=skappa0s
			)
			ests.append(hattheta)
			ses.append(se)
			if lower:
				bounds.append(hattheta - scale * se)
			else:
				bounds.append(hattheta + scale * se)

		self.estimates = np.array(ests)
		self.ses = np.array(ses)
		self.cis = np.array(bounds)
		return dict(
			estimates=self.estimates,
			ses=self.ses,
			cis=self.cis
		)