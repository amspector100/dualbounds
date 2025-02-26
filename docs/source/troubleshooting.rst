How can I choose the outcome model?
===================================

See the :doc:`Model selection <tutorials/model_selection>` page. In randomized experiments, we *strongly* recommend using the multiplier bootstrap. 

Large standard errors
=====================

Occasionally, ``DualBounds`` will give very large standard errors. Some possible fixes:

1. Run the ``DualBounds`` diagnostics (see :doc:`Diagnostics <tutorials/diagnostics>`). 
2. Often, this happens when the outcome model makes a few very inaccurate predictions. To avoid this, ensure you have scaled X appropriately/clip outcome model predictions. 
3. Set ``dual_solver='se'`` within ``DualBounds.fit()``. This will solve an optimization problem which makes the upper/lower confidence bounds as informative as possible. This is more computationally expensive, however.

Numerical stability
===================

If ``DualBounds`` is numerically unstable, try setting ``dual_solver='se'`` within ``DualBounds.fit()``. If you are using a support restriction, see the section below.


Incorporating structural constraints is slow
============================================

Incorporating structural constraints is typically slow when the outcome model you fit is not compatible with the support restrictions.

For example, if you assert via a support restriction that :math:`Y(0) \leq Y(1)`, the estimated law of :math:`Y(1) \mid X` should stochastically dominates that of :math:`Y(0) \mid X`.

See also :doc:`support restrictions <tutorials/support_restrictions>`.

Strange results
===============

Occasionally, ``DualBounds`` will return nonsensical estimates. For example, the lower bound on a probability may be occasionally negative. This is because ``DualBounds`` estimators are asymptotically normal, possibly conservative sample means.

For example, if the population lower bound is 0, the ``DualBounds`` estimator will be distributed as :math:`N(\mu,\sigma^2)` for some :math:`\mu \le 0`. As the outcome model gets better, :math:`\mu` will get closer to the true lower bound (zero). However, this means that dual lower bound will have a :math:`50\%` chance of being negative.

To avoid this problem, we have two possible solutions:

1. Perform model selection to fit a better outcome model (improved the out-of-sample :math:`R^2`).
2. Reduce the standard error (see the "standard errors" section above). We have never seen a ``DualBounds`` return a nonsensical estimate when using ``dual_solver=se``. However, this method is computationally expensive.

If this problem will not go away, this often means that the parameter in your problem is fully unidentified, or at the very least, the true partial identification bound is essentially uninformative. The only way to get sharper confidence intervals is to make more assumptions.