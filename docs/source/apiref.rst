API Reference
=============

Generic DualBounds
==================
.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ~dualbounds.generic.DualBounds
   ~dualbounds.delta.DeltaDualBounds

   :template:
   ~dualbounds.generic.plug_in_no_covariates

Bootstrap methods
=================
.. autosummary::
   :toctree: generated/
   :template:

   ~dualbounds.bootstrap.dualbound_multiplier_bootstrap
   ~dualbounds.bootstrap.multiplier_bootstrap

Lee bounds
==========
.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ~dualbounds.lee.LeeDualBounds

   :template:
   ~dualbounds.lee.lee_bound_no_covariates
   ~dualbounds.lee.compute_analytical_lee_bound
   ~dualbounds.lee.compute_cvar

Variance bounds
===============
.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ~dualbounds.varite.VarITEDualBounds
   ~dualbounds.varcate.VarCATEDualBounds

Utility functions
=================

Interpolation
-------------
.. autosummary::
   :toctree: generated/
   
   :template:
   ~dualbounds.interpolation.adaptive_interpolate
   ~dualbounds.interpolation.nn_interpolate
   ~dualbounds.interpolation.linear_interpolate

Miscallaneous
-------------
.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ~dualbounds.utilities.BatchedCategorical

   :template:
   ~dualbounds.utilities.compute_est_bounds
   ~dualbounds.utilities.weighted_quantile
   ~dualbounds.utilities.adjust_support_size
   ~dualbounds.utilities.apply_pool_factorial
