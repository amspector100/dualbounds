Generic DualBounds
==================
.. autosummary::
   :toctree: generated/generic/
   :template: autosummary/class.rst

   ~dualbounds.generic.DualBounds
   ~dualbounds.delta.DeltaDualBounds

   :template:
   ~dualbounds.generic.plug_in_no_covariates

Bootstrap methods
=================
.. autosummary::
   :toctree: generated/bootstrap/
   :template:

   ~dualbounds.bootstrap.dualbound_multiplier_bootstrap
   ~dualbounds.bootstrap.multiplier_bootstrap

Lee bounds
==========
.. autosummary::
   :toctree: generated/lee/
   :template: autosummary/class.rst

   ~dualbounds.lee.LeeDualBounds

   :template:
   ~dualbounds.lee.lee_bound_no_covariates
   ~dualbounds.lee.compute_analytical_lee_bound

Variance bounds
===============
.. autosummary::
   :toctree: generated/varbounds/
   :template: autosummary/class.rst

   ~dualbounds.varite.VarITEDualBounds
   ~dualbounds.varcate.VarCATEDualBounds
   :exclude-members: diagnostics

Distributional regression
=========================

.. autosummary::
   :toctree: generated/dist_reg/
   :template: autosummary/class.rst

   ~dualbounds.dist_reg.DistReg
   ~dualbounds.dist_reg.BinaryDistReg
   ~dualbounds.dist_reg.CtsDistReg
   ~dualbounds.dist_reg.QuantileDistReg
   ~dualbounds.dist_reg.MonotoneLogisticReg
   ~dualbounds.dist_reg.ModelSelector
   ~dualbounds.dist_reg.cross_fit_predictions

Utility functions
=================

Synthetic data generation
-------------------------
.. autosummary::
   :toctree: generated/gen_data/
   
   :template:
   ~dualbounds.gen_data.gen_regression_data
   ~dualbounds.gen_data.gen_lee_bound_data


Interpolation
-------------
.. autosummary::
   :toctree: generated/interp/
   
   :template:
   ~dualbounds.interpolation.adaptive_interpolate
   ~dualbounds.interpolation.nn_interpolate
   ~dualbounds.interpolation.linear_interpolate

Miscellaneous
-------------
.. autosummary::
   :toctree: generated/misc/
   :template: autosummary/class.rst

   ~dualbounds.utilities.BatchedCategorical

   :template:
   ~dualbounds.utilities.compute_est_bounds
   ~dualbounds.utilities.weighted_quantile
   ~dualbounds.utilities.adjust_support_size