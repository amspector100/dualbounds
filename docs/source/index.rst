========================================
Welcome to the dualbounds documentation!
========================================

Dual bounds is a method to perform inference on partially identified causal estimands (and certain identified estimands as well). In randomized experiments, dual bounds can wrap around any machine learning algorithm to provide provably valid inference, even if the underlying ML model is misspecified or inconsistent. When applied to observational data, dual bounds also have strong double-robustness guarantees. See, e.g., `Ji et al 2023`_ for more details.

.. _Ji et al 2023: https://arxiv.org/abs/2310.08115

``dualbounds`` is a python package for computing dual bounds. It is also designed to flexibly wrap around custom ML models. See the tutorials for more details.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   quickstart
   tutorials
   apiref


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
