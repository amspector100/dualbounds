========================================
Welcome to the dualbounds documentation!
========================================

Dual bounds are method to perform inference on partially identified causal estimands (and certain identified causal estimands as well). In randomized experiments, the dual bounds framework can wrap around any machine learning algorithm to provide provably valid inference, even if the underlying machine learning model is misspecified or inconsistent. When applied to observational data, dual bounds also have strong double-robustness guarantees. See, e.g., `Ji et al 2023`_ for more details.

.. _Ji et al 2023: https://arxiv.org/abs/2310.08115

``dualbounds`` is a python package which allows computation of dual bounds in 2-3 lines of code. ``dualbounds`` is built to be flexible, so analysts can input their own custom machine learning models. See the tutorials for more details.

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
