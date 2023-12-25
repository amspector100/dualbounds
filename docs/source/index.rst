========================================
Welcome to the dualbounds documentation!
========================================

Dual bounds are a powerful tool to perform inference on partially identified causal estimands (and certain identified causal estimands as well). In randomized experiments, the dual bounds framework can wrap around any machine learning algorithm to provide provably valid inference, even if the underlying machine learning model is misspecified or inconsistent. See, e.g., `Ji et al 2023`_ for more details.

.. _Ji et al 2023: https://arxiv.org/abs/2310.08115

``dualbounds`` is a python pacakge which makes it easy to compute dual bounds in only a few lines of code. ``dualbounds`` is built to be flexible, so analysts can input their own custom machine learning models. See usage for more details!

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   usage
   apiref


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`