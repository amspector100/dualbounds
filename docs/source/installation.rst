Getting Started
===============

Installation
------------

To install dualbounds, just use pip:

``pip install dualbounds``

Quickstart
----------

Given a data-matrix ``X``, a treatment vector ``W``, (optional) propensity scores ``pis``, and a response vector ``y``, ``dualbounds`` can compute bounds on partially identified causal estimands. Below, we show how to bound the proportion of individuals whose individual treatment effect is positive. By default, ``dualbounds`` uses cross-validated (Logistic) Ridge regressions with treatment-covariate interactions as the outcome model, although the user can instead supply custom machine learning models (see ``usage``).

.. code-block:: python

	import dualbounds as db
	from db.generic import DualBounds

	# Generate synthetic data from a heavy-tailed linear model
	data = db.gen_data.gen_regression_data(
		n=500, # Number of datapoints
		p=500, # Dimensionality
		r2=0.6, # population R^2
		tau=1, # treatment effect
		eps_dist='laplace', # heavy-tailed epsilon
	)