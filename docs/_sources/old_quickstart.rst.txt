Quickstart
==========

The main class in the package is ``dualbounds.generic.DualBounds``, which computes dual bounds on a partially identified estimand of the form

E[f(Y(0), Y(1), X)].

As an input, it takes a data-matrix ``X``, a treatment vector ``W``, (optional) propensity scores ``pis``, and a response vector ``y``.

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