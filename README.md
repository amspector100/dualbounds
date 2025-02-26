A python implementation of the dual bounds framework for inference on partially identified estimands (and the solutions to optimization problems more generally). See https://dualbounds.readthedocs.io/en/latest/ for detailed documentation and tutorials.

## Installation

To install ``dualbounds``, just use pip:

``pip install dualbounds``

## Documentation

Documentation and tutorials are available at https://dualbounds.readthedocs.io/en/latest/.

## Quickstart

Given a response vector ``y``, binary treatment vector ``W``, covariate matrix ``X``, and an (optional) propensity score vector ``pis``, dualbounds allows analysts to perform inference on partially identified estimands of the form `E[f(Y(0), Y(1), X)]`, for any choice of f. For example, the code below shows how to perform inference on `P(Y(0) < Y(1))`, the proportion of individuals who benefit from the treatment. Dual bounds can wrap on top of *any* machine learning model to provide provably valid confidence intervals in randomized experiments.

```
	import dualbounds as db
	from dualbounds.generic import DualBounds

	# Generate synthetic data
	data = db.gen_data.gen_regression_data(n=900, p=30, sample_seed=123)

	# Initialize dual bounds object
	dbnd = DualBounds(
	    f=lambda y0, y1, x: y0 < y1,
	    covariates=data['X'],
	    treatment=data['W'],
	    outcome=data['y'],
	    propensities=data['pis'],
	    outcome_model='ridge',
	)

	# Compute dual bounds and observe output
	results = dbnd.fit(alpha=0.05).results()


```

## Reference

If you use ``dualbounds`` in an academic publication, please consider citing [Ji, Lei, and Spector (2023)](https://arxiv.org/abs/2310.08115). The bibtex is below:

```
@misc{ji2023modelagnostic,
      title={Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects}, 
      author={Wenlong Ji and Lihua Lei and Asher Spector},
      year={2023},
      eprint={2310.08115},
      archivePrefix={arXiv},
      primaryClass={econ.EM}
}
```