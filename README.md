A python implementation of the dual bounds framework for inference on partially identified estimands (and the solutions to optimization problems more generally). See https://amspector100.github.io/dualbounds/ for detailed documentation and tutorials.

## Installation

To install ``dualbounds``, just use pip:

``pip install dualbounds``

## Documentation

Documentation and tutorials are available at https://amspector100.github.io/dualbounds/.

## Quickstart

Given a response vector ``y``, binary treatment vector ``W``, covariate matrix ``X``, and an (optional) propensity score vector ``pis``, dualbounds allows analysts to perform inference on partially identified estimands of the form E[f(Y(0), Y(1), X)], for any choice of f. For example, the code below shows how to perform inference in P(Y(0) < Y(1)), the proportion of individuals who benefit from the treatment. Dual bounds can wrap on top of *any* machine learning model to provide provably valid confidence intervals in randomized experiments.

```
	import dualbounds as db
	# Generate synthetic data from a heavy-tailed linear model
	data = db.gen_data.gen_regression_data(
		n=900, # Num. datapoints
		p=30, # Num. covariates
		r2=0.95, # population R^2
		tau=3, # average treatment effect
		interactions=True, # ensures treatment effect is heterogenous
		eps_dist='laplace', # heavy-tailed residuals
		sample_seed=123, # random seed
	)

	# Initialize dual bounds object
	dbnd = db.generic.DualBounds(
		f=lambda y0, y1, x: y0 < y1, # defines the estimand
		X=data['X'], # n x p covariate matrix
		W=data['W'], # n-length treatment vector
		y=data['y'], # n-length outcome vector
		pis=data['pis'], # n-length propensity scores (optional)
		Y_model='ridge', # description of model for Y | X, W
	)

	# Perform inference
	dbnd.compute_dual_bounds(
		nfolds=5, # number of cross-fitting folds
		alpha=0.05 # nominal level
	)

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