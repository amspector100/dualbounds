import os
import sys
import numpy as np

# Add path to allow import of code
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory))

# Import the actual stuff
import dualbounds

# for profiling
import inspect

def _run_se_computation_test(
	dim,
	f,
	se_function,
	arg_names,
	n=500,
	reps=10000,
	testname='',
	param_shift=0,
	decimal=2,
):
	"""
	Generic function to test whether a delta method computation
	is correct.

	dim : int
		Number of parameters
	f : function
		f(*params) = target of inference 
	se_function : function
		function which computes SES given vectors of data
	arg_names : list
		List of length dim of input names for se_function 
	n : int
		Number of data-points per dataset
	reps : int
		Number of datasets
	shift : float
		params ~ N(param_shift, 1).
	decimal : positive float
		Number of decimals to ensure accuracy
	"""
	np.random.seed(123)
	# Create parameters
	mu = np.random.randn(dim) + param_shift
	sigmas = np.random.uniform(0.1, 0.9, size=dim)
	theta = f(*mu)
	# sample data
	data = np.random.randn(dim, reps, n) #+ mu.reshape(-1, 1, 1)
	data *= sigmas.reshape(-1, 1, 1)
	data += mu.reshape(-1, 1, 1)
	# Loop through and compute estimators
	ests = np.zeros(reps); ses = np.zeros(reps)
	for r in range(reps):
		kwargs = {
			arg_names[d]:data[d, r] for d in range(dim)
		}
		est, se = se_function(**kwargs)
		ests[r] = est
		ses[r] = se
	# test approximate unbiasedness
	np.testing.assert_array_almost_equal(
		ests.mean(),
		theta,
		decimal=decimal,
		err_msg=f"{testname} plug-in estimator is not unbiased"
	)
	# test SE estimation
	np.testing.assert_array_almost_equal(
		ests.std(),
		ses.mean(),
		decimal=decimal,
		err_msg=f"{testname} SE estimator is not consistent"
	)

def run_all_tests(test_classes):
	"""
	Usage: 
	context.run_all_tests(
		[TestClass(), TestClass2()]
	)
	This is useful for making pytest play nice with cprofilev.
	"""
	def is_test(method):
		return str(method).split(".")[1][0:4] == 'test'
	for c in test_classes:
		attrs = [getattr(c, name) for name in c.__dir__()]
		test_methods = [
			x for x in attrs if inspect.ismethod(x) and is_test(x)
		]
		for method in test_methods:
			method()