import os
import sys

# Add path to allow import of code
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory))

# Import the actual stuff
import dualbounds

# for profiling
import inspect

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