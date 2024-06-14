"""
Contains various interpolation functions used to 
go from discrete dual variables to continuous ones.
"""

import numpy as np
import scipy as sp
from . import utilities

def adaptive_interpolate(x: np.array, y: np.array, newx: np.array):
	"""
	Adaptively chooses between linear and nearest-neighbor
	interpolation.

	Parameters
	----------
	x : np.array
		n-length array of inputs. Must be sorted, although
		this is not explicitly enforced to save time.
	y : np.array
		n-length array of outputs
	newx : np.array
		m-length array of new inputs

	Returns
	-------
	newy : np.array
		m-length array of interpolated outputs
	"""
	if len(np.unique(y)) <= 2 and len(y) > 2:
		return nn_interpolate(x, y, newx)
	else:
		return linear_interpolate(x, y, newx)


def nn_interpolate(x: np.array, y: np.array, newx: np.array):
	"""
	Nearest-neighbor interpolation.

	Parameters
	----------
	x : np.array
		n-length array of inputs. Must be sorted, although
		this is not explicitly enforced to save time.
	y : np.array
		n-length array of outputs
	newx : np.array
		m-length array of new inputs

	Returns
	-------
	newy : np.array
		m-length array of interpolated outputs
	"""
	# Find nearest neighbors
	if not utilities.haslength(newx):
		newx = np.array([newx])
	n = len(x)
	rinds = np.minimum(np.searchsorted(x, newx, side='left'), n-1)
	linds = np.maximum(rinds-1, 0)
	inds = np.stack([linds, rinds], axis=1)
	dists = np.abs(x[inds] - newx.reshape(-1, 1))
	nbrs = inds[(np.arange(len(newx)), np.argmin(dists, axis=1))]
	# Return
	return y[nbrs]


def linear_interpolate(x: np.array, y: np.array, newx: np.array):
	"""
	Linear interpolation.

	Parameters
	----------
	x : np.array
		n-length array of inputs. Must be sorted, although
		this is not explicitly enforced to save time.
	y : np.array
		n-length array of outputs
	newx : np.array
		m-length array of new inputs

	Returns
	-------
	newy : np.array
		m-length array of interpolated outputs
	"""
	if not utilities.haslength(newx):
		newx = np.array([newx])
	# for now, check sorting (TODO DELETE)
	# if np.any(np.sort(x) != x):
	# 	raise ValueError("NOT SORTED")
	# interpolate points in the range of x
	haty = np.interp(newx, x, y)
	# Check if there are any points outside the boundaries
	lflags = newx < x[0]
	uflags = newx > x[-1]
	if not (np.any(lflags) or np.any(uflags)):
		return haty

	# Prevent div by 0 errors/warnings
	x, index = np.unique(x, return_index=True)
	y = y[index]

	# adjust for points < x.min()
	ldx = (y[1] - y[0]) / (x[1] - x[0])
	haty[lflags] = y[0] + (newx[lflags] - x[0]) * ldx
	# adjust for points > x.max()
	udx = (y[-1] - y[-2]) / (x[-1] - x[-2])
	haty[uflags] = y[-1] + (newx[uflags] - x[-1]) * udx
	return haty

def spline_interpolate(
    x, y, newx
):	
	if not utilities.haslength(newx):
		newx = np.array([newx])
	spline_rep = sp.interpolate.splrep(x=x, y=y)
	return sp.interpolate.splev(newx, spline_rep)