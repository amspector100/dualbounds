"""
Contains various interpolation functions used to 
go from discrete dual variables to continuous ones.
"""

import numpy as np
import scipy as sp
from . import utilities

def linear_interpolate(x, y, newx):
	"""
	x : np.array
		n-length array of inputs. Must be sorted, although
		this is not explicitly enforced to save time.
	y : np.array
		n-length array of outputs
	newx : np.array
		m-length array of new inputs
	"""
	if not utilities.haslength(newx):
		newx = np.array([newx])
	# for now, check sorting (TODO DELETE)
	# if np.any(np.sort(x) != x):
	# 	raise ValueError("NOT SORTED")
	# interpolate points in the range of x
	haty = np.interp(newx, x, y)
	# adjust for points < x.min()
	lflags = newx < x[0]
	ldx = (y[1] - y[0]) / (x[1] - x[0])
	haty[lflags] = y[0] + (newx[lflags] - x[0]) * ldx
	# adjust for points > x.max()
	uflags = newx > x[-1]
	udx = (y[-1] - y[-2]) / (x[-1] - x[-2])
	haty[uflags] = y[-1] + (newx[uflags] - x[-1]) * udx
	return haty

def spline_interpolate(
    x, y, newx
):
    spline_rep = sp.interpolate.splrep(x=x, y=y)
    return sp.interpolate.splev(newx, spline_rep)