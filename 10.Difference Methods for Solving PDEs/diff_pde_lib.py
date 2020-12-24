#-----------------------------------------------------------------------------
"""
AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    September 2, 2008

MODIFIED:
	Aleksandar Ivanov <aleksandar.ivanov@student.fmf.uni-lj.si>
	University of Ljubljana
	December 15, 2020
"""
#-----------------------------------------------------------------------------

import numpy as np
import tridiagonal as tri
import matplotlib.pyplot as plt
import scipy.linalg as la

#----------------------------------------------------------------------------- 

def ftcs(x, t, a, b, f, Q, K=1):
	"""Use the FTCS (forward time, centered space) scheme to solve the equation
	    
	       du     d  du
	       -- = K -- -- + Q(x) u
    	   dt     dx dx

	INPUT:
		x  - list or NumPy array specifying the spacial lattice points
		t  - list or NumPy array specifying the temporal lattice points
		a  - list or NumPy array specifying the boundary condtition at x[0]
		b  - list or NumPy array specifying the boundary condtition at x[-1]
		f  - list or NumPy array specifying the initial condtition at t[0]
		Q  - list or NumPy array specifying the function Q at the points x
		K  - value of the diffusion coefficient

	OUTPUT:
		u  - 2D NumPy array with the solution to the equation. A single row has
		     values corresponding to the one single time value. A single column
		     has values corresponding to the one single space value.
	"""

	# Number of spatial points and step (assumed constant)
	m = len(x)
	h = x[1] - x[0]

	# Number of temporal points and step (assumed constant)
	n = len(t)
	k = t[1] - t[0]

	# Compute the value of r
	# Note that if r > 1/2 then this procedure may be unstable
	r = K * k / h**2

	if np.abs(r) > 1/2:
		print(f'WARNING: The procedure with r={r} may be unstable.')
		print('Try choosing a value less than 1/2 to stabilize it.')

	# Initialize solution array
	u = np.zeros((m, n), dtype=np.complex64)

	# Initial condition
	u[:,0] = f

	# Boundary conditions
	u[0,:] = a
	u[-1,:] = b

	# Main loop for solution
	for j in range(n - 1):

		# idicator
		print(f'FTCS, solving time step {j + 1} out of {n}')
		print(np.max(u[:,j]))

		# calculate Q part
		kQu = k * np.array(tri.multiply(np.zeros(m - 3), Q[1:-1], np.zeros(m - 3), u[1:-1,j]))

		u[1:-1,j+1] = r * u[0:-2,j] + (1 - 2 * r) * u[1:-1,j] + r * u[2:,j] + kQu

	return u

#----------------------------------------------------------------------------- 

def cncn(x, t, a, b, f, Q, K=1, c=None):
	"""Use the modified Crank-Nicolson scheme to solve the equation
	    
	       du     d  du
	       -- = K -- -- + Q(x) u
    	   dt     dx dx

	INPUT:
		x  - list or NumPy array specifying the spacial lattice points
		t  - list or NumPy array specifying the temporal lattice points
		a  - list or NumPy array specifying the boundary condtition at x[0]
		b  - list or NumPy array specifying the boundary condtition at x[-1]
		f  - list or NumPy array specifying the initial condtition at t[0]
		Q  - list or NumPy array specifying the function Q at the points x
		K  - value of the diffusion coefficient
		c  - derivative averaging parameter. c = 1/2 for the normal Crank-Nicolson
		     method. c = 0 for FTCS. Defaults to the optimal value of 1/2 - 1/(12 r)

	OUTPUT:
		u  - 2D NumPy array with the solution to the equation. A single row has
		     values corresponding to the one single time value. A single column
		     has values corresponding to the one single space value.
	"""


	# Number of spatial points and step
	m = len(x)
	h = x[1] - x[0]

	# Number of temporal points and step
	n = len(t)
	k = t[1] - t[0]

	# Compute the value of K * k / ( h * h ).
	r = K * k / h**2
	print(f'You are using r={r}')

	# Choose optimal c if not given
	if c is None:
		c = 1/2 - 1/(12 * r)

	# Initialize solution array
	u = np.zeros((m, n), dtype=np.complex64)

	# Initial condition
	u[:,0] = f

	# Boundary conditions
	u[0,:] = a
	u[-1,:] = b

	# The discretization used leads to a tridiagonal linear system. We need to
	# construct the diagonals of two tridiagonal matrices A and B such that we
	# are solving the equation A U(t[n+1]) = B U(t[n])
	"""
	aA = - c * r * np.ones(m - 1)
	dA = (1 + 2 * c * r) * np.ones(m)

	A_banded = np.vstack((np.hstack((aA, [0])), dA, np.hstack(([0], aA))))

	aB = (1 - c) * r * np.ones(m - 1)
	dB = (1 - 2 * (1 - c) * r) * np.ones(m) + k * Q

	B_banded = np.vstack((np.hstack((aB, [0])), dB, np.hstack(([0], aB))))
	"""

	aA = - r/4 * np.ones(m - 1)
	dA = (1 + r/2) * np.ones(m) + k * Q / 2 

	A_banded = np.vstack((np.hstack((aA, [0])), dA, np.hstack(([0], aA))))

	# Main loop to get the solution
	for j in range(n - 1):

		# idicator
		print(f'Crank-Nicolson, solving time step {j + 1} out of {n}')
		print(np.max(u[:,j]))
		"""
		# construct right hand side
		b = tri.multiply(aB, dB, aB, u[:,j])
		
		# solve left hand side
		u[:,j+1] = tri.solve(aA, dA, aA, b)
		"""

		b = tri.multiply(np.conj(aA), np.conj(dA), np.conj(aA), u[:,j])

		u[:,j+1] = la.solve_banded((1,1), A_banded, b)

	return u

#----------------------------------------------------------------------------- 
