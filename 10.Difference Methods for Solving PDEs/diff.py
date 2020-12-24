import tridiagonal as tri

import numpy as np
import scipy.linalg as la
import scipy.signal as sg

import matplotlib.pyplot as plt
from matplotlib import rc

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Crank-Nicolson scheme
def cncn(x, t, a, b, f, Q, K=1):
	"""Use the symmetrized Crank-Nicolson scheme to solve the equation
	    
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


	# Number of spatial points and step
	m = len(x)
	h = x[1] - x[0]

	# Number of temporal points and step
	n = len(t)
	k = t[1] - t[0]

	# Compute the value of K * k / ( h * h ).
	r = K * k / h**2
	print(f'You are using r={r}')

	# Initialize solution array
	u = np.zeros((m, n), dtype=np.complex64)

	# Initial condition
	u[:,0] = f

	# Boundary conditions
	u[0,:] = a
	u[-1,:] = b

	# The discretization used leads to a tridiagonal linear system. We need to
	# construct the diagonals of the tridiagonal matrix A such that we
	# are solving the equation A U(t[n+1]) = A* U(t[n])

	aA = - r/4 * np.ones(m - 1)
	dA = (1 + r/2) * np.ones(m) + k * Q / 2 

	A_banded = np.vstack((np.hstack((aA, [0])), dA, np.hstack(([0], aA))))

	# Main loop to get the solution
	for j in range(n - 1):

		# idicator
		print(f'Crank-Nicolson, solving time step {j + 1} out of {n}')
		
		# calculate right hand side
		b = tri.multiply(np.conj(aA), np.conj(dA), np.conj(aA), u[:,j])

		# solve for u
		u[:,j+1] = la.solve_banded((1,1), A_banded, b)

	return u

#------------------------------------------------------------------------------
'''
Check speed of Crank-Nicolson
'''

times_M = []

# Time lattice
tmin = 0
tmax = 2*np.pi
N = 100
t = np.linspace(tmin, tmax, N)

# BCs
a = np.zeros(N)
b = np.zeros(N)

# changing space points
M_lst = range(5, 1000)

for M in M_lst:
	x = np.linspace(-40, 40, M)

	# Initial condition
	f = np.random.rand(M)

	# Solve equation
	start = timer()
	psi = cncn(x, t, a, b, f, 0, 1j)
	end = timer()

	times_M.append(end - start)


times_N = []

# Space lattice
xmin = -40
xmax = 40
M = 100
x = np.linspace(xmin, xmax, M)

# changing space points
N_lst = range(5, 1000)

for N in N_lst:

	t = np.linspace(0, 6.5, N)

	# BCs
	a = np.zeros(N)
	b = np.zeros(N)

	# Initial condition
	f = np.random.rand(M)

	# Solve equation
	start = timer()
	psi = cncn(x, t, a, b, f, 0, 1j)
	end = timer()

	times_N.append(end - start)

# Plotting
fig, axs = plt.subplots(1)

plt.xlabel(r'Number of Space/Time Points $M/N$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.yscale('log')
plt.xscale('log')
plt.title(r'Evaluation Time as a Function of Number of Points')
plt.grid()

plt.plot(M_lst, times_M, 'o', label='Space')
plt.plot(N_lst, times_N, 'o', label='Time')

plt.legend(loc='lower right')
plt.savefig('images/times.png')
plt.show()


#------------------------------------------------------------------------------
'''
Evolving the coherent state with Crank-Nicolson
'''

# potential
def V(x):
	return x**2 / 2

# initial position of peak
lmd = -10

# initial wavefunction
def psi_0(x, lmd):
	return (np.pi)**(-1/4) * np.exp(-(x - lmd)**2 / 2)

def exact(x, t, lmd):
	ret = []

	for xi in x:
		tmp = (np.pi)**(-1/4) * np.exp(-(xi - lmd * np.cos(t))**2 / 2 - 1j*t/2 - 1j*xi*lmd*np.sin(t) - 1j/4*lmd**2*np.sin(2*t))
		ret.append(tmp)

	return np.array(ret)

# Space lattice
xmin = -40
xmax = 40
M = 1000
x = np.linspace(xmin, xmax, M)

# Time lattice
tmin = 0
tmax = 4*np.pi
N = 10000
t = np.linspace(tmin, tmax, N)

# Initial and boundary conditions
f = psi_0(x, lmd)
a = np.zeros(N)
b = np.zeros(N)

# Solve equation
psi = cncn(x, t, a, b, f, 1j * V(x), 1j)
"""
# Plotting solution
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Position $x$')
ax.set_zlabel(r'Wavefunction $\Psi (x,t)$')
ax.set_title(fr'Wavefunction over Time')

t_dummy, x_dummy = np.meshgrid(t, x)

surf = ax.plot_surface(t_dummy, x_dummy, np.abs(psi), linewidth=0, antialiased=False, rstride=1, cstride=1)

plt.savefig('images/cncn_sol.png')
#plt.show()
plt.close()
"""
#fins peak
peaks = []
for n in range(N):
	peaks.append(max(range(M), key=np.abs(psi[:,n]).__getitem__))
peaks = np.array(peaks)

# Plotting
fig, axs = plt.subplots(1)

plt.ylabel(r'Peak Position $x_0$')
plt.xlabel(r'Time $t$')
plt.title(r'Peak Position over Time')
plt.grid()

plt.plot(t, x[peaks], 'ko')

plt.legend(loc='best')
plt.savefig('images/peak_pos.png')
plt.show()

# Plotting
fig, axs = plt.subplots(1)

plt.ylabel(r'Peak Height $\Psi_0$')
plt.xlabel(r'Time $t$')
plt.title(r'Peak Height over Time')
plt.grid()

plt.plot(t, [np.abs(psi[peaks[n],n]) for n in range(N)], 'ko')

plt.legend(loc='best')
plt.savefig('images/peak_height.png')
plt.show()


#------------------------------------------------------------------------------
'''
Evolving the Gaussian in V=0 with Crank-Nicolson
'''

# parameters
lmd = 0.25
sgm0 = 0.05
p0 = 50*np.pi

# initial wavefunction
def psi_0(x, lmd, sgm0, p0):
	return (2*np.pi*sgm0**2)**(-1/4) * np.exp(-(x - lmd)**2 / (2*sgm0)**2 + 1j*p0*(x - lmd))

def exact(x, t, lmd):
	ret = []

	for xi in x:
		tmp = (np.pi)**(-1/4) * np.exp(-(xi - lmd * np.cos(t))**2 / 2 - 1j*t/2 - 1j*xi*lmd*np.sin(t) - 1j/4*lmd**2*np.sin(2*t))
		ret.append(tmp)

	return np.array(ret)

# Space lattice
xmin = -0.5
xmax = 1.5
M = 1000
x = np.linspace(xmin, xmax, M)

# Time lattice
tmin = 0
tmax = 0.003
N = 375
t = np.linspace(tmin, tmax, N)

# Initial and boundary conditions
f = psi_0(x, lmd, sgm0, p0)
a = np.zeros(N)
b = np.zeros(N)

# Solve equation
psi = cncn(x, t, a, b, f, 0, 1j)

# Plotting solution
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Position $x$')
ax.set_zlabel(r'Wavefunction $\Psi (x,t)$')
ax.set_title(fr'Wavefunction over Time')

t_dummy, x_dummy = np.meshgrid(t, x)

surf = ax.plot_surface(t_dummy, x_dummy, np.abs(psi), linewidth=0, antialiased=False, rstride=1, cstride=1)

plt.savefig('images/cncn_0_sol.png')
#plt.show()
plt.close()

# find peak
peaks = []
for n in range(N):
	peaks.append(max(range(M), key=np.abs(psi[:,n]).__getitem__))
peaks = np.array(peaks)

# find std
std = []
for n in range(N):
	std.append(np.std(psi[:,n]))

# Plotting
fig, axs = plt.subplots(1)

plt.ylabel(r'Peak Position $x_0$')
plt.xlabel(r'Time $t$')
plt.title(r'Peak Position over Time')
plt.grid()

plt.plot(t, x[peaks], 'ko')

plt.legend(loc='best')
plt.savefig('images/peak_0_pos.png')
plt.show()

# Plotting
fig, axs = plt.subplots(1)

plt.ylabel(r'Peak Height $\Psi_0$')
plt.xlabel(r'Time $t$')
plt.title(r'Peak Height over Time')
plt.ylim(1.9, 3.0)
plt.grid()

plt.plot(t, [np.abs(psi[peaks[n],n]) for n in range(N)], 'ko')

plt.legend(loc='best')
plt.savefig('images/peak_0_height.png')
plt.show()

# Plotting
fig, axs = plt.subplots(1)

plt.ylabel(r'Peak Width $\sigma$')
plt.xlabel(r'Time $t$')
plt.title(r'Peak Width over Time')
plt.grid()

plt.plot(t, std, 'ko')

plt.legend(loc='best')
plt.savefig('images/peak_0_width.png')
plt.show()
