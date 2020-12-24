import diffeq as de
import tridiagonal as tri

import numpy as np
import scipy.special as spe
import scipy.sparse as sp

import matplotlib.pyplot as plt
from matplotlib import rc

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Parameters of the system
D = 0.1
a = 1
sigma = 0.1
A0 = 1

# Gaussian with size A0 at x=mu
def gauss(x, mu, sigma, A0=1):
	return A0 * np.exp(- (x - mu)**2 / (2 * sigma**2))

# Solving the equation with periodic boundary conditions
# and the Fourier method

# lattice of space points
x_num = 100
x = np.linspace(0, a, x_num, endpoint=False)

# lattice of time points
t_end = 0.0003
t_num = 100
t = np.linspace(0, t_end, t_num)

# initial condition (gaussian)
y0 = gauss(x, a/2, sigma, A0)

# transform initial condition
tildey0 = np.fft.fft(y0)

# Fourier transformed equation
def four_diff(y, t):

	# vector of k values
	k = np.array(range(len(y)), dtype=np.float64)

	return - 4 * np.pi**2 * D / a**2 * k**2 * y

# solution to FT equation
tildey = de.rku4(four_diff, tildey0, t)

# solution back to normal
y = np.fft.ifft(tildey)


# Plotting solution
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel(r'Position $x$')
ax.set_ylabel(r'Time $t$')
ax.set_zlabel(r'Temperature $T(x,t)$')
ax.set_title(fr'Temperature Portrait over Time')

x_dummy, t_dummy = np.meshgrid(x, t)

surf = ax.plot_surface(x_dummy, t_dummy, y, linewidth=0, antialiased=False)

plt.savefig('images/temp_four_per.pdf')
plt.show()



# Solving the equation with Dirichlet boundary conditions
# and the Fourier method

# lattice of space points
x_num = 200
x = np.linspace(-a, a, x_num, endpoint=False)

# lattice of time points
t_end = 0.001
t_num = 100
t = np.linspace(0, t_end, t_num)

# initial condition (approximate gaussian)
y0 = gauss(x, a/2, sigma, A0) - gauss(x, -a/2, sigma, A0)
y0 = y0 + y0[0] * (2 * np.heaviside(x, 1/2) - 1)

# transform initial condition
tildey0 = -np.imag(np.fft.ifftshift(np.fft.fft(np.fft.fftshift(y0))))

# Fourier transformed equation
def four_diff(y, t):

	# vector of k values
	k = np.array(range(len(y)), dtype=np.float64)

	return - np.pi**2 * D / a**2 * k**2 * y

# solution to FT equation
tildey = de.rku4(four_diff, tildey0, t)

# solution back to normal
y = np.imag(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(tildey))))/len(tildey)

# Plotting solution
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel(r'Position $x$')
ax.set_ylabel(r'Time $t$')
ax.set_zlabel(r'Temperature $T(x,t)$')
ax.set_title(fr'Temperature Portrait over Time')

x_dummy, t_dummy = np.meshgrid(x, t)

surf = ax.plot_surface(x_dummy, t_dummy, np.real(y), linewidth=0, antialiased=False)

plt.savefig('images/temp_four_diri.pdf')
plt.show()



# Solving the equation with Dirichlet boundary conditions
# and the collocation method

# lattice of space points
x_num = 200
x = np.linspace(0, a, x_num)
deltax = x[1] - x[0]

# lattice of time points
t_end = 1
t_num = 1000
t = np.linspace(0, t_end, t_num)
deltat = t[1] - t[0]

# initial condition (approximate gaussian)
y0 = gauss(x, a/2, sigma, A0)
y0 = y0 - y0[0]


"""
# Efficient way, currently does not work

# setup of equations for implicit Euler
ones1 = np.ones(x_num - 3)
ones2 = np.ones(x_num - 2)
a1 = (1 - 3 * D * deltat / deltax**2) * ones1
a2 = (1 + 3 * D * deltat / deltax**2) * ones1
b1 = (4 + 6 * D * deltat / deltax**2) * ones2
b2 = (4 - 6 * D * deltat / deltax**2) * ones2

# initial condition for c
c = [y0[1:-1]]

# solve system with implicit Euler
for i in range(1, len(t)):

	# (A + deltat * B / 2) * oldc
	tmpc = tri.multiply(a2, b2, a2, c[-1])

	# solve tridiagonal system
	newc = tri.solve(a1, b1, a1, tmpc)

	# save solution
	c.append(newc)

# fix up c
c = np.array(c)
zer = np.zeros(t_num)
zer.shape = (t_num, 1)

c = np.hstack((-c[:,0:1], zer, c, zer, -c[:,-1:]))
c = np.transpose(c)

# if we agree to calculate the function only at the collocation points
# then we don't need the full form of B_K(x), only that is it equal to
# [0, 1, 4, 1, 0] at the consecutive points from x_(k-2) to x_(k+2)

# make B_k(x_j)
a = np.ones(x_num + 1)
d = 4 * np.ones(x_num + 2)	

# calculate y
y = []
for i in range(t_num):

	# calculate B_k(x_j) on column of c
	tmpy = tri.multiply(a, d, a, c[:,i])

	# save column as row
	y.append(tmpy)

# vectorize y
y = np.array(y)


# Plotting solution
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel(r'Position $x$')
ax.set_ylabel(r'Time $t$')
ax.set_zlabel(r'Temperature $T(x,t)$')
ax.set_title(fr'Temperature Portrait over Time')

x_dummy, t_dummy = np.meshgrid(x, t)

surf = ax.plot_surface(x_dummy, t_dummy, y[:,1:-1], linewidth=0, antialiased=False)

plt.savefig('images/temp_colloc_diri.pdf')
plt.show()
"""


# Inefficient way

# matrices A and B
A = sp.diags([np.ones(x_num - 3), 4 * np.ones(x_num - 2), np.ones(x_num - 3)], (-1, 0, 1)).A
invA = np.linalg.inv(A)
B = (6 * D / deltax**2) * sp.diags([np.ones(x_num - 3), -2 * np.ones(x_num - 2), np.ones(x_num - 3)], (-1, 0, 1)).A

C = np.linalg.inv(A - deltat/2 * B)
D = A + deltat/2 * B

# initial condition for c
c = [np.dot(invA, y0[1:-1])]

for _ in range(1, t_num):

	#calculate y with Euler
	newc = np.dot(C, np.dot(D, c[-1]))

	# save result
	c.append(newc)

# fix up c
c = np.array(c)
zer = np.zeros(t_num)
zer.shape = (t_num, 1)

c = np.hstack((-c[:,0:1], zer, c, zer, -c[:,-1:]))
c = np.transpose(c)

# if we agree to calculate the function only at the collocation points
# then we don't need the full form of B_K(x), only that is it equal to
# [0, 1, 4, 1, 0] at the consecutive points from x_(k-2) to x_(k+2)

# make B_k(x_j)
Bspl = sp.diags([np.ones(x_num + 1), 4 * np.ones(x_num + 2), np.ones(x_num + 1)], (-1, 0, 1)).A

# calculate y
y = np.dot(Bspl, c)
y = np.transpose(y)

# Plotting solution
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_xlabel(r'Position $x$')
ax.set_ylabel(r'Time $t$')
ax.set_zlabel(r'Temperature $T(x,t)$')
ax.set_title(fr'Temperature Portrait over Time')

x_dummy, t_dummy = np.meshgrid(x, t)

surf = ax.plot_surface(x_dummy, t_dummy, y[:,1:-1], linewidth=0, antialiased=False)

plt.savefig('images/temp_colloc_diri.pdf')
plt.show()
