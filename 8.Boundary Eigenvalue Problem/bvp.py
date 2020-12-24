import bvp_lib as bvp

import numpy as np
import scipy.optimize as op
import scipy.signal as sg

import matplotlib.pyplot as plt
from matplotlib import rc

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


#------------------------------------------------------------------------------
'''
Solving the infinite square well on the interval x \in [0, \pi]
'''

# Exact eigenfunctions
def inf_exact(x, E):
	return np.sin((E)**(1/2) * x)

# Boundary conditions
a = 0
b = 0

# Array of points to solve the equation at
x_num = 1000
x = np.linspace(0, np.pi, x_num)


# Solving the problem with the shooting method

# Function that returns standard form for each E
def inf_g(E):
	
	def inf_f(y, x):
		return np.array([y[1], - E * y[0]], dtype=np.float64)

	return inf_f

# Soltuions of the BVP (tolerance = 10**(-8))
ya, Ea = bvp.eig_shoot(inf_g, a, b, 0.6, 1.1, x)
yb, Eb = bvp.eig_shoot(inf_g, a, b, 3.5, 3.9, x)
yc, Ec = bvp.eig_shoot(inf_g, a, b, 8.5, 9.1, x)
yd, Ed = bvp.eig_shoot(inf_g, a, b, 15.4, 15.7, x)

print(f'The energy eigenvalues are:{Ea, Eb, Ec, Ed}')

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Position $x$')
plt.ylabel(r'Wavefunction $\psi (x)$')
plt.title(r'Infinite Well, Shooting Method')
plt.ylim(-0.75, 1.10)
plt.grid()

plt.plot([0, 0, np.pi, np.pi], [5, 0, 0, 5], color='k', linestyle='--', label='Potential Well')

plt.plot(x, ya, '-', label=fr'$E={Ea}$')
plt.plot(x, yb, '-', label=fr'$E={Eb}$')
plt.plot(x, yc, '-', label=fr'$E={Ec}$')
plt.plot(x, yd, '-', label=fr'$E={Ed}$')

plt.legend(loc='best')
plt.savefig('images/inf_shoot_wave.png')
plt.show()


# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Position $x$')
plt.ylabel(r'Relative Error $|1 - \tilde{\psi} / \psi (\tilde{E})|$')
plt.title(r'Infinite Well, Shooting Method, Tolerance$=10^{-8}$')
plt.yscale('log')
plt.ylim(6.6*10**(-7), 4.4*10**(-5))
plt.grid()

plt.plot([0, 0, np.pi, np.pi], [5, 0, 0, 5], color='k', linestyle='--', label='Potential Well')

plt.plot(x, np.abs(1 - ya/np.max(ya)/inf_exact(x, Ea)), '-', label=fr'$E={Ea}$')
plt.plot(x, np.abs(1 - yb/np.max(yb)/inf_exact(x, Eb)), '-', label=fr'$E={Eb}$')
plt.plot(x, np.abs(1 - yc/np.max(yc)/inf_exact(x, Ec)), '-', label=fr'$E={Ec}$')
plt.plot(x, np.abs(1 - yd/np.max(yd)/inf_exact(x, Ed)), '-', label=fr'$E={Ed}$')

plt.legend(loc='center left')
plt.savefig('images/inf_shoot_err.png')
plt.show()



# Solving the problem with the finite difference method

# Define input function for lin_fd method
def inf_v(x, E):
	return np.array([-E] * len(x), dtype=np.float64)


# Interval on which to search for eigenvalue
E1 = 0
E2 = 17
E_num = 100000
E_lst = np.linspace(E1, E2, E_num)

# Tolerance on second end
b_tol = 10**(-14)

# Absolute maximal deviations from 0
y_amp_lst = []

# loop over energies
for E in E_lst:

	# indicator
	print(f'Calculating for E={E}')
	
	# solve the equation
	y = bvp.lin_fd(0, inf_v(x, E), 0, x, a, b + b_tol)

	# find the maximum
	y_amp_lst.append(np.max(np.abs(y)))

# Found eigenvalues
eigs = E_lst[sg.find_peaks(y_amp_lst)[0]]

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Energy $E$')
plt.ylabel(r'Maximal Amplitude $\mathrm{max}_{x}|\tilde{\psi}(x)|$ [arb. unit]')
plt.title(fr'Infinite Well, Finite Difference Method, $N_E = {E_num}$')
plt.yscale('log')
plt.grid()

plt.plot(E_lst, y_amp_lst/max(y_amp_lst), label='Response')
plt.axvline(eigs[0], color='k', linestyle='--', label='Eigenvalues')

for E in eigs:
	plt.axvline(E, color='k', linestyle='--')

plt.legend(loc='best')
plt.savefig('images/inf_fd_eigs.png')
plt.show()

print(f'The energy eigenvalues between {E1} and {E2} are:')
print(eigs)

# Solutions for eigenvalues
y_lst = []
for E in eigs:
	y = bvp.lin_fd(0, inf_v(x, E), 0, x, a, b + b_tol)

	y_lst.append(y)

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Position $x$')
plt.ylabel(r'Absolute Error $|\psi (\tilde{E}) - \tilde{\psi}|$')
plt.title(fr'Infinite Well, Finite Difference Method, $N_x={x_num}$')
plt.yscale('log')
plt.grid()

for i in range(len(eigs)):
	if y_lst[i][1] > 0:
		y_normed = y_lst[i]/np.max(np.abs(y_lst[i]))
	else:
		y_normed = -y_lst[i]/np.max(np.abs(y_lst[i]))

	plt.plot(x, np.abs(y_normed - inf_exact(x, eigs[i])), label=fr'$E={eigs[i]}$')


plt.legend(loc='best')
plt.savefig('images/inf_fd_err.png')
plt.show()	


#------------------------------------------------------------------------------
'''
Solving the finite square well on the of width \pi
'''

# depth parameter of the well
V_0 = 10

# the potential
def fin_V(x):
	return V_0 * (1 + np.heaviside(x - np.pi/2, 0) - np.heaviside(x + np.pi/2, 0))

def fin_exact(x, E):
	kappa = (V_0 - E)**(1/2)
	k = (E)**(1/2)

	psi1_even = np.cos(k * x)
	psi1_odd = np.sin(k * x)
	psi2_even = np.exp(kappa * (np.pi / 2 - np.abs(x))) * np.cos(k * np.pi / 2)
	psi2_odd = np.sign(x) * np.exp(kappa * (np.pi / 2 - np.abs(x))) * np.sin(k * np.pi / 2)

	return [(1 + np.heaviside(x - np.pi/2, 0) - np.heaviside(x + np.pi/2, 0)) * psi2_even +
			(np.heaviside(x + np.pi/2, 0) - np.heaviside(x - np.pi/2, 0)) * psi1_even,
			(1 + np.heaviside(x - np.pi/2, 0) - np.heaviside(x + np.pi/2, 0)) * psi2_odd +
			(np.heaviside(x + np.pi/2, 0) - np.heaviside(x - np.pi/2, 0)) * psi1_odd]



# Solving the problem with the shooting method

# Function that returns standard form for each E
def fin_g(E):

	def fin_f(y, x):
		return np.array([y[1], (fin_V(x) - E) * y[0]], dtype=np.float64)
		
	return fin_f


# Array of points to solve the equation at
x_min = 0
x_max = 2*np.pi
x_num = 1000
x = np.linspace(x_min, x_max, x_num)

# Size of well in indices
i_max = int(((np.pi/(x_max - x_min)/2) * x_num)) 

# We will shoot from 0 so we have to solve the problem twice. The first time 
# we use cosine like initial conditions to get the even wavefunctions, while the
# second time we use sine like initial conditions to get the odd wavefunctions.

# Cosine like boundary conditions
a = 1
adot = 0
b = 0

# Solution (tolerance = 10**(-8))
ya, Ea = bvp.eig_shoot(fin_g, a, b, 0.6, 1.1, x, adot=adot)
yc, Ec = bvp.eig_shoot(fin_g, a, b, 5.5, 6.5, x, adot=adot)

print(f'The even energy eigenvalues are:{Ea, Ec}')

# Sine like boundary conditions
a = 0
adot = 1
b = 0

# Solution (tolerance = 10**(-8))
yb, Eb = bvp.eig_shoot(fin_g, a, b, 2.5, 3.0, x, adot=adot)
yd, Ed = bvp.eig_shoot(fin_g, a, b, 8.5, 9.1, x, adot=adot)

print(f'The odd energy eigenvalues are:{Eb, Ed}')

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Position $x$')
plt.ylabel(r'Wavefunction $\psi (x)$')
plt.title(fr'Finite Well $V_0={V_0}$, Shooting Method')
plt.ylim(-1.10, 1.10)
plt.grid()


plt.axvline(np.pi/2, color='k', linestyle='--', label='Potential Well')

plt.plot(x, ya, '-', label=fr'$E={Ea}$')
plt.plot(x, yb, '-', label=fr'$E={Eb}$')
plt.plot(x, yc, '-', label=fr'$E={Ec}$')
plt.plot(x, yd, '-', label=fr'$E={Ed}$')

plt.legend(loc='best')
plt.savefig('images/fin_shoot_wave.png')
plt.show()


# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Position $x$')
plt.ylabel(r'Relative Error $|\tilde{\psi} - \psi (\tilde{E})|$')
plt.title(fr'Finite Well $V_0={V_0}$, Shooting Method, Tolerance$=10^{-8}$')
plt.yscale('log')
plt.grid()

plt.axvline(np.pi/2, color='k', linestyle='--', label='Potential Well')

plt.plot(x, np.abs(ya/np.max(ya[:i_max]) - fin_exact(x, Ea)[0]), '-', label=fr'$E={Ea}$')
plt.plot(x, np.abs(yb/np.max(yb[:i_max]) - fin_exact(x, Eb)[1]), '-', label=fr'$E={Eb}$')
plt.plot(x, np.abs(yc/np.max(yc[:i_max]) - fin_exact(x, Ec)[0]), '-', label=fr'$E={Ec}$')
plt.plot(x, np.abs(yd/np.max(yd[:i_max]) - fin_exact(x, Ed)[1]), '-', label=fr'$E={Ed}$')

plt.legend(loc='best')
plt.savefig('images/fin_shoot_err.png')
plt.show()



# Solving the problem with the finite difference method

# Define input function for lin_fd method
def fin_v(x, E):
	return fin_V(x) - E

# Boundary conditions
a = 1
b_cos = a
b_sin = -a

# Array of points to solve the equation at
x_min = -3*np.pi
x_max = 3*np.pi
x_num = 3000
x = np.linspace(x_min, x_max, x_num)

# Size of well in indices
i_min = int(((1 - np.pi/(x_max - x_min)) * x_num) / 2)
i_max = int(((1 + np.pi/(x_max - x_min)) * x_num) / 2)

# Interval on which to search for eigenvalue
E1 = 0
E2 = 10.0
E_num = 100000
E_lst = np.linspace(E1, E2, E_num)

# Absolute maximal deviations from 0
y_cos_amp_lst = []
y_sin_amp_lst = []

# loop over energies
for E in E_lst:

	# indicator
	print(f'Calculating for E={E}')

	# solve the equation for cos-like bcs
	y_cos = bvp.lin_fd(0, fin_v(x, E), 0, x, a, b_cos)

	# solve the equation for sin-like bcs
	y_sin = bvp.lin_fd(0, fin_v(x, E), 0, x, a, b_sin)
	
	# compute and save the maxima
	y_cos_amp_lst.append(np.max(np.abs(y_cos[i_min:i_max])))
	y_sin_amp_lst.append(np.max(np.abs(y_sin[i_min:i_max])))

# The energy eigenvalues
eigs_even = E_lst[sg.find_peaks(y_cos_amp_lst)[0]]
eigs_odd = E_lst[sg.find_peaks(y_sin_amp_lst)[0]]

eigs = np.sort(np.concatenate((eigs_even, eigs_odd)))

print('The energy eigenvalues are:')
print(f'even:{eigs_even}, odd:{eigs_odd}')

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Energy $E$')
plt.ylabel(r'Maximal Amplitude $\mathrm{max}_{x}|\tilde{\psi}(x)|$ [arb. unit]')
plt.title(fr'Finite Well $V_0={V_0}$, Finite Difference Method, $N_E = {E_num}$')
plt.yscale('log')
plt.grid()

plt.plot(E_lst, y_cos_amp_lst/max(y_cos_amp_lst), label='Even Response')
plt.plot(E_lst, y_sin_amp_lst/max(y_sin_amp_lst), label='Odd Response')
plt.axvline(eigs[0], color='k', linestyle='--', label='Eigenvalues')

for E in eigs:
	plt.axvline(E, color='k', linestyle='--')

plt.legend(loc='best')
plt.savefig('images/fin_fd_eigs.png')
plt.show()

print(f'The energy eigenvalues between {E1} and {E2} are:')
print(f'even: {eigs_even}, odd: {eigs_odd}')

# Solutions for eigenvalues
y_lst = []
for i in range(len(eigs)):
	if i % 2 == 0:
		y = bvp.lin_fd(0, fin_v(x, eigs[i]), 0, x, a, b_cos)
	else:
		y = bvp.lin_fd(0, fin_v(x, eigs[i]), 0, x, a, b_sin)

	y_lst.append(y)

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Position $x$')
plt.ylabel(r'Absolute Error $|\psi (\tilde{E}) - \tilde{\psi}|$')
plt.title(fr'Finite Well $V_0={V_0}$, Finite Difference Method, $N_x={x_num}$')
plt.ylim(5.0*10**(-10), 10**(-2))
plt.xlim(-5, 5)
plt.yscale('log')
plt.grid()

for i in range(len(eigs)):
	if i in [0]:
		y_normed = -y_lst[i]/np.max(np.abs(y_lst[i][i_min:i_max]))
	else:
		y_normed = y_lst[i]/np.max(np.abs(y_lst[i][i_min:i_max]))

	plt.plot(x, np.abs(y_normed - fin_exact(x, eigs[i])[i % 2]), label=fr'$E={eigs[i]}$')

plt.axvline(-np.pi/2, color='k', linestyle='--')
plt.axvline(np.pi/2, color='k', linestyle='--')

plt.legend(loc='best')
plt.savefig('images/fin_fd_err.png')
plt.show()	
