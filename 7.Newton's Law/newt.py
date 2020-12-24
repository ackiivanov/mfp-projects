import diffeq_2 as de

import numpy as np
import scipy.special as sp
import scipy.signal as sg

import matplotlib.pyplot as plt
from matplotlib import rc

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Parameters of our system
x0 = 1
v0 = 0

# Differential equation in 1de standard form
def f1(x, t):
	return np.array([x[1], - np.sin(x[0])], dtype=np.float64)

# Differential equation in 2de standard form
def f2(x):
	return - np.sin(x)

# Exact solution to differential equation
def x_exact(t, x0):
	return 2 * np.arcsin(np.sin(x0/2) * sp.ellipj(sp.ellipk((np.sin(x0/2))**2)
		   - t, (np.sin(x0/2))**2)[0])

# Energy of system
def erg(x, v):
	return - np.cos(x) + v**2/2

# Exact period of oscilaltion
def T_exact(x0):
	return 4 * sp.ellipk((np.sin(x0/2))**2)



# Looking for the necessary step size for three digit accuracy
# constant N

# set number of points
N = 1000

# setup of h
hmin = 0.001
hmax = 0.7
hnum = 10000
h_lst = hmin + (hmax-hmin)*(np.linspace(0, 1, hnum))**4

# lists of errors for different methods
err_eule = []
err_rku4 = []
err_verl = []
err_pefr = []

# look though all step sizes
for h in h_lst:

	# indicator
	print(f'Calculating for h={h}')

	# list of time points
	t = np.linspace(0, N*h, N)

	# exact solution
	sol = x_exact(t, x0)

	# calculate error
	err_eule.append(np.max(np.abs(de.euler(f1, [x0, v0], t)[:,0] - sol)))
	err_rku4.append(np.max(np.abs(de.rku4(f1, [x0, v0], t)[:,0] - sol)))
	err_verl.append(np.max(np.abs(de.verlet(f2, x0, v0, t)[0] - sol)))
	err_pefr.append(np.max(np.abs(de.pefrl(f2, x0, v0, t)[0] - sol)))

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Step Size $h$')
plt.ylabel(r'Error Estimate $\mathrm{max} |x_i - x_i^{\mathrm{exact}}|$')
plt.title(r'Error Estimate as a Function of Step Size, Constant $N$')
plt.ylim(10**(-16), 10**(2))
plt.yscale('log')
plt.grid()

plt.plot(h_lst, err_eule, 'o-', label='Euler')
plt.plot(h_lst, err_rku4, 'o-', label='RK4')
plt.plot(h_lst, err_verl, 'o-', label='Verlet')
plt.plot(h_lst, err_pefr, 'o-', label='PEFRL')

plt.axhline(10**(-3), color='k', label=r'Desired accuracy')

plt.legend(loc='best')
plt.savefig('images/err_N_2.png')
plt.show()



# Looking for the necessary step size for three digit accuracy
# constant deltat

# set time interval
deltat = 25

# setup of h
hmin = 0.001
hmax = 0.7
hnum = 1000
h_lst = hmin + (hmax-hmin)*(np.linspace(0, 1, hnum))**4

# lists of errors for different methods
err_eule = []
err_rku4 = []
err_verl = []
err_pefr = []

# look though all step sizes
for h in h_lst:

	# indicator
	print(f'Calculating for h={h}')

	# list of time points
	t = np.linspace(0, deltat, int(deltat/h))

	# exact solution
	sol = x_exact(t, x0)

	# calculate error
	err_eule.append(np.max(np.abs(de.euler(f1, [x0, v0], t)[:,0] - sol)))
	err_rku4.append(np.max(np.abs(de.rku4(f1, [x0, v0], t)[:,0] - sol)))
	err_verl.append(np.max(np.abs(de.verlet(f2, x0, v0, t)[0] - sol)))
	err_pefr.append(np.max(np.abs(de.pefrl(f2, x0, v0, t)[0] - sol)))

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Step Size $h$')
plt.ylabel(r'Error Estimate $\mathrm{max} |x_i - x_i^{\mathrm{exact}}|$')
plt.title(r'Error Estimate as a Function of Step Size, Constant $\Delta \tau$')
plt.ylim(10**(-16), 10**(2))
plt.yscale('log')
plt.grid()

plt.plot(h_lst, err_eule, 'o-', label='Euler')
plt.plot(h_lst, err_rku4, 'o-', label='RK4')
plt.plot(h_lst, err_verl, 'o-', label='Verlet')
plt.plot(h_lst, err_pefr, 'o-', label='PEFRL')

plt.axhline(10**(-3), color='k', label=r'Desired accuracy')

plt.legend(loc='best')
plt.savefig('images/err_T.png')
plt.show()



# Error with RKF method
# constant N

# set time interval
deltat = 100

# setup of h
hmin = 0.001
hmax = 0.7

# tolerance
eps = 10**(-8)

# lists of errors for different methods
sol_rkf = de.rkf(f1, 0, deltat, [x0, v0], eps, hmax, hmin)
err_rkf = np.abs(sol_rkf[1][:,0] - x_exact(sol_rkf[0], x0))

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Time $\tau$')
plt.ylabel(r'Error Estimate $|x_i - x_i^{\mathrm{exact}}|$')
plt.title(r'Error Estimate over Time for RKF4(5)')
plt.ylim(10**(-16), 10**(2))
plt.yscale('log')
plt.grid()

plt.plot(sol_rkf[0], err_rkf, 'o-')
plt.axhline(eps, color='k', label=r'Tolerance')

plt.legend(loc='best')
plt.savefig('images/err_RKF.png')
plt.show()



# Looking at the amplitude devitions
# constant deltat

# set time interval
deltat = 50

# step size
h = 0.1

# list of time points
t = np.linspace(0, deltat, int(deltat/h))

# solutions
sol = x_exact(t, x0)
sol_eule = de.euler(f1, [x0, v0], t)[:,0]
sol_rku4 = de.rku4(f1, [x0, v0], t)[:,0]
sol_verl = de.verlet(f2, x0, v0, t)[0]
sol_pefr = de.pefrl(f2, x0, v0, t)[0]

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Time $\tau$')
plt.ylabel(r'Solution $\theta (\tau)$')
plt.title(fr'Solution over Time for Different Methods, $h = {h}$')
plt.ylim(-2, 2)
plt.grid()

plt.plot(t, sol_eule, 'o-', label='Euler')
plt.plot(t, sol_rku4, 'o-', label='RK4')
plt.plot(t, sol_verl, 'o-', label='Verlet')
plt.plot(t, sol_pefr, 'o-', label='PEFRL')

plt.legend(loc='best')
plt.savefig('images/sol_T.png')
plt.show()



# Looking at the energy devitions

# set time interval
deltat = 50

# step size
h = 0.1

# list of time points
t = np.linspace(0, deltat, int(deltat/h))

# solutions
sol_eule = de.euler(f1, [x0, v0], t)
sol_rku4 = de.rku4(f1, [x0, v0], t)
sol_verl = de.verlet(f2, x0, v0, t)
sol_pefr = de.pefrl(f2, x0, v0, t)

# energies
erg_eule = erg(sol_eule[:,0], sol_eule[:,1])
erg_rku4 = erg(sol_rku4[:,0], sol_rku4[:,1])
erg_verl = erg(sol_verl[0], sol_verl[1])
erg_pefr = erg(sol_pefr[0], sol_pefr[1])

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Time $\tau$')
plt.ylabel(r'Energy $E$')
plt.title(fr'Energy over Time for Different Methods, $h = {h}$')
plt.grid()

plt.plot(t, erg_eule, 'o-', color='#1f77b4', label='Euler')
plt.plot(t, erg_rku4, 'o-', color='#ff7f0e', label='RK4')
plt.plot(t, erg_verl, 'o-', color='#2ca02c', label='Verlet')
plt.plot(t, erg_pefr, 'o-', color='#d62728', label='PEFRL')

plt.axhline(erg(x0, v0), color='k', label='Analytical')

plt.legend(loc='best')
plt.savefig('images/erg_T.png')
plt.show()



# Looking at the energy devition, RKF

# set time interval
deltat = 50

# setup of h
hmin = 0.001
hmax = 0.7

# tolerance
eps = 10**(-8)

# solution
sol_rkf = de.rkf(f1, 0, deltat, [x0, v0], eps, hmax, hmin)

# energy
erg_rkf = erg(sol_rkf[1][:,0], sol_rkf[1][:,1])

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Time $\tau$')
plt.ylabel(r'Energy $E$')
plt.title(fr'Energy over Time for RKF4(5)')
plt.grid()

plt.plot(sol_rkf[0], erg_rkf, 'o-', label='RKF4(5)')
plt.axhline(erg(x0, v0), color='k', label='Analytical')

plt.legend(loc='best')
plt.savefig('images/erg_RKF.png')
plt.show()



# Phase portraits
# constant deltat

# set time interval
deltat = 50

# step size
h = 1

# list of time points
t = np.linspace(0, deltat, int(deltat/h))

# solutions
sol_eule = de.euler(f1, [x0, v0], t)
sol_rku4 = de.rku4(f1, [x0, v0], t)
sol_verl = de.verlet(f2, x0, v0, t)
sol_pefr = de.pefrl(f2, x0, v0, t)

# dummy plotting variables
x_dummy = np.linspace(-3, 3, 100)
v_dummy = np.linspace(-2, 2, 100)
erg_dummy = [[erg(x, v) for x in x_dummy] for v in v_dummy]

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Position $\theta$')
plt.ylabel(r'Momentum $\dot{\theta}$')
plt.title(fr'Phase Portraits for Different Methods, $h = {h}$')
plt.xlim(-3, 3)
plt.ylim(-2, 2)
plt.grid()

plt.plot(sol_eule[:,0], sol_eule[:,1], '-', color='#1f77b4', linewidth=2, label='Euler')
plt.plot(sol_rku4[:,0], sol_rku4[:,1], '-', color='#ff7f0e', linewidth=2, label='RK4')
plt.plot(sol_verl[0], sol_verl[1], '-', color='#2ca02c', linewidth=2, label='Verlet')
plt.plot(sol_pefr[0], sol_pefr[1], '-', color='#d62728', linewidth=2, label='PEFRL')

plt.contour(x_dummy, v_dummy, erg_dummy, 10, colors='k', linestyles='dashed', linewidths=1)
plt.contourf(x_dummy, v_dummy, erg_dummy, 250, alpha=0.7/3, antialiased=True)
plt.contourf(x_dummy, v_dummy, erg_dummy, 260, alpha=0.7/3, antialiased=True)
plt.contourf(x_dummy, v_dummy, erg_dummy, 270, alpha=0.7/3, antialiased=True)

clb = plt.colorbar()
clb.ax.set_title(r'Energy $E$')
plt.legend(loc='upper right')
plt.savefig('images/phase_portrait_0.png')
plt.show()



# Finding the period

# list of differernt initial conditions
x0_N = 100
x0_lst = np.linspace(0.01, np.pi - 0.1, x0_N)

# set time interval
deltat = 100

# step size and tolerance
hmin = 0.001
hmax = 0.1
eps = 10**(-8)

# list of time points
t = np.linspace(0, deltat, int(deltat/hmin))

# lists of periods
per_eule = []
per_rku4 = []
per_verl = []
per_pefr = []
per_rkf  = []

# loop over linital conditions
for x0 in x0_lst:

	# indicator
	print(f'Calculating for theta0 = {x0}')

	# solutions
	sol_eule = de.euler(f1, [x0, 0], t)
	sol_rku4 = de.rku4(f1, [x0, 0], t)
	sol_verl = de.verlet(f2, x0, 0, t)
	sol_pefr = de.pefrl(f2, x0, 0, t)
	sol_rkf  = de.rkf(f1, 0, deltat, [x0, 0], eps, hmax, hmin)

	# find peaks
	peaks_eule = t[sg.find_peaks(sol_eule[:,0])[0]]
	peaks_rku4 = t[sg.find_peaks(sol_rku4[:,0])[0]]
	peaks_verl = t[sg.find_peaks(sol_verl[0])[0]]
	peaks_pefr = t[sg.find_peaks(sol_pefr[0])[0]]
	peaks_rkf  = sol_rkf[0][sg.find_peaks(sol_rkf[1][:,0])[0]]

	# average period
	per_eule.append(np.average(np.diff(peaks_eule)))
	per_rku4.append(np.average(np.diff(peaks_rku4)))
	per_verl.append(np.average(np.diff(peaks_verl)))
	per_pefr.append(np.average(np.diff(peaks_pefr)))
	per_rkf.append(np.average(np.diff(peaks_rkf)))

# Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Initial Condition $\theta_0$')
plt.ylabel(r'Period $T$')
plt.title(fr'Period as a function of Initial Condition for Different Methods, $h = {hmin}$')
plt.grid()

plt.plot(x0_lst, per_eule, 'o-', color='#1f77b4', label='Euler')
plt.plot(x0_lst, per_rku4, 'o-', color='#ff7f0e', label='RK4')
plt.plot(x0_lst, per_verl, 'o-', color='#2ca02c', label='Verlet')
plt.plot(x0_lst, per_pefr, 'o-', color='#d62728', label='PEFRL')
plt.plot(x0_lst, per_rkf, 'o-', color='#9467bd', label='RKF4(5)')

plt.plot(x0_lst, T_exact(x0_lst), '-', color='k', label='Analytical')


plt.legend(loc='best')
plt.savefig('images/periods.png')
plt.show()
