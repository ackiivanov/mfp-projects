import diffeq as de

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

from timeit import default_timer as timer


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Parameters of our system
k = 1
Tout = 0
T0 = 100

# Differential equation in standard form
def f(T, t):
	return -k*(T - Tout)

# Exact solution
def exact(t):
	return Tout + (T0 - Tout)*np.exp(-k*t)
exact = np.vectorize(exact)



# Error with constant N

# set number of points
N = 1000

# setup of h
hmin = 0.001
hmax = 3.0
hnum = 10000
h_lst = hmin + (hmax-hmin)*(np.linspace(0, 1, hnum))**4

# lists of errors for different methods
err_eule = []
err_heun = []
err_rku4 = []
err_pc4m = []

# look though all step sizes
for h in h_lst:

	# indicator
	print(f'Calculating for h={h}')

	# list of time points
	t = np.linspace(0, N*h, N)

	# exact solution
	sol = exact(t)

	# calculate error
	err_eule.append(np.average(np.abs(de.euler(f, T0, t) - sol)))
	err_heun.append(np.average(np.abs(de.heun(f, T0, t) - sol)))
	err_rku4.append(np.average(np.abs(de.rku4(f, T0, t) - sol)))
	err_pc4m.append(np.average(np.abs(de.pc4(f, T0, t) - sol)))

#Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Step Size $h$')
plt.ylabel(r'Error Estimate $\mathrm{avg} |x_i - x_i^{\mathrm{exact}}|$')
plt.title(r'Error Estimate as a Function of Step Size, Constant $N$')
plt.ylim(10**(-16), 10**(1))
plt.yscale('log')
plt.grid()

plt.plot(h_lst, err_eule, 'o-', label='Euler')
plt.plot(h_lst, err_heun, 'o-', label='Heun')
plt.plot(h_lst, err_rku4, 'o-', label='RK4')
plt.plot(h_lst, err_pc4m, 'o-', label='PC4')

plt.axvline(1.30, color='k', label=r'Critical Step $h=1.30$')
plt.axvline(2.00, color='k', label=r'Critical Step $h=2.00$')
plt.axvline(2.79, color='k', label=r'Critical Step $h=2.79$')

plt.legend(loc='best')
plt.savefig('images/err_N.png')
plt.show()



# Error with constant T

# set interval size
T = 150

# setup of h
hmin = 0.001
hmax = 2.0
hnum = 1000
h_lst = hmin + (hmax-hmin)*(np.linspace(0, 1, hnum))**2

# lists of errors for different methods
err_eule = []
err_heun = []
err_rku4 = []
err_rk45 = []
err_pc4m = []

# look though all step sizes
for h in h_lst:

	# indicator
	print(f'Calculating for h={h}')

	# list of time points
	t = np.linspace(0, T, int(T/h))

	# calculate error
	err_eule.append(np.average(np.abs(de.euler(f, T0, t) - exact(t))))
	err_heun.append(np.average(np.abs(de.heun(f, T0, t) - exact(t))))
	err_rku4.append(np.average(np.abs(de.rku4(f, T0, t) - exact(t))))
	err_pc4m.append(np.average(np.abs(de.pc4(f, T0, t) - exact(t))))

#Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Step Size $h$')
plt.ylabel(r'Error Estimate $\mathrm{avg} |x_i - x_i^{\mathrm{exact}}|$')
plt.title(r'Error Estimate as a Function of Step Size, Constant $T$')
plt.ylim(10**(-16), 10**(1))
plt.yscale('log')
plt.grid()

plt.plot(h_lst, err_eule, 'o-', label='Euler')
plt.plot(h_lst, err_heun, 'o-', label='Heun')
plt.plot(h_lst, err_rku4, 'o-', label='RK4')
plt.plot(h_lst, err_pc4m, 'o-', label='PC4')

plt.legend(loc='best')
plt.savefig('images/err_T.png')
plt.show()



# RKF45 error 

# set interval size
T = 150

# setup of h
hmin = 10**(-6)
hmax_lst = [2.0, 3.0, 4.0]

# tolerance
eps = 10**(-8)

# error and solution lists
num_sol = []
err = []

for hmax in hmax_lst:

	# RKF45 solution
	y = de.rkf(f, 0, T, T0, eps, hmax, hmin)
	num_sol.append(y)

	# calculate error
	err.append(np.abs(y[1] - exact(y[0])))

#Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Time $\tau$')
plt.ylabel(r'Error $|x - x^{\mathrm{exact}}|$')
plt.title(fr'Runge-Kutta-Fehlberg 4(5) Error, $T={T}$, $\epsilon = {eps:.3e}$')
plt.ylim(10**(-16), 10**(-4))
plt.yscale('log')
plt.grid()

plt.plot(num_sol[0][0], err[0], 'o-', label='$h_{\mathrm{max}} < 2.79$')
plt.plot(num_sol[1][0], err[1], 'o-', label='$2.79 < h_{\mathrm{max}} < 3.22$')
plt.plot(num_sol[2][0], err[2], 'o-', label='$h_{\mathrm{max}} > 3.22$')
plt.axhline(eps, color='k')

plt.legend(loc='best')
plt.savefig('images/err_RKF45.png')
plt.show()



# Speed with constant T

# set iterval size
T = 150

# number of points
Nmin = 1
Nmax = 10**4
N_lst = range(Nmin, Nmax)

# lists of times for different methods
tim_eule = []
tim_heun = []
tim_rku4 = []
tim_pc4m = []

# look though all numbers N
for N in N_lst:

	# indicator
	print(f'Calculating for N={N}')
	
	# list of time points
	t = np.linspace(0, T, N)

	# measure speed for euler
	start = timer()
	y = de.euler(f, T0, t)
	end = timer()
	tim_eule.append(end-start)

	# measure speed for heun
	start = timer()
	y = de.heun(f, T0, t)
	end = timer()
	tim_heun.append(end-start)

	# measure speed for rku4
	start = timer()
	y = de.rku4(f, T0, t)
	end = timer()
	tim_rku4.append(end-start)

	# measure speed for pc4
	start = timer()
	y = de.pc4(f, T0, t)
	end = timer()
	tim_pc4m.append(end-start)


# Plotting
fig, axs = plt.subplots(1)

plt.xlabel(r'Number of Points $N$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.title(r'Evaluation Time as a Function of Point Number, Constant T')
plt.yscale('log')
plt.xscale('log')
plt.grid()

plt.plot(N_lst, tim_eule, 'o-', label='Euler')
plt.plot(N_lst, tim_heun, 'o-', label='Heun')
plt.plot(N_lst, tim_rku4, 'o-', label='RK4')
plt.plot(N_lst, tim_pc4m, 'o-', label='PC4')

plt.legend(loc='best')
plt.savefig('images/times_T_loglog.png')
plt.show()



# RKF45 speed for different tolerances

# set interval size
T = 150

# setup of h
hmin = 10**(-8)
hmax = [2.0, 3.0, 4.0]

# tolerance setup
ln_eps_min = -12
ln_eps_max = 0
eps_num = 1000
eps_lst = np.exp(np.linspace(ln_eps_min, ln_eps_max, eps_num))

# error list
tim_sub = []
tim_btw = []
tim_sup = []

for eps in eps_lst:

	# measure speed for RKF45, hmax sub
	start = timer()
	y = de.rkf(f, 0, T, T0, eps, hmax[0], hmin)
	end = timer()
	tim_sub.append(end-start)

	# measure speed for RKF45, hmax sub
	start = timer()
	y = de.rkf(f, 0, T, T0, eps, hmax[1], hmin)
	end = timer()
	tim_btw.append(end-start)

	# measure speed for RKF45, hmax sub
	start = timer()
	y = de.rkf(f, 0, T, T0, eps, hmax[2], hmin)
	end = timer()
	tim_sup.append(end-start)

#Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Tolerance $\epsilon$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.title(fr'Runge-Kutta-Fehlberg 4(5) Evaluation Time, $T={T}$')
plt.yscale('log')
plt.xscale('log')
plt.grid()

plt.plot(eps_lst, tim_sub, 'o-', label='$h_{\mathrm{max}} < 2.79$')
plt.plot(eps_lst, tim_btw, 'o-', label='$2.79 < h_{\mathrm{max}} < 3.22$')
plt.plot(eps_lst, tim_sup, 'o-', label='$h_{\mathrm{max}} > 3.22$')

plt.legend(loc='best')
plt.savefig('images/times_RKF45_eps.png')
plt.show()


# Initial condition
x0 = 1

# Additional differential equation in standard form
def g(x, t):
	return -x + np.sin(2.617993878*(t + 1))



# Constant-step solutions

# set interval size
T = 100

# setup of h
h = 0.1

# list of time points
t = np.linspace(0, T, int(T/h))

# calculate
sol_eule = de.euler(g, x0, t)
sol_heun = de.heun(g, x0, t)
sol_rku4 = de.rku4(g, x0, t)
sol_pc4m = de.pc4(g, x0, t)

#Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Time $\tau$')
plt.ylabel(r'Solution $x(\tau)$')
plt.title(fr'Solution of Additional Problem, $T={T}$, $h={h}$')
plt.grid()

plt.plot(t, sol_eule, '-', label='Euler')
plt.plot(t, sol_heun, '-', label='Heun')
plt.plot(t, sol_rku4, '-', label='RK4')
plt.plot(t, sol_pc4m, '-', label='PC4')
plt.axhline(-0.3568268606, color='k')
plt.axhline(0.3568268606, color='k')

plt.legend(loc='best')
plt.savefig('images/add_sol_small.png')
plt.show()



# Variable-step solutions

# set interval size
T = 100

# setup of h and tolerance
hmin = 10**(-3)
hmax = 1.0
eps = 10**(-3)

# calculate
sol_rkf45 = de.rkf(g, 0, T, x0, eps, hmin, hmax)

#Plotting
figure, axes = plt.subplots(1)

plt.xlabel(r'Time $\tau$')
plt.ylabel(r'Solution $x(\tau)$')
plt.title(fr'Solution of Additional Problem, $T={T}$, $\epsilon = {eps:.3e}$, '\
		  fr'$h_{{\mathrm{{max}}}}={hmax}$')
plt.grid()

plt.plot(sol_rkf45[0], sol_rkf45[1], '-', label='RKF4(5)')
plt.axhline(-0.3568268606, color='k')
plt.axhline(0.3568268606, color='k')

plt.legend(loc='best')
plt.savefig('images/add_sol_RKF45.png')
plt.show()
