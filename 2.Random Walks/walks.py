import random as rd 
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import scipy.stats as sp
from scipy import odr


# Random seed setup
# the module random automatically uses system time as the seed

# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Return random Pareto distributed number
def rand_r(alpha, x_m=1):
	return x_m * (rd.random())**(-1/alpha)

# Return random angle
def rand_phi():
	return 2 * np.pi * rd.random()


#Levy flights
def flights(n_min, n_max, n_step, N, mu_min, mu_max, mu_step,
		  plot_walk=False, plot_fit=False, plot_gamma=True):
	r"""
	flights runs a simulation of Levy flights and calculates the dependence
	$\gamma(\mu)$ where $\gamma$ is defined by $\sigma^2(n) \propto n^{\gamma}$
	and $\mu - 1$ is the Pareto distribution power.

	Input:
	n_min - minimal number of steps in the walk
	n_max - maximal number of steps in the walk
	n_step - step size to walk the interval [n_min, n_max] with
	N - number of iterations for each set of parameters
	mu_min - minimal $\mu$ (values less than 1 don't make sense)
	mu_max - maximal $\mu$
	mu_step - step size to walk the interval [mu_min, mu_max] with
	plot_walk(=False) - plots the walk in the x-y plane
	plot_fit(=False) - plots the fit for getting $\gamma$
	plot_gamma(=True) - plots the $\gamma$ vs. $\mu$ curve

	Output:
	None
	"""
	# list of different n
	n_lst = range(n_min, n_max, n_step)

	# list of different mu
	mu_lst = np.linspace(mu_min, mu_max, mu_step)

	# list of the output gamma and its error
	gamma_lst = []
	gamma_err_lst = []

	# loop over different mu
	for mu in mu_lst:
		# table of MAD and its error
		MAD_sq = []
		MAD_sq_err = []

		# print parameters
		print('Flights simulatation with N={}, mu={:.3f}'.format(N, mu))

		# loop over the list of different n
		for n in n_lst:
			# Progress bar (69 can be whatever integer)		
			print('#'*(69*n//n_max), end='\r', flush=True)

			# loop over N times to build statistics
			final_r = []
			for _ in range(N):
				# Lists for x and y if plotting the walk
				if plot_walk:
					x_lst = [0]
					y_lst = [0]

				# do n steps starting from (0,0)
				x = 0
				y = 0
				for _ in range(n):
					# pick r and phi
					r = rand_r(mu - 1)
					phi = rand_phi()

					# increment x and y
					x = x + r * np.cos(phi)
					y = y + r * np.sin(phi)

					# add x and y to list if plotting walk
					if plot_walk:
						x_lst.append(x)
						y_lst.append(y)

				# plot the walk if requested
				if plot_walk:
					plt.title(r'Radnom flight, $\mu = {}$, $n = {}$'.format(mu, n))
					plt.xlabel(r'$x$')
					plt.ylabel(r'$y$')

					plt.plot(x_lst, y_lst)
					plt.plot([0, x_lst[-1]], [0, y_lst[-1]], 'ro')

					plt.grid()
					plt.show()

				# record final r
				final_r.append((x**2+y**2)**(1/2))

			# calculate MAD**2 for the N final radii
			MAD_sq.append((sp.median_abs_deviation(final_r))**2)
			MAD_sq_err.append(0.181/(N)**(1/2))

		# Model function and object
		def fit_function(beta, x):
			A, B = beta
			return A * x + B

		model = odr.Model(fit_function)

		# Data object setup for log values
		x = np.log(n_lst)
		y = np.log(MAD_sq)
		sy = np.divide(MAD_sq_err, MAD_sq)

		data = odr.RealData(x, y, sy=sy)

		#Set up ODR with model and data
		odrm = odr.ODR(data, model, beta0=[1.0, 1.0])

		#Run regression
		out = odrm.run()

		#Extract parameters
		betaout = out.beta
		betaerr = out.sd_beta

		if plot_fit:
			# Fit curve and confidence intervals
			nstd = 1.0
			betaout_up = betaout + nstd * betaerr
			betaout_dw = betaout - nstd * betaerr

			x_dummy = np.linspace(min(x), max(x), 1000)
			fit = fit_function(betaout, x_dummy)
			fit_up = fit_function(betaout_up, x_dummy)
			fit_dw = fit_function(betaout_dw, x_dummy)

			# Plotting
			figure, axes = plt.subplots(1)

			plt.title(r'Power law fit for $\ln(\mathrm{MAD}^2(r))$ vs. $\ln(t)$')
			plt.xlabel(r'Time $\ln(t)$')
			plt.ylabel(r'$\hat{\sigma}^2$ estimator $\ln(\mathrm{MAD}^2(r))$')
			plt.grid()

			plt.errorbar(x, y, yerr=sy, fmt='ko', label='Data')
			plt.plot(x_dummy, fit,
					 label=r'Fit $\ln(\mathrm{MAD}^2(r)) = \gamma \ln(t) + C$')
			axes.fill_between(x_dummy, fit_up, fit_dw, alpha=0.25,
							  label='Confidence interval')

			text = (r"$\gamma=({:0.3f}\pm{:0.3f})$".format(betaout[0], betaerr[0])
					+"\n"+r"$C=({:0.3f}\pm{:0.3f})$".format(betaout[1],betaerr[1]))
			boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
			axes.text(0.05, 0.95, text, transform=axes.transAxes, fontsize=14,
	    	    verticalalignment='top', bbox=boxprops)

			plt.legend(loc='lower right')
			plt.show()

		# store the gamma
		gamma_lst.append(betaout[0])
		gamma_err_lst.append(betaerr[0])


		# Reset printed output
		print('')

	# Plot gamma vs. mu graph if reqired
	if plot_gamma:
		def prediction(mu):
			if mu >= 1 and mu <= 3:
				return 2/(mu - 1)
			elif mu >= 3:
				return 1
			# won't throw an error if unexpected values are put in for mu
			else:
				return np.nan

		plt.title(r'Anomalous diffusion (flights)')
		plt.xlabel(r'Pareto distribution power $\mu$')
		plt.ylabel(r'$\hat{\sigma}^2$ power $\gamma$')
		plt.ylim(-1, 27) # make more general
		plt.grid()

		mu_dummy = np.linspace(mu_min, mu_max, 1000)

		plt.errorbar(mu_lst, gamma_lst, yerr=gamma_err_lst, fmt='ko', label='Simulated')
		plt.plot(mu_dummy, [prediction(mu) for mu in mu_dummy], label='Theoretical')

		plt.legend(loc='upper right')
		plt.savefig('images/' + 'flights_gamma_mu.png', dpi=1200)
		plt.show()


#Levy walks
def walks(t_min, t_max, t_step, N, mu_min, mu_max, mu_step,
		  plot_walk=False, plot_fit=False, plot_gamma=True):
	r"""
	walks runs a simulation of Levy walks and calculates the dependence
	$\gamma(\mu)$ where $\gamma$ is defined by $\sigma^2(n) \propto n^{\gamma}$
	and $\mu - 1$ is the Pareto distribution power.

	Input:
	t_min - minimal time for a walk
	t_max - maximal time for a walk
	t_step - step size to walk the interval [t_min, t_max] with
	N - number of iterations for each set of parameters
	mu_min - minimal $\mu$ (values less than 1 don't make sense)
	mu_max - maximal $\mu$
	mu_step - step size to walk the interval [mu_min, mu_max] with
	plot_walk(=False) - plots the walk in the x-y plane
	plot_fit(=False) - plots the fit for getting $\gamma$
	plot_gamma(=True) - plots the $\gamma$ vs. $\mu$ curve

	Output:
	None
	"""	
	# list of different t
	t_lst = range(t_min, t_max, t_step)

	# list of different mu
	mu_lst = np.linspace(mu_min, mu_max, mu_step)

	# list of the output gamma and its error
	gamma_lst = []
	gamma_err_lst = []

	# loop over different mu
	for mu in mu_lst:
		# table of MAD and its error
		MAD_sq = []
		MAD_sq_err = []

		# print parameters
		print('Walks simulatation with N={}, mu={:.3f}'.format(N, mu))

		# loop over the list of different t
		for final_t in t_lst:
			# Progress bar (69 can be whatever integer)		
			print('#'*(69*final_t//t_max), end='\r', flush=True)

			# loop over N times to build statistics
			final_r = []
			for _ in range(N):
				# Lists for x and y if plotting the walk
				if plot_walk:
					x_lst = [0]
					y_lst = [0]

				# step starting from (0,0) until time becomes final_t
				x = 0
				y = 0
				t = 0
				while t <= final_t:
					# pick r and phi
					r = rand_r(mu - 1)
					phi = rand_phi()

					# increment x, y and t
					x = x + r * np.cos(phi)
					y = y + r * np.sin(phi)
					t = t + r

					# add x and y to list if plotting walk
					if plot_walk:
						x_lst.append(x)
						y_lst.append(y)

				# do a partial step to reach time t
				# pick phi
				phi = rand_phi()

				# increment x and y
				x = x + (final_t - t) * np.cos(phi)
				y = y + (final_t - t) * np.sin(phi)

				# add x and y to list if plotting walk
				if plot_walk:
					x_lst.append(x)
					y_lst.append(y)

				# plot the walk if requested
				if plot_walk:
					plt.title(r'Radnom walk, $\mu = {}$, $t_f = {:.4f}$'.format(mu, t))
					plt.xlabel(r'$x$')
					plt.ylabel(r'$y$')

					plt.plot(x_lst, y_lst)
					plt.plot([0, x_lst[-1]], [0, y_lst[-1]], 'ro')

					plt.grid()
					plt.show()

				# record final r
				final_r.append((x**2+y**2)**(1/2))

			# calculate MAD**2 for the N final radii
			MAD_sq.append((sp.median_abs_deviation(final_r))**2)
			MAD_sq_err.append(0.181/(N)**(1/2))

		# Model function and object
		def fit_function(beta, x):
			A, B = beta
			return A * x + B

		model = odr.Model(fit_function)

		# Data object setup for log values
		x = np.log(t_lst)
		y = np.log(MAD_sq)
		sy = np.divide(MAD_sq_err, MAD_sq)

		data = odr.RealData(x, y, sy=sy)

		#Set up ODR with model and data
		odrm = odr.ODR(data, model, beta0=[1.0, 1.0])

		#Run regression
		out = odrm.run()

		#Extract parameters
		betaout = out.beta
		betaerr = out.sd_beta

		if plot_fit:
			# Fit curve and confidence intervals
			nstd = 1.0
			betaout_up = betaout + nstd * betaerr
			betaout_dw = betaout - nstd * betaerr

			x_dummy = np.linspace(min(x), max(x), 1000)
			fit = fit_function(betaout, x_dummy)
			fit_up = fit_function(betaout_up, x_dummy)
			fit_dw = fit_function(betaout_dw, x_dummy)

			# Plotting
			figure, axes = plt.subplots(1)

			plt.title(r'Power law fit for $\ln(\mathrm{MAD}^2(r))$ vs. $\ln(t)$')
			plt.xlabel(r'Time $\ln(t)$')
			plt.ylabel(r'$\hat{\sigma}^2$ estimator $\ln(\mathrm{MAD}^2(r))$')
			plt.grid()

			plt.errorbar(x, y, yerr=sy, fmt='ko', label='Data')
			plt.plot(x_dummy, fit,
					 label=r'Fit $\ln(\mathrm{MAD}^2(r)) = \gamma \ln(t) + C$')
			axes.fill_between(x_dummy, fit_up, fit_dw, alpha=0.25,
							  label='Confidence interval')

			text = (r"$\gamma=({:0.3f}\pm{:0.3f})$".format(betaout[0], betaerr[0])
					+"\n"+r"$C=({:0.3f}\pm{:0.3f})$".format(betaout[1],betaerr[1]))
			boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
			axes.text(0.05, 0.95, text, transform=axes.transAxes, fontsize=14,
	    	    verticalalignment='top', bbox=boxprops)

			plt.legend(loc='lower right')
			plt.show()

		# store the gamma
		gamma_lst.append(betaout[0])
		gamma_err_lst.append(betaerr[0])


		# Reset printed output
		print('')

	# Plot gamma vs. mu graph if reqired
	if plot_gamma:
		def prediction(mu):
			if mu > 1 and mu <= 2:
				return 2
			elif mu > 2 and mu <=3:
				return 4 - mu
			elif mu > 3:
				return 1
			# won't throw an error if unexpected values are put in for mu
			else:
				return np.nan

		plt.title(r'Anomalous diffusion (walks)')
		plt.xlabel(r'Pareto distribution power $\mu$')
		plt.ylabel(r'$\hat{\sigma}^2$ power $\gamma$')
		plt.grid()

		mu_dummy = np.linspace(mu_min, mu_max, 1000)

		plt.errorbar(mu_lst, gamma_lst, yerr=gamma_err_lst, fmt='ko', label='Simulated')
		plt.plot(mu_dummy, [prediction(mu) for mu in mu_dummy], label='Theoretical')

		plt.legend(loc='upper right')
		plt.savefig('images/' + 'walks_gamma_mu.png', dpi=1200)
		plt.show()



# Run flights simulation 
flights(1000, 10000, 100, 100, 1.05, 4.0, 30)

# Run walks simulation
walks(1000, 10000, 100, 200, 1.05, 4.0, 30)
