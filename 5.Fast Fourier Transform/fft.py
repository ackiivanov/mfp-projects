import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

from timeit import default_timer as timer
from scipy import odr

import pandas as pd


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def Plotter(x, y, kind, name, xname, yname):

	#Plotting
	figure, axes = plt.subplots(1)

	if kind == 'fft':
		plt.xlabel(fr'{xname} $\nu$ [1/s]')
		plt.ylabel(fr'{yname} $|\mathcal{{F}}|$')
		plt.yscale('log')
		plt.xlim(0,10000)
	elif kind == 'acr':
		plt.xlabel(fr'{xname} $t$ [s]')
		plt.ylabel(fr'{yname} $|\mathcal{{A}}|$')
		plt.xlim(0, 0.2)

	plt.title(name)
	plt.grid()

	plt.errorbar(x, y, fmt='k-', label='Data')

	plt.legend(loc='best')
	
	plt.savefig('images/' + name.replace(' ', '_') + '.png')
	plt.show()


# paths for data
file_paths = ['data/bubomono.txt', 'data/bubo2mono.txt', 'data/mix.txt',
			  'data/mix1.txt', 'data/mix2.txt', 'data/mix22.txt']

# time length of files in seconds
T = 5

# dict to store frequencies and fft 
data = {}

# lop over all files
for name in file_paths:

	# import
	x = np.array(pd.read_csv(name))
	N = x.shape[0]
	x = x.reshape(N)

	# frequency range
	nu_range = np.fft.fftfreq(N, d=T/N)

	# fourier transform
	ftr = np.fft.fft(x)/N

	# autocorrelation time range
	t_range = np.linspace(-T, T, 2*N-1)

	# autocorrelation
	acr = np.correlate(x, x, mode='full')

	# sava to data
	data[name] = [nu_range[:N//2], ftr[:N//2], t_range, acr]


# plot ffts
for name in file_paths:

	# run plotter for fourier transform
	Plotter(data[name][0], np.abs(data[name][1]), 'fft',
			f'Fourier Transform of {name[5:]}', 'Frequency', 'Fourier Transform')

	# run plotter for autocorrelation
	Plotter(data[name][2], data[name][3], 'acr',
			f'Autocorrelation of {name[5:]}', 'Time', 'Autocorrelation')



# Speed of fft

# list of different N
N_lst = range(3000, 50000)

# list of times
times = []

# loop over all N
for N in N_lst:
	# indicator
	print(f'Calculating for {N}')
	
	# random array
	x = np.random.random(N)

	# measure time
	start = timer()
	y = np.fft.fft(x)
	end = timer()
	times.append(end-start)


# N ln N fit

# Model function and object
def fit_function(beta, x):
	A, B = beta
	return A * x * np.log(x) + B

model = odr.Model(fit_function)

# Log data object
x = N_lst
y = times

data = odr.RealData(x, y)

# Set up ODR with model and data
odrm = odr.ODR(data, model, beta0=[1.0, 1.0])

# Run regression
out = odrm.run()

# Extract parameters
betaout = out.beta
betaerr = out.sd_beta
print('Fit parameters and their std deviations')
print('-------------------')
for i in range(len(betaout)):
	print('Parameter {}: '.format(i + 1) + str(betaout[i]) + ' +- ' + str(betaerr[i]))

# Fit curve and confidence intervals
nstd = 1.0
betaout_up = betaout + nstd * betaerr
betaout_dw = betaout - nstd * betaerr

x_dummy = np.linspace(min(x), max(x), 1000)
fit = fit_function(betaout, x_dummy)
fit_up = fit_function(betaout_up, x_dummy)
fit_dw = fit_function(betaout_dw, x_dummy)

# Plotting
fig, axs = plt.subplots(1)

plt.xlabel(r'Size of Array $N$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.yscale('log')
plt.title(r'Evaluation Time of FFT as a Function of Array Size')
plt.grid()

plt.plot(x, y, 'ko', label='Data')
plt.plot(x_dummy, fit, label=r'Fit $t = A\, N \ln(N) + B$')
axs.fill_between(x_dummy, fit_up, fit_dw, alpha=0.25, label='Confidence interval')

plt.legend(loc='lower right')
plt.savefig('images/fft_times.png')
plt.show()



# Speed of autocorrelation

# list of different N
N_lst = range(100, 15000)

# list of times
times = []

# loop over all N
for N in N_lst:
	# indicator
	print(f'Calculating for {N}')
	
	# random array
	x = np.random.random(N)

	# measure time
	start = timer()
	y = np.correlate(x, x, mode='full')
	end = timer()
	times.append(end-start)


# Power law fit

# Model function and object
def fit_function(beta, x):
	A, B = beta
	return A * x + B

model = odr.Model(fit_function)

# Log data object
x = np.log(N_lst)
y = np.log(times)

data = odr.RealData(x, y)

# Set up ODR with model and data
odrm = odr.ODR(data, model, beta0=[1.0, 1.0])

# Run regression
out = odrm.run()

# Extract parameters
betaout = out.beta
betaerr = out.sd_beta
print('Fit parameters and their std deviations')
print('-------------------')
for i in range(len(betaout)):
	print('Parameter {}: '.format(i + 1) + str(betaout[i]) + ' +- ' + str(betaerr[i]))

# Fit curve and confidence intervals
nstd = 1.0
betaout_up = betaout + nstd * betaerr
betaout_dw = betaout - nstd * betaerr

x_dummy = np.linspace(min(x), max(x), 1000)
fit = fit_function(betaout, x_dummy)
fit_up = fit_function(betaout_up, x_dummy)
fit_dw = fit_function(betaout_dw, x_dummy)

# Plotting
fig, axs = plt.subplots(1)

plt.xlabel(r'Size of Array $\ln(N)$')
plt.ylabel(r'Evaluation Time $\ln \left( t/[s] \right)$')
plt.title(r'Evaluation Time of Autocorrelation as a Function of Array Size')
plt.grid()

plt.plot(x, y, 'ko', label='Data')
plt.plot(x_dummy, fit, label=r'Fit $\ln(t) = A\, \ln(N) + B$')
axs.fill_between(x_dummy, fit_up, fit_dw, alpha=0.25, label='Confidence interval')

text = (r'$A=({:0.3f}\pm{:0.3f})$'.format(betaout[0], betaerr[0])+'\n'+
	    r'$B=({:0.3f}\pm{:0.3f})$'.format(betaout[1],betaerr[1]))
boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
axs.text(0.05, 0.95, text, transform=axs.transAxes, fontsize=14,
		  verticalalignment='top', bbox=boxprops)

plt.legend(loc='lower right')
plt.savefig('images/acr_times.png')
plt.show()
