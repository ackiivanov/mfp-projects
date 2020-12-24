import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

from timeit import default_timer as timer
from scipy import odr

import pandas as pd


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# Discrete Fourier Transform
def dft(x, shift=False):
	'''Computes the Discrete Fourier Transform of an array.
	Shifting is done if enabled.'''

	# convert x to np.array
	x = np.asarray(x, dtype=complex)

	# size of array
	N = x.shape[0]

	# positions in f-space
	k = np.arange(N)

	# positions in t-space
	n = k.reshape((N, 1))

	# exponential
	E = np.exp(-2j*np.pi * n*k / N)

	# shift if necessary
	if shift:
		return E@x * np.exp(1j * np.pi * k)
	
	return E@x


def idft(x, shift=False):
	'''Computes the Discrete Inverse Fourier Transform of an array.
	Shifting is done if enabled.'''

	# convert x to np.array
	x = np.asarray(x, dtype=complex)

	# size of array
	N = x.shape[0]

	# positions in f-space
	k = np.arange(N)

	# positions in t-space
	n = k.reshape((N, 1))

	# exponential
	E = np.exp(2j*np.pi * n*k / N)

	# shift if necessary
	if shift:
		return E@x * np.exp(-1j * np.pi * k)
	
	return E@x



# FT of Gaussian

# function that returns gaussian
def gaussian(x, mu=0, sigma=1):
	
	# convert x to array
	x = np.asarray(x, dtype=float)

	return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))

# number of points
N = 1000

# t-range
tmin = -50
tmax = 50
T = tmax - tmin
t_lst = np.linspace(tmin, tmax, N, endpoint=False)

# f-range
nuc = N/(2*T)
nu_lst = np.linspace(-nuc, nuc, N, endpoint=False)

# sampling and critical frequencies
print('Sampling frequency: {}'.format(N/T))
print('Critical frequency: {}'.format(nuc))

# gaussian signal
mu = 0
sigma = 1
sig = gaussian(t_lst, mu=mu, sigma=sigma)


# plotting the gaussian
plt.title(r'Gaussian')
plt.xlabel(r'Time $t$')
plt.ylabel(fr'Signal $N({mu},{sigma})$')
plt.grid()

t_dummy = np.linspace(tmin, tmax, 1000)

plt.plot(t_dummy, gaussian(t_dummy, mu=mu, sigma=sigma), '-', label='Curve')
plt.plot(t_lst, sig, 'ro', label='Samples')

plt.legend(loc='best')
plt.savefig('images/gauss_sig.png')
plt.show()


# fourier transform
ftr = dft(sig, shift=True)/N
#ftr = dft(sig)/N
ftr_rolled = np.roll(ftr, N//2)


# Plotting fourier transform
fig, axs = plt.subplots(3, 1, sharex=True)

# Real part
ax = axs[0]
ax.set_ylabel(r'Real $\Re(\mathcal{F})$')
ax.grid()

ax.plot(nu_lst, np.real(ftr_rolled), 'b-')

# Imaginary part
ax = axs[1]
ax.set_ylabel(r'Imaginary $\Im(\mathcal{F})$')
ax.grid()

ax.plot(nu_lst, np.imag(ftr_rolled), 'r-')

# Plot spectral power
ax = axs[2]
ax.set_ylabel(r'Magnitude $|\mathcal{F}|$')
ax.grid()

ax.plot(nu_lst, np.abs(ftr_rolled), 'g-')

# beautification
fig.suptitle('Fourier Transform of Gaussian')
ax.set_xlabel(r'Frequency $\nu$')
fig.align_ylabels(axs)

plt.savefig('images/gauss_ft.png')
plt.show()


# inverse transform
iftr = idft(ftr)
iftr_rolled = np.roll(iftr, N//2)


# Plotting inverse fourier transform
fig, axs = plt.subplots(3, 1, sharex=True)

# Real part
ax = axs[0]
ax.set_ylabel(r'Real $\Re(\mathcal{F}^{-1})$')
ax.grid()

ax.plot(t_lst, np.real(iftr_rolled), 'b-')

# Imaginary part
ax = axs[1]
ax.set_ylabel(r'Imaginary $\Im(\mathcal{F}^{-1})$')
ax.grid()

ax.plot(t_lst, np.imag(iftr_rolled), 'r-')

# Plot spectral power
ax = axs[2]
ax.set_ylabel(r'Magnitude $|\mathcal{F}^{-1}|$')
ax.grid()

ax.plot(t_lst, np.abs(iftr_rolled), 'g-')

# beautification
fig.suptitle('Inverse Fourier Transform of Gaussian')
ax.set_xlabel(r'Time $t$')
fig.align_ylabels(axs)

plt.savefig('images/gauss_ift.png')
plt.show()


# error plot
plt.title(r'Error Estimate for Gaussian')
plt.xlabel(r'Time $t$')
plt.ylabel(fr'Error Estimate $|N({mu},{sigma}) - $' + 
		   fr'$\mathcal{{F}}^{{-1}}(\mathcal{{F}}(N({mu},{sigma})))|$')
plt.yscale('log')
plt.grid()

plt.plot(t_lst, np.abs(sig - iftr_rolled), '-o')

plt.savefig('images/gauss_prec.png')
plt.show()



# FT of Regular Periodic Function

def per_func(x, nu1, nu2, nu3, A1=1, A2=1, A3=1):
	return (A1*np.sin(2*np.pi*nu1*x) + A2*np.sin(2*np.pi*nu2*x) +
			A3*np.sin(2*np.pi*nu3*x))


# number of points
N = 200

# t-range
tmin = 0
tmax = 240
T = tmax - tmin
t_lst = np.linspace(tmin, tmax, N, endpoint=False)

# f-range
nuc = N/(2*T)
nu_lst = np.linspace(-nuc, nuc, N, endpoint=False)

# sampling and critical frequencies
print('Sampling frequency: {}'.format(N/T))
print('Critical frequency: {}'.format(nuc))

# regular periodic signal
nu1 = 1/30
nu2 = 1/20
nu3 = 1/10
A1 = 1
A2 = -4
A3 = 9
sig = per_func(t_lst, nu1, nu2, nu3, A1, A2, A3)

# plotting the periodic signal
plt.title(r'Regular Periodic Signal')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Signal')
plt.grid()

t_dummy = np.linspace(tmin, tmax, 1000)

plt.plot(t_dummy, per_func(t_dummy, nu1, nu2, nu3, A1, A2, A3), '-', label='Curve')
plt.plot(t_lst, sig, 'ro', label='Samples')

plt.legend(loc='best')
plt.savefig('images/regper_sig.png')
plt.show()


# fourier transform
#ftr = dft(sig, shift=True)/N
ftr = dft(sig)/N
ftr_rolled = np.roll(ftr, N//2)


# Plotting fourier transform
fig, axs = plt.subplots(3, 1, sharex=True)

# Real part
ax = axs[0]
ax.set_ylabel(r'Real $\Re(\mathcal{F})$')
ax.grid()

ax.plot(nu_lst, np.real(ftr_rolled), 'b-')

# Imaginary part
ax = axs[1]
ax.set_ylabel(r'Imaginary $\Im(\mathcal{F})$')
ax.grid()

ax.plot(nu_lst, np.imag(ftr_rolled), 'r-')

# Plot spectral power
ax = axs[2]
ax.set_ylabel(r'Magnitude $|\mathcal{F}|$')
ax.grid()

ax.plot(nu_lst, np.abs(ftr_rolled), 'g-')

# beautification
fig.suptitle('Fourier Transform of a Regular Periodic Function')
ax.set_xlabel(r'Frequency $\nu$')
fig.align_ylabels(axs)

plt.savefig('images/regper_ft.png')
plt.show()


# inverse transform
iftr = idft(ftr)
#iftr_rolled = np.roll(iftr, N//2)


# Plotting inverse fourier transform
fig, axs = plt.subplots(3, 1, sharex=True)

# Real part
ax = axs[0]
ax.set_ylabel(r'Real $\Re(\mathcal{F}^{-1})$')
ax.grid()

ax.plot(t_lst, np.real(iftr), 'b-')

# Imaginary part
ax = axs[1]
ax.set_ylabel(r'Imaginary $\Im(\mathcal{F}^{-1})$')
ax.grid()

ax.plot(t_lst, np.imag(iftr), 'r-')

# Plot spectral power
ax = axs[2]
ax.set_ylabel(r'Magnitude $|\mathcal{F}^{-1}|$')
ax.grid()

ax.plot(t_lst, np.abs(iftr), 'g-')

# beautification
fig.suptitle('Inverse Fourier Transform of a Regular Periodic Function')
ax.set_xlabel(r'Time $t$')
fig.align_ylabels(axs)

plt.savefig('images/regper_ift.png')
plt.show()


# error plot
plt.title(r'Error Estimate for a Regular Periodic Function')
plt.xlabel(r'Time $t$')
plt.ylabel(fr'Error Estimate $|f(t) - $' + 
		   r'$\mathcal{F}^{-1}(\mathcal{F}(f))(t)|$')
plt.yscale('log')
plt.grid()

plt.plot(t_lst, np.abs(sig - iftr), '-o')

plt.savefig('images/regper_prec.png')
plt.show()



# FT of Irregular Periodic Function

def per_func(x, nu1, nu2, nu3, A1=1, A2=1, A3=1):
	return (A1*np.sin(2*np.pi*nu1*x) + A2*np.sin(2*np.pi*nu2*x) +
			A3*np.sin(2*np.pi*nu3*x))


# number of points
N = 100

# t-range
tmin = 0
tmax = 70
T = tmax - tmin
t_lst = np.linspace(tmin, tmax, N, endpoint=False)

# f-range
nuc = N/(2*T)
nu_lst = np.linspace(-nuc, nuc, N, endpoint=False)

# sampling and critical frequencies
print('Sampling frequency: {}'.format(N/T))
print('Critical frequency: {}'.format(nuc))

# regular periodic signal
nu1 = 1/(3*np.pi**2)
nu2 = 1/7
nu3 = 1/2.7358
A1 = 3
A2 = -7
A3 = 9
sig = per_func(t_lst, nu1, nu2, nu3, A1, A2, A3)

# plotting the periodic signal
plt.title(r'Irregular Periodic Signal')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Signal')
plt.grid()

t_dummy = np.linspace(tmin, tmax, 1000)

plt.plot(t_dummy, per_func(t_dummy, nu1, nu2, nu3, A1, A2, A3), '-', label='Curve')
plt.plot(t_lst, sig, 'ro', label='Samples')

plt.legend(loc='best')
plt.savefig('images/irregper_sig.png')
plt.show()


# fourier transform
#ftr = dft(sig, shift=True)/N
ftr = dft(sig)/N
ftr_rolled = np.roll(ftr, N//2)


# Plotting fourier transform
fig, axs = plt.subplots(3, 1, sharex=True)

# Real part
ax = axs[0]
ax.set_ylabel(r'Real $\Re(\mathcal{F})$')
ax.grid()

ax.plot(nu_lst, np.real(ftr_rolled), 'b-')

# Imaginary part
ax = axs[1]
ax.set_ylabel(r'Imaginary $\Im(\mathcal{F})$')
ax.grid()

ax.plot(nu_lst, np.imag(ftr_rolled), 'r-')

# Plot spectral power
ax = axs[2]
ax.set_ylabel(r'Magnitude $|\mathcal{F}|$')
ax.grid()

ax.plot(nu_lst, np.abs(ftr_rolled), 'g-')

# beautification
fig.suptitle('Fourier Transform of an Irregular Periodic Function')
ax.set_xlabel(r'Frequency $\nu$')
fig.align_ylabels(axs)

plt.savefig('images/irregper_ft.png')
plt.show()


# inverse transform
iftr = idft(ftr)
#iftr_rolled = np.roll(iftr, N//2)


# Plotting inverse fourier transform
fig, axs = plt.subplots(3, 1, sharex=True)

# Real part
ax = axs[0]
ax.set_ylabel(r'Real $\Re(\mathcal{F}^{-1})$')
ax.grid()

ax.plot(t_lst, np.real(iftr), 'b-')

# Imaginary part
ax = axs[1]
ax.set_ylabel(r'Imaginary $\Im(\mathcal{F}^{-1})$')
ax.grid()

ax.plot(t_lst, np.imag(iftr), 'r-')

# Plot spectral power
ax = axs[2]
ax.set_ylabel(r'Magnitude $|\mathcal{F}^{-1}|$')
ax.grid()

ax.plot(t_lst, np.abs(iftr), 'g-')

# beautification
fig.suptitle('Inverse Fourier Transform of an Irregular Periodic Function')
ax.set_xlabel(r'Time $t$')
fig.align_ylabels(axs)

plt.savefig('images/irregper_ift.png')
plt.show()


# continue reconstruction over larger domain
iftr_contd = np.hstack([iftr, iftr[:N//2]])

# error plot
plt.title(r'Error Estimate for an Irregular Periodic Function')
plt.xlabel(r'Time $t$')
plt.ylabel(fr'Error Estimate $|f(t) - $' + 
		   r'$\mathcal{F}^{-1}(\mathcal{F}(f))(t)|$')
plt.yscale('log')
plt.grid()

t_dummy = np.hstack([t_lst, np.linspace(tmax, tmax + T/2, N//2)])

plt.plot(t_dummy, np.abs(per_func(t_dummy, nu1, nu2, nu3, A1, A2, A3)
		 - iftr_contd), '-o')

plt.savefig('images/irregper_prec.png')
plt.show()



# Example of aliasing with Gaussian

# function that returns f-sgifted gaussian
def gaussian(x, mu=0, sigma=1, f_mu=0):
	
	# convert x to array
	x = np.asarray(x, dtype=float)

	return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2)+2j*np.pi*f_mu*x)

# number of points
N = 1000

# t-range
tmin = -50
tmax = 50
T = tmax - tmin
t_lst = np.linspace(tmin, tmax, N, endpoint=False)

# f-range
nuc = N/(2*T)
nu_lst = np.linspace(-nuc, nuc, N, endpoint=False)

# sampling and critical frequencies
print('Sampling frequency: {}'.format(N/T))
print('Critical frequency: {}'.format(nuc))

# gaussian signal
mu = 0
sigma = 0.1
f_mu = 3
sig = gaussian(t_lst, mu=mu, sigma=sigma, f_mu=f_mu)


# fourier transform
ftr = dft(sig, shift=True)/N
#ftr = dft(sig)/N
ftr_rolled = np.roll(ftr, N//2)


# Plotting fourier transform
plt.title(r'Fourier Transform of Frequency-shifted Gaussian')
plt.xlabel(r'Frequency $\nu$')
plt.ylabel(r'Real $\Re(\mathcal{F})$')
plt.grid()

nu_dummy = np.linspace(-nuc, nuc, 1000)

# results of theoretical prediction
plt.plot(nu_dummy, np.real(1/np.sqrt(2*np.pi)*gaussian(nu_dummy, mu=f_mu,
		 sigma=1/(2*np.pi*sigma))), '-', label='Actual Fourier Transform')

# results dft calculation
plt.plot(nu_lst, np.real(N/T * ftr_rolled), 'r-', label='Discrete Fourier Transform')

plt.legend(loc='best')
plt.savefig('images/aliasing.png')
plt.show()



# Speed of dft

# list of different N
N_lst = range(10, 1000)

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
	y = dft(x)
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
nstd = 5.0
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
plt.title(r'Evaluation Time as a Function of Array Size')
plt.grid()

plt.plot(x, y, 'ko', label='Data')
plt.plot(x_dummy, fit, label=r'Fit $\ln(t/[s]) = A \ln(N) + B$')
axs.fill_between(x_dummy, fit_up, fit_dw, alpha=0.25, label='Confidence interval')

text = (r'$A=({:0.3f}\pm{:0.3f})$'.format(betaout[0], betaerr[0])
		+'\n'+r'$B=({:0.3f}\pm{:0.3f})$'.format(betaout[1],betaerr[1]))
boxprops = dict(boxstyle='round', facecolor='ivory', alpha=0.5)
axs.text(0.05, 0.95, text, transform=axs.transAxes, fontsize=14,
		  verticalalignment='top', bbox=boxprops)

plt.legend(loc='lower right')
plt.savefig('images/times.png')
plt.show()



# Bach samples

# import data and calculate
file_paths = ['data/Bach.882.txt', 'data/Bach.1378.txt', 'data/Bach.2756.txt',
			  'data/Bach.5512.txt', 'data/Bach.11025.txt', 'data/Bach.44100.txt']

# lists for data, length, frequency
nu_range_lst = []
ftr_rolled_lst = []

# lop over all files
for name in file_paths:
	
	# import and save
	x = np.array(pd.read_csv(name))
	N = x.shape[0]
	x = x.reshape(N)

	# critical frequency and range
	nuc = int(name[10:-4])/2
	nu_range_lst.append(np.linspace(-nuc, nuc, N, endpoint=False)[N//2:])

	#fourier transform
	x_rolled = np.fft.fftshift(x)
	ftr = np.fft.fft(x)/N
	ftr_rolled = np.fft.fftshift(ftr)
	ftr_rolled_lst.append(ftr_rolled[N//2:])


# Plotting fourier transform
fig, axs = plt.subplots(2, 3, sharey=True, sharex=True)

# plot 882 sampling
ax = axs[0, 0]
ax.set_xlabel(r'Frequency $\nu$')

ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.grid()

ax.plot(nu_range_lst[0], np.abs(ftr_rolled_lst[0])**2, 'k-')

# plot 1378 sampling
ax = axs[0, 1]
ax.set_xlabel(r'Frequency $\nu$')
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.set_xlim(0,2000)
ax.grid()

ax.plot(nu_range_lst[1], np.abs(ftr_rolled_lst[1])**2, 'k-')

# plot 2756 sampling
ax = axs[0, 2]
ax.set_xlabel(r'Frequency $\nu$')
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.grid()

ax.plot(nu_range_lst[2], np.abs(ftr_rolled_lst[2])**2, 'k-')

# plot 5512 sampling
ax = axs[1, 0]
ax.set_xlabel(r'Frequency $\nu$')
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.grid()

ax.plot(nu_range_lst[3], np.abs(ftr_rolled_lst[3])**2, 'k-')

# plot 11025 sampling
ax = axs[1, 1]
ax.set_xlabel(r'Frequency $\nu$')
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.grid()

ax.plot(nu_range_lst[4], np.abs(ftr_rolled_lst[4])**2, 'k-')

# plot 44100 sampling
ax = axs[1, 2]
ax.set_xlabel(r'Frequency $\nu$')
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.grid()

ax.plot(nu_range_lst[5], np.abs(ftr_rolled_lst[5])**2, 'k-')

# beautification
fig.suptitle('Fourier Transform of Bach Recording')
fig.align_ylabels(axs[:,0])
fig.align_ylabels(axs[:,1])

plt.savefig('images/bach_ft.png')
plt.show()



# Guitar and tuning fork

# import data and calculate
file_paths = ['data/kitara.txt', 'data/vilice440hz.txt']

# lists for data, length, frequency
nu_range_lst = []
ftr_rolled_lst = []

# lop over all files
for name in file_paths:
	
	# import and save
	x = np.array(pd.read_csv(name))
	N = x.shape[0]
	x = x.reshape(N)

	# critical frequency and range
	nuc = 44100/2
	nu_range_lst.append(np.linspace(-nuc, nuc, N, endpoint=False)[N//2:])

	#fourier transform
	x_rolled = np.fft.fftshift(x)
	ftr = np.fft.fft(x)/N
	ftr_rolled = np.fft.fftshift(ftr)
	ftr_rolled_lst.append(ftr_rolled[N//2:])


# Plotting fourier transform of guitar
fig, ax = plt.subplots(1)

# Plot spectral power
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.set_yscale('log')
ax.set_xlim(0, 7000)
ax.grid()

ax.plot(nu_range_lst[0], np.abs(ftr_rolled_lst[0])**2, 'b-')

# beautification
fig.suptitle('Fourier Transform of Guitar Note')
ax.set_xlabel(r'Frequency $\nu$ [1/s]')

plt.savefig('images/guitar_ft.png')
plt.show()


# Plotting fourier transform of tuning fork on guitar soundboard
fig, ax = plt.subplots(1)

# Plot spectral power
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.set_yscale('log')
ax.set_xlim(0, 7000)
ax.grid()

ax.plot(nu_range_lst[1], np.abs(ftr_rolled_lst[1])**2, 'b-')

# beautification
fig.suptitle('Fourier Transform of Tuning Fork on Guitar Soundboard')
ax.set_xlabel(r'Frequency $\nu$ [1/s]')

plt.savefig('images/fork_ft.png')
plt.show()



# AkRes samples

# import data and calculate
file_paths = ['data/poskus1_akres.txt', 'data/poskus1_akres_novi.dat',
			  'data/poskus2_akres.txt', 'data/poskus2_akres_novi.dat',
			  'data/poskus3_akres.txt', 'data/poskus3_akres_novi.dat']

# lists for data, length, frequency
nu_range_lst = []
ftr_rolled_lst = []

# lop over all files
for name in file_paths:

	# import and save
	x = np.array(pd.read_csv(name))
	N = x.shape[0]
	x = x.reshape(N)

	# critical frequency and range
	nuc = 44100/2
	nu_range_lst.append(np.linspace(-nuc, nuc, N, endpoint=False)[N//2:])

	#fourier transform
	x_rolled = np.fft.fftshift(x)
	ftr = np.fft.fft(x)/N
	ftr_rolled = np.fft.fftshift(ftr)
	ftr_rolled_lst.append(ftr_rolled[N//2:])


# Plotting fourier transform
fig, axs = plt.subplots(2, 3, sharey=False, sharex=True)

# akres1
ax = axs[0, 0]
ax.set_title('Careful Hit 1')
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.grid()

ax.plot(nu_range_lst[0], np.abs(ftr_rolled_lst[0])**2, 'k-')

# akres1_new
ax = axs[0, 1]
ax.set_title('Weak Hit')
ax.set_xlim(0, 1600)
ax.grid()

ax.plot(nu_range_lst[1], np.abs(ftr_rolled_lst[1])**2, 'k-')

# akres2
ax = axs[0, 2]
ax.set_title('Careful Hit 2')
ax.grid()

ax.plot(nu_range_lst[2], np.abs(ftr_rolled_lst[2])**2, 'k-')

# akres2_new
ax = axs[1, 0]
ax.set_title('Powerful Hit')
ax.set_xlabel(r'Frequency $\nu$')
ax.set_ylabel(r'Power $|\mathcal{F}|^2$')
ax.grid()

ax.plot(nu_range_lst[3], np.abs(ftr_rolled_lst[3])**2, 'k-')

# akres3
ax = axs[1, 1]
ax.set_title('Careful Hit 3')
ax.set_xlabel(r'Frequency $\nu$')
ax.grid()

ax.plot(nu_range_lst[4], np.abs(ftr_rolled_lst[4])**2, 'k-')

# akres3_new
ax = axs[1, 2]
ax.set_title('Very Powerful Hit')
ax.set_xlabel(r'Frequency $\nu$')
ax.grid()

ax.plot(nu_range_lst[5], np.abs(ftr_rolled_lst[5])**2, 'k-')

# beautification
fig.suptitle('Fourier Transform of Acoustic Resonator Hits')
fig.align_ylabels(axs[:,0])

plt.savefig('images/akres_ft.png')
plt.show()
