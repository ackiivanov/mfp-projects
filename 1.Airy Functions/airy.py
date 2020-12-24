import mpmath as mp
import matplotlib.pyplot as plt
from matplotlib import rc
from timeit import default_timer as timer


# Seting mpmath precision
EPS = 10**(-20)
mp.dps = 20


# Taylor series approximation

# Constants a and b to 19 places
a = mp.mpf(0.355028053887817239)
b = mp.mpf(0.258819403792806798)

# Function f of series
def f(x, er=EPS):
	s = mp.mpf(0)
	term = mp.mpf(1)
	k = mp.mpf(0)
	while mp.fabs(term) > mp.mpf(er):
		s = s + term
		term = term * x**3/(9*k**2 + 15*k + 6)
		k = k + 1

	return s

# Function g of series
def g(x, er=EPS):
	s = mp.mpf(0)
	term = x
	k = mp.mpf(0)
	while mp.fabs(term) > mp.mpf(er):
		s = s + term
		term = term * x**3/(9*k**2 + 21*k + 12)
		k = k + 1

	return s

# Airy Ai function approximated for small positive argument
def Ai_x0(x):
	return a*f(x) - b*g(x)

# Airy Bi function approximated for small positive argument
def Bi_x0(x):
	return mp.sqrt(3) * (a*f(x) + b*g(x))


# Asymptotic approximation

# Function L of series
def L(x, er=EPS):
	s = mp.mpf(0)
	term = mp.mpf(1)
	k = mp.mpf(0)
	new_term = mp.mpf(5/(72*x))
	while mp.fabs(term) > mp.mpf(er) and mp.fabs(new_term) < mp.fabs(term):
		s = s + term
		term = new_term
		new_term = new_term * (6*k+1)*(6*k+5)/(72*(k+1)*x)
		k = k + 1

	return s

# Function P of series
def P(x, er=EPS):
	s = mp.mpf(0)
	term = mp.mpf(1)
	k = mp.mpf(0)
	new_term = mp.mpf(-385/(10368*x**2))
	while mp.fabs(term) > mp.mpf(er) and mp.fabs(new_term) < mp.fabs(term):
		s = s + term
		term = new_term
		new_term = new_term * (12*k+1)*(12*k+5)*(12*k+7)*(12*k+11)/(-10368*(k+1)*(2*k+1)*x**2)
		k = k + 1

	return s

# Function Q of series
def Q(x, er=EPS):
	s = mp.mpf(0)
	term = mp.mpf(5/(72*x))
	k = mp.mpf(0)
	new_term = mp.mpf(-85085/(2239488*x**3))
	while mp.fabs(term) > mp.mpf(er) and mp.fabs(new_term) < mp.fabs(term):
		s = s + term
		term = new_term
		new_term = new_term * (12*k+7)*(12*k+11)*(12*k+13)*(12*k+17)/(-10368*(k+1)*(2*k+3)*x**2)
		k = k + 1

	return s

# Airy Ai function for large positive argument
def Ai_xpinf(x):
	ksi = 2*mp.fabs(x)**(3/2)/3
	return mp.exp(-ksi)/(2*mp.pi**(1/2)*x**(1/4))*L(-ksi)

# Airy Bi function for large positive argument
def Bi_xpinf(x):
	ksi = 2*mp.fabs(x)**(3/2)/3
	return mp.exp(ksi)/(mp.pi**(1/2)*x**(1/4))*L(ksi)

# Airy Ai function for large negative argument
def Ai_xminf(x):
	ksi = 2*mp.fabs(x)**(3/2)/3
	return (Q(ksi)*mp.sin(ksi-mp.pi/4)+P(ksi)*mp.cos(ksi-mp.pi/4))/(mp.pi**(1/2)*(-x)**(1/4))

# Airy Bi function for large negative argument
def Bi_xminf(x):
	ksi = 2*mp.fabs(x)**(3/2)/3
	return (-P(ksi)*mp.sin(ksi-mp.pi/4)+Q(ksi)*mp.cos(ksi-mp.pi/4))/(mp.pi**(1/2)*(-x)**(1/4))


# Matpoltlib setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Names for the different cases
rng = ['A+', 'A-', 'B+', 'B-']
names = [r'$\mathrm{Ai}$ for positive argument, ',
		 r'$\mathrm{Ai}$ for negative argument, ',
		 r'$\mathrm{Bi}$ for positive argument, ',
		 r'$\mathrm{Bi}$ for negative argument, ']

# Going over the cases
for i in range(len(rng)):
	# Setting up the range of arguments
	if '+' in rng[i]:
		x_lst = mp.linspace(0.1, 20, 200)
	else:
		x_lst = mp.linspace(-20, 0.1, 200)

	# Calculating functions, errors and time
	f_x0_lst = []
	f_xinf_lst = []
	f_x0_lst_time = []
	f_xinf_lst_time = []
	f_x0_abs = []
	f_xinf_abs = []
	f_x0_rel = []
	f_xinf_rel = []
	for x in x_lst:
		# with Taylor series
		if 'A' in rng[i]:
			start = timer()
			A1 = Ai_x0(x)
			end = timer()
		else:
			start = timer()
			A1 = Bi_x0(x)
			end = timer()
		f_x0_lst.append(A1)
		f_x0_lst_time.append(end-start)

		# with asymptotic series
		if rng[i] == 'A+':
			start = timer()
			A2 = Ai_xpinf(x)
			end = timer()
		elif rng[i] == 'A-':
			start = timer()
			A2 = Ai_xminf(x)
			end = timer()
		elif rng[i] == 'B+':
			start = timer()
			A2 = Bi_xpinf(x)
			end = timer()
		else:
			start = timer()
			A2 = Bi_xminf(x)
			end = timer()	
		f_xinf_lst.append(A2)
		f_xinf_lst_time.append(end-start)

		# calculating absolute Error
		if 'A' in rng[i]:
			f_x0_abs.append(mp.fabs(A1-mp.airyai(x)))
			f_xinf_abs.append(mp.fabs(A2-mp.airyai(x)))
		else:
			f_x0_abs.append(mp.fabs(A1-mp.airybi(x)))
			f_xinf_abs.append(mp.fabs(A2-mp.airybi(x)))

	
		# calculating relative error
		if 'A' in rng[i]:
			f_x0_rel.append(mp.fabs(A1/mp.airyai(x)-1))
			f_xinf_rel.append(mp.fabs(A2/mp.airyai(x)-1))
		else:
			f_x0_rel.append(mp.fabs(A1/mp.airybi(x)-1))
			f_xinf_rel.append(mp.fabs(A2/mp.airybi(x)-1))

	# Plotting absolute error vs x
	plt.title(names[i] + 'Absolute Error')
	plt.xlabel(r'$x$')
	plt.ylabel(r'Absolute Error $\varepsilon$')
	plt.yscale('log')
	plt.grid()

	plt.plot(x_lst, f_x0_abs, label='Taylor')
	plt.plot(x_lst, f_xinf_abs, label='Asymptotic')

	plt.legend()
	plt.savefig('images/' + rng[i] + '_abs.png')
	plt.show()

	# Plotting time vs x
	plt.title(names[i] + 'Evaluation Time')
	plt.xlabel(r'$x$')
	plt.ylabel(r'Evaluation Time $t$ [s]')
	plt.yscale('log')
	plt.grid()

	plt.plot(x_lst, f_x0_lst_time, label='Taylor')
	plt.plot(x_lst, f_xinf_lst_time, label='Asymptotic')

	plt.legend()
	plt.savefig('images/' + rng[i] + '_time.png')
	plt.show()

	# Plotting relative error vs x
	plt.title(names[i] + 'Relative Error')
	plt.xlabel(r'$x$')
	plt.ylabel(r'Relative Error $\rho$')
	plt.yscale('log')
	plt.grid()

	plt.plot(x_lst, f_x0_rel, label='Taylor')
	plt.plot(x_lst, f_xinf_rel, label='Asymptotic')

	plt.legend()
	plt.savefig('images/' + rng[i] + '_rel.png')
	plt.show()


# Finding zeros

# Function phi from text
def phi(x):
	return x**(2/3)*(1+5/(48*x**2) - 5/(36*x**4) + 77125/(82944*x**6) - 108056875/(6967296*x**8))

# Approximate zeros of Ai
def Ai_z(s):
	return -phi(3*mp.pi*(4*s-1)/8)

# Approximate zeros of Bi
def Bi_z(s):
	return	-phi(3*mp.pi*(4*s-3)/8)


# Actual zeros
Ai_zeros = []
Bi_zeros = []
Ai_zeros_approx = []
Bi_zeros_approx = []
for s in range(1, 101):
	A1 = Ai_z(s)
	B1 = Bi_z(s)
	A2 = mp.findroot(mp.airyai, Ai_z(s))
	B2 = mp.findroot(mp.airybi, Bi_z(s))

	Ai_zeros_approx.append(A1)
	Bi_zeros_approx.append(B1)
	Ai_zeros.append(A2)
	Bi_zeros.append(B2)

print(mp.nstr(Ai_zeros))
print(mp.nstr(Bi_zeros))

# Plotting zeros and error
s_lst = mp.linspace(1, 100, 100)

# Plotting absolute error
plt.title(r'Airy functions' zeros, Absolute Error')
plt.xlabel(r'Zeros $\{z_s\}_{s=0}^{100}$')
plt.ylabel(r'Absolute Error $\varepsilon$')
plt.yscale('log')
plt.grid()

plt.plot(Ai_zeros, [mp.fabs(Ai_zeros[i] - Ai_zeros_approx[i]) for i in range(len(s_lst))], 'bo', label=r'$\mathrm{Ai}$ function')
plt.plot(Bi_zeros, [mp.fabs(Bi_zeros[i] - Bi_zeros_approx[i]) for i in range(len(s_lst))], 'ro', label=r'$\mathrm{Bi}$ function')

plt.legend()
plt.savefig('images/' + 'zeros_abs.png')
plt.show()

# Plotting relative error
plt.title(r'Airy functions' zeros, Relative Error')
plt.xlabel(r'Zeros $-\{z_s\}_{s=0}^{100}$')
plt.ylabel(r'Relative Error $\rho$')
plt.yscale('log')
plt.grid()

plt.plot(Ai_zeros, [mp.fabs(Ai_zeros[i]/Ai_zeros_approx[i])-1 for i in range(len(s_lst))], 'bo', label=r'$\mathrm{Ai}$ function')
plt.plot(Bi_zeros, [mp.fabs(Bi_zeros[i]/Bi_zeros_approx[i])-1 for i in range(len(s_lst))], 'ro', label=r'$\mathrm{Bi}$ function')

plt.legend()
plt.savefig('images/' + 'zeros_rel.png')
plt.show()
