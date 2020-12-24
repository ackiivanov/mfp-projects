import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from timeit import default_timer as timer
import scipy.stats as stats
import scipy.sparse as sparse


# Plotting setup
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# create sparse and symmetric matrix
def sprandsym(n, density=0.01):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X) 
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    result = result.toarray()
    return result


# Jacobi iteration for symmetric matrix
def jacobi(A, eps=10**(-6)):

	# offset from diagonal
	def offset_sym(A):
		n = len(A)
		return 2*sum([A[i,j]**2 for i in range(n) for j in range (i+1, n)])

	# define skewed sign
	def sgn(x):
		if np.abs(x) > eps:
			return np.sign(x)
		else:
			return 1

	# copy matrix
	M = np.copy(A)

	# size of square matrix
	n = len(A)

	# define overall transition matrix
	Z = np.identity(n)

	# sweep until desired eps reached
	while offset_sym(M) > eps:

		# One sweep of the matrix
		for p in range(n):
			for q in range(p+1, n):

				# skip iteration if the element is alredy zero 
				if np.abs(M[p,q]) < eps:
					continue

				# find tan, cos and sin of angle of rotation
				a = (M[q,q]-M[p,p])/(2*M[p,q])
				t = sgn(a)/(np.abs(a) + np.sqrt(a**2+1))
				c = 1/np.sqrt(1+t**2)
				s = t * c

				# define transition matrix P
				P = np.identity(n)
				P[p,p] = c
				P[p,q] = s
				P[q,p] = -s
				P[q,q] = c

				# update Z
				Z = Z@P

				# update M
				M = P.T@M@P

	return M, Z


# Householder rotations for QR decomposition, square matrix
def householder(A):
	
	# copy matrix A
	M = np.copy(A)

	# size of square matrix
	n = len(A)

	# orthogonal matrix Q
	Q = np.eye(n)

	# loop over columns
	for p in range(n):
		# unit vector
		e = np.zeros(n-p)
		e[0] = 1

		# normal to hyperplane
		w = M[p:,p] + np.sign(M[p,p])*np.linalg.norm(M[p:,p])*e
		w.shape = (n-p,1)
		
		# transition matrix P
		P = np.block([[np.eye(p), np.zeros((p,n-p))],
			[np.zeros((n-p, p)), np.eye(n-p)-2/(w.T@w)*w@w.T]])
		
		# update Q
		Q = Q@P

		# update A
		M = P@M

	return Q, M


# Power iteration method for hermitian matrix
def iteration(A, x_0=None, eps=10**(-6)):
	
	# define hermitian conjugate
	def H(A):
		return np.conjugate(A.T)

	# define Rayleigh quotient for stopping condition
	def rayl(y, x):
		# y = A*x
		return complex(H(x)@y/(H(x)@x))

	# define dot product that returns scalar
	def dot(x, y):
		return complex(H(x)@y)

	# define norm
	def norm(x):
		return np.sqrt(dot(x, x))

	# size of square matrix
	n = len(A)

	# if x_0 isn't given generate a random one
	if x_0 is None:
		x_0 = np.random.random(size=(n,1))

	# eigenvalue and eigenvector lists
	lmd_lst = []
	vec_lst = []

	# loop for each eigenvalue
	while len(lmd_lst) < n:
		# normalize initial guess
		x = x_0/norm(x_0)

		# calculate reduced matrix on vector
		y = (A@x - sum([lmd_lst[i]*dot(vec_lst[i], x)*vec_lst[i]
			 for i in range(len(lmd_lst))]))

		# loop until desired eps reached
		while norm(y - rayl(y, x)*x) > eps:
			y_norm = norm(y)
			x = y/y_norm
			y = (A@x - sum([lmd_lst[i]*dot(vec_lst[i], x)*vec_lst[i]
				 for i in range(len(lmd_lst))]))

		# add eigenvalue and normalized eigenvector to list
		lmd_lst.append(rayl(y, x))
		vec_lst.append(x)

	return lmd_lst, vec_lst


# makes the unperturbed Hamiltonian
def make_H_0(n):
	energies = [i + 1/2 for i in range(n)]
	return np.diag(energies)


# makes the q matrix
def make_q(n):
	# starting matrix
	q = np.zeros((n, n))
	
	# loop through the matrix
	for j in range(n):

		if j >= 1:
			q[j-1][j] = np.sqrt(j/2)
		if j <= n - 2:
			q[j+1][j] = np.sqrt((j + 1)/2)
		
	return q


# makes the q^2 matrix
def make_q2(n):
	# starting matrix
	q2 = np.zeros((n, n))

	# loop through the matrix
	for j in range(n):

		q2[j][j] = j + 1/2

		if j >= 2:
			q2[j-2][j] = np.sqrt(j*(j - 1))/2
		if j <= n - 3:
			q2[j+2][j] = np.sqrt((j + 1)*(j + 2))/2

	return q2


# makes the q^4 matrix
def make_q4(n):
	# starting matrix
	q4 = np.zeros((n, n))

	# loop through the matrix
	for j in range(n):

		q4[j][j] = 3/4*(2*j**2 + 2*j + 1)

		if j >= 2:
			q4[j-2][j] = np.sqrt(j*(j - 1))*(j - 1/2)
		if j >= 4:
			q4[j-4][j] = np.sqrt(j*(j - 1)*(j - 2)*(j - 3))/4
		if j <= n - 5:
			q4[j+4][j] = np.sqrt((j + 1)*(j + 2)*(j + 3)*(j + 4))/4
		if j <= n - 3:
			q4[j+2][j] = np.sqrt((j + 1)*(j + 2))*(2*j + 3)/2

	return q4


# Testing speed of implementation vs. built-in for a random matrix

# list of different N
N_lst = range(1, 60)

# times
jaco_times = []
hous_times = []
iter_times = []
eigh_times = []

# loop over different N
for N in N_lst:

	# indicator
	print('Calculating for N={}'.format(N))

	# generate matrix
	A = np.random.random(size=(N,N))
	A = (A + A.T)/2

	# test for Jacobi
	start = timer()
	D1, V1 = jacobi(A)
	end = timer()
	jaco_times.append(end - start)

	# test for Householder
	start = timer()
	V1, D1 = householder(A)
	end = timer()
	hous_times.append(end - start)

	# test for standard iteration
	start = timer()
	E1, V1 = iteration(A)
	end = timer()
	iter_times.append(end - start)

	# test for built-in function eigh
	start = timer()
	E1, V1 = np.linalg.eigh(A)
	end = timer()
	eigh_times.append(end - start)	


# plotting
plt.title(r'Evaluation Time as a Function of Linear Size')
plt.xlabel(r'Linear Size of Matrix $N$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.yscale('log')
plt.grid()

plt.plot(N_lst, jaco_times, '-o', label=r'Jacobi')
plt.plot(N_lst, hous_times, '-o', label=r'Householder')
plt.plot(N_lst, iter_times, '-o', label=r'Iteration')
plt.plot(N_lst, eigh_times, '-o', label=r'\texttt{np.linalg.eigh}')

plt.legend(loc='best')
plt.savefig('images/' + 'eval_times.png')
plt.show()



# Testing speed of implementation vs. built-in for a sparse matrix

# list of different N
N_lst = range(1, 100)

# times
jaco_times = []
hous_times = []
eigh_times = []

# loop over different N
for N in N_lst:

	# indicator
	print('Calculating for N={}'.format(N))

	# generate matrix
	A = sprandsym(N)

	# test for Jacobi
	start = timer()
	D1, V1 = jacobi(A)
	end = timer()
	jaco_times.append(end - start)
	
	# test for Householder
	start = timer()
	Q1, R1 = householder(A)
	end = timer()
	hous_times.append(end - start)

	# test for built-in function eigh
	start = timer()
	E1, V1 = np.linalg.eigh(A)
	end = timer()
	eigh_times.append(end - start)	

# plotting
plt.title(r'Evaluation Time as a Function of Linear Size, Sparse Matrix')
plt.xlabel(r'Linear Size of Matrix $N$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.yscale('log')
plt.grid()

plt.plot(N_lst, jaco_times, '-o', label=r'Jacobi')
plt.plot(N_lst, hous_times, '-o', label=r'Householder')
plt.plot(N_lst, eigh_times, '-o', label=r'\texttt{np.linalg.eigh}')

plt.legend(loc='best')
plt.savefig('images/' + 'eval_times_sparse.png')
plt.show()



# Testing precision of implementation vs. built-in for a random matrix

# list of different N
N_lst = range(1, 60)

# precision lists
jaco_prec = []
hous_prec = []
eigh_prec = []

# loop over different N
for N in N_lst:

	# indicator
	print('Calculating for N={}'.format(N))

	# generate matrix
	A = np.random.random(size=(N,N))
	A = (A + A.T)/2
	
	# test for Jacobi
	D1, V1 = jacobi(A, eps=10**(-15))
	jaco_prec.append(np.abs(np.amax(V1@D1@V1.T-A)))
	
	# test for Householder
	Q1, R1 = householder(A)
	hous_prec.append(np.abs(np.amax(Q1@R1-A)))
	
	# test for built-in function eigh
	E1, V1 = np.linalg.eigh(A)
	eigh_prec.append(np.abs(np.amax(V1@np.diag(E1)@V1.T - A)))

# plotting
plt.title(r'Error Estimate as a Function of Linear Size')
plt.xlabel(r'Linear Size of Matrix $N$')
plt.ylabel(r'Error Estimate $\varepsilon$')
plt.yscale('log')
plt.grid()

plt.plot(N_lst, jaco_prec, '-o', label=r'Jacobi')
plt.plot(N_lst, hous_prec, '-o', label=r'Householder')
plt.plot(N_lst, eigh_prec, '-o', label=r'\texttt{np.linalg.eigh}')

plt.legend(loc='best')
plt.savefig('images/' + 'eval_prec.png')
plt.show()



# Testing precision of implementation vs. built-in for a sparse matrix

# list of different N
N_lst = range(1, 60)

# times
jaco_prec = []
hous_prec = []
eigh_prec = []

# loop over different N
for N in N_lst:

	# indicator
	print('Calculating for N={}'.format(N))

	# generate matrix
	A = np.random.random(size=(N,N))
	A = (A + A.T)/2
	
	# test for Jacobi
	D1, V1 = jacobi(A, eps=10**(-15))
	jaco_prec.append(np.abs(np.amax(V1@D1@V1.T-A)))
	
	# test for Householder
	Q1, R1 = householder(A)
	hous_prec.append(np.abs(np.amax(Q1@R1-A)))
	
	# test for built-in function eigh
	E1, V1 = np.linalg.eigh(A)
	eigh_prec.append(np.abs(np.amax(V1@np.diag(E1)@V1.T - A)))

# plotting
plt.title(r'Error Estimate as a Function of Linear Size, Sparse Matrix')
plt.xlabel(r'Linear Size of Matrix $N$')
plt.ylabel(r'Error Estimate $\varepsilon$')
plt.yscale('log')
plt.grid()

plt.plot(N_lst, jaco_prec, '-o', label=r'Jacobi')
plt.plot(N_lst, hous_prec, '-o', label=r'Householder')
plt.plot(N_lst, eigh_prec, '-o', label=r'\texttt{np.linalg.eigh}')

plt.legend(loc='best')
plt.savefig('images/' + 'eval_prec_sparse.png')
plt.show()



# Eigenpairs as a function of lambda

# Size of matrix
N = 100

# List of lambdas
lmd_lst = np.linspace(0, 1.01, 2000)

# create the matrices
H_0 = make_H_0(N)
q4 = make_q4(N)

# eigenvalue and eigenvector list
eig_lst = []
vec_lst = []

# calculate the eigenpairs
for lmd in lmd_lst:
	E, V = np.linalg.eigh(H_0+lmd*q4)

	eig_lst.append(E)
	vec_lst.append(V)

# make them into arrays for easier handling
eig_lst = np.array(eig_lst)
vec_lst = np.array(vec_lst)

# plotting eigenvalues

plt.title(r'Eigenvalues as a Function of Perturbation Parameter')
plt.xlabel(r'Perturbation Parameter $\lambda$')
plt.ylabel(r'Eigenvalues $E_n (\lambda) - n - \frac{1}{2}$')
plt.grid()

# plot the first 10
for n in range(10):
	plt.plot(lmd_lst, eig_lst[:,n]-n-1/2, '-', label=r'$n = {}$'.format(n))

plt.legend(loc='best')
plt.savefig('images/' + 'eig_lambda.png')
plt.show()

# plotting eigenvectors

# eigenvector to plot
n = 0

plt.title(r'Eigenvector Components as a Function of Perturbation Parameter')
plt.xlabel(r'Perturbation Parameter $\lambda$')
plt.ylabel(r'Eigenvector $| {} \rangle$ Components'.format(n))
plt.grid()

# plot some components of n-th eigenvector
lines = ['-.', '-', '--']
for i in range(max(n-6, 0), min(n+10, N)):
	plt.plot(lmd_lst, np.abs(vec_lst[:,i,n]), lines[int(np.sign(i - n))+1],
			 label=r'$\langle {}^{{(0)}} | {} \rangle$'.format(i, n))

plt.legend(loc='upper right')
plt.savefig('images/' + 'vec{}_lambda.png'.format(n))
plt.show()



# Eigenpairs as a function of N

# choose lambda and list of different N
lmd = 0.1
N_min = 10
N_max = 41
N_lst = range(N_min, N_max)

# creat list of eigenvalues
eig_lst = []
vec_lst = []

# loop over list of N
for N in N_lst:
	# create NxN matrices
	H_0 = make_H_0(N)
	q4 = make_q4(N)

	# calculate eigenpairs
	E, V = np.linalg.eigh(H_0+lmd*q4)

	# save eigenpairs upto N_min
	eig_lst.append(E[:N_min])
	vec_lst.append(V[:N_min,:N_min])

eig_lst = np.array(eig_lst)
vec_lst = np.array(vec_lst)

# plotting eigenvalues

plt.title(r'Eigenvalues as a Function of Linear Matrix Size')
plt.xlabel(r'Linear Matrix Size $N$')
plt.ylabel(r'Eigenvalues $E_n$')
plt.grid()

# plot the first 10
for n in range(N_min):
	plt.plot(N_lst, eig_lst[:,n], '-', label=r'$n = {}$'.format(n))

plt.legend(loc='best')
plt.savefig('images/' + 'eig_N.png')
plt.show()

# plotting eigenvectors

plt.title(r'Eigenvector Component as a Function of Linear Matrix Size')
plt.xlabel(r'Linear Matrix Size $N$')
plt.ylabel(r'Eigenvector Component $\langle n^{{(0)}}| n \rangle$ Components')
plt.grid()

# plot parallel components of first 10 vectors
for n in range(N_min-1, -1, -1):
	plt.plot(N_lst, np.abs(vec_lst[:,n,n]), '-', label=r'$n = {}$'.format(n))

plt.legend(loc='upper right')
plt.savefig('images/' + 'vec_N.png'.format(n))
plt.show()



# Testing speed q vs. q^2 vs. q^4

# list of different N
N_lst = range(1, 1000)

# times
q_times = []
q2_times = []
q4_times = []

# loop over different N
for N in N_lst:

	# indicator
	print('Calculating for N={}'.format(N))

	# test for q
	start = timer()
	q = make_q(N)
	A = q@q@q@q
	end = timer()
	q_times.append(end - start)

	# test for q^2
	start = timer()
	q2 = make_q2(N)
	B = q2@q2
	end = timer()
	q2_times.append(end - start)

	# test for q^4
	start = timer()
	q4 = make_q4(N)
	end = timer()
	q4_times.append(end - start)

# plotting
plt.title(r'Evaluation Time as a Function of Linear Matrix Size')
plt.xlabel(r'Linear Size of Matrix $N$')
plt.ylabel(r'Evaluation Time $t$ [s]')
plt.yscale('log')
plt.grid()

plt.plot(N_lst, q_times,'-o', markersize=3, label=r'Make $q$')
plt.plot(N_lst, q2_times, '-o', markersize=3, label=r'Make $q^2$')
plt.plot(N_lst, q4_times, '-o', markersize=3, label=r'Make $q^4$')

plt.legend(loc='best')
plt.savefig('images/' + 'q_times.png')
plt.show()



# Testing precision q vs. q^4 and q^2 vs. q^4

# list of different N
N_lst = range(3, 1000)

# Error estimate lists
q_max_err = []
q2_max_err = []
q_avg_err = []
q2_avg_err = []

# loop over different N
for N in N_lst:

	# indicator
	print('Calculating for N={}'.format(N))

	# make matrices
	q = make_q(N)
	q2 = make_q2(N)
	q4 = make_q4(N)

	#calculate difference
	A = np.abs(q4 - q@q@q@q)
	B = np.abs(q4 - q2@q2)

	# save error
	q_max_err.append(np.max(A))
	q2_max_err.append(np.max(B))
	q_avg_err.append(np.average(A[:N-2,:N-2]))
	q2_avg_err.append(np.average(B[:N-1,N-1]))

# plotting maximal error
plt.title(r'Maximal Element Error vs. Linear Matrix Size')
plt.xlabel(r'Linear Size of Matrix $N$')
plt.ylabel(r'Error Estimate $\mathrm{max}(\varepsilon)$')
plt.yscale('log')
plt.grid()

plt.plot(N_lst, q_max_err,'-o', markersize=3, label=r'$\hat{q^4} - \hat{q}^4$')
plt.plot(N_lst, q2_max_err, '-o', markersize=3, label=r'$\hat{q^4} - \hat{q^2}^2$')

plt.legend(loc='best')
plt.savefig('images/' + 'q_max_err.png')
plt.show()

# plotting uncorrupted average error
plt.title(r'Average Error Excluding Last Rows and Columns vs. Size')
plt.xlabel(r'Linear Size of Matrix $N$')
plt.ylabel(r'Error Estimate $\mathrm{avg}_{i=0}^{N-3}(\varepsilon)$')
plt.yscale('log')
plt.grid()

plt.plot(N_lst, q_avg_err,'o', markersize=3, label=r'$\hat{q^4} - \hat{q}^4$')
plt.plot(N_lst, q2_avg_err, 'o', markersize=3, label=r'$\hat{q^4} - \hat{q^2}^2$')

plt.legend(loc='best')
plt.savefig('images/' + 'q_avg_err.png')
plt.show()



# Extra problem

# define potential
def V_pot(x):
	return -2*(x**2) + (x**4)/10

# define state from coeficients
def state(coefs, x):
	phi = np.polynomial.hermite.Hermite(
		[1/np.sqrt(2**n*np.math.factorial(n)*np.sqrt(np.pi))
		*np.exp(-x**2/2)*coefs[n] for n in range(len(coefs))])

	return phi.__call__(x)


# set matrix size
N = 100

# make matrices
H_0 = make_H_0(N)
q2 = make_q2(N)
q4 = make_q4(N)
H = H_0 - 5/2*q2 + 1/10*q4

# find the eigenpairs
E, V = np.linalg.eigh(H)

# plot the eigenvalues
plt.title(r'Eigenvalues and Potential')
plt.xlabel(r'Position $q$')
plt.ylabel(r'Potential $V(q)$')
plt.grid()

x_dummy = np.linspace(-5, 5, 1000)

# plot the potential
plt.plot(x_dummy, [V_pot(x) for x in x_dummy], 'k-')

# plot the first 10 eigenvalues
for n in range(18):
	plt.axhline(y=E[n], label=r'$E_{{{}}} = {:.3f}$'.format(n, E[n]))


plt.legend(loc='upper right', prop={'size': 8})
plt.savefig('images/' + 'extra_eig.png')
plt.show()

# plot the eigenvectors
plt.title(r'Eigenstates and Potential')
plt.xlabel(r'Position $q$')
plt.ylabel(r'Potential $V(q)$ / State $\Psi(q)$')
plt.yticks([])
plt.grid()

x_dummy = np.linspace(-5, 5, 1000)

# plot the potential
plt.plot(x_dummy, [V_pot(x) for x in x_dummy]
		 -min([V_pot(x) for x in x_dummy]), color='grey')

# plot the first 3 states
plt.axhline(y=0, linewidth=1.5, color='k')
plt.plot(x_dummy, [-5*state(V[:,0], x) for x in x_dummy], 'r-', label=r'$n = 0$')
plt.axhline(y=7, linewidth=1.5, color='k')
plt.plot(x_dummy, [5*state(V[:,1], x)+7 for x in x_dummy], 'b-', label=r'$n = 1$')
plt.axhline(y=14, linewidth=1.5, color='k')
plt.plot(x_dummy, [5*state(V[:,2], x)+14 for x in x_dummy], 'g-', label=r'$n = 2$')

plt.legend(loc='upper right')
plt.savefig('images/' + 'extra_vec.png')
plt.show()
