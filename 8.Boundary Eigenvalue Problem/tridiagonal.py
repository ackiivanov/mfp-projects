#-----------------------------------------------------------------------------

"""Routines to solve a tridiagonal matrix equation Ax=b where A is
tridiagonal with main diagonal d, subdiagonal a, and superdiagonal c.

USAGE:
    factor(a, d, c)        # a and d are modified (LU factored)
    x = solve(a, d, c, b)

AUTHOR:
    Jonathan Senning <jonathan.senning@gordon.edu>
    Gordon College
    September 2, 2008
"""

#-----------------------------------------------------------------------------

def factor( a, d, c ):
    """Performs LU factorization on tridiagonal matrix A

    USAGE:
        factor( a, d, c )

    INPUT:
        a, d, c    - lists or NumPy arrays specifying the diagonals of the
                     tridiagonal matrix A.  a is the subdiagonal with a[0]
                     being the A[1,0] value, d is the main diagonal with
                     d[0] being the A[0,0] value and c is the superdiagonal
                     with c[0] being the A[0,1] value.

    OUTPUT:
        a, d, c    - arrays containing the data for the factored matrix

    NOTE:
        For this to be sure to work A should be strictly diagonally
        dominant, meaning that |d(i)| > |a(i-1)| + |c(i)| for each i.
        This ensures that pivoting will not be necessary.
    """

    n = len( d )

    for i in range( 1, n ):
        a[i-1] = a[i-1] / d[i-1]
        d[i] = d[i] - a[i-1] * c[i-1]

    return

#-----------------------------------------------------------------------------

def solve( a, d, c, b ):
    """Solves Ax=b for x with factored tridigonal A having diagonals a, d, c

    USAGE:
        x = solve( a, d, c, b )

    INPUT:
        a, d, c    - lists or NumPy arrays specifying the diagonals of the
                     factored tridiagonal matrix A.  These are produced by
                     factor().
        b          - right-hand-side vector

    OUTPUT:
        x          - float list: solution vector
    """

    n = len( d )

    # This is a bit confusing but it keeps things efficient and avoids
    # changing b, something that it desirable to avoid side effects.  The
    # prototypical way to write the code below would be
    #
    #     for i in xrange( 1, n ):
    #         b[i] = b[i] - a[i-1] * b[i-1]
    #     x[n-1] = b[n-1] / d[n-1]
    #     for i in xrange( n-2, -1, -1 ):
    #         x[i] = ( b[i] - c[i] * x[i+1] ) / d[i]
    #
    # but this changes b.  We use the fact that the portion of x computed
    # in the second loop corresponds to values in b that are no longer
    # needed so we can use x to hold the modified b values as long as they
    # are needed.

    x = [0] * n
    x[0] = b[0]

    for i in range( 1, n ):
        x[i] = b[i] - a[i-1] * x[i-1]

    x[n-1] = x[n-1] / d[n-1]

    for i in range( n-2, -1, -1 ):
        x[i] = ( x[i] - c[i] * x[i+1] ) / d[i]

    return x
