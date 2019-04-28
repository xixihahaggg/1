import numpy as np
from scipy.linalg import circulant
def project2semidefinite(X):
    U,S,V = np.linalg.svd(X)
    S = np.maximum(S,0)
    M = np.linalg.multi_dot([U, np.diag(S), V])
    return M

def project2circulant(X):
    t = []

    n =  X.shape[0]
    for k in range(n):
        rho = 1/(2*(n-k))
        r = 0
        for i in range(0,n-k):
            r +=  X[i,i+k] + X[i+k,i]
        t.append(rho * r)
    A = circulant(t)
    return A

A = np.random.random([5,5])
H = project2semidefinite(A)
J = project2circulant(A)
def project2toeplitz(A):
    return A + project2circulant(project2semidefinite(A)) - project2semidefinite(A)
def f_nrom(A):
    X=np.square(A)
    Y = np.sqrt(X)
    return Y, np.sum(Y)
X = project2toeplitz(A)
print(X, A)
print(f_nrom(X-A))
