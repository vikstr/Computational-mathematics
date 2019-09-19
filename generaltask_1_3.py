import sys
import numpy as np
import scipy
import scipy.linalg
from scipy.linalg import lu, inv
n=4
A = []
for i in range(n):
    row = input().split()
    for j in range(n):
        row[j]=int(row[j])
    A.append(row)
b = list()
for k in range(n):
    x = input()
    b.append(int(x))
U = np.zeros((n,n), dtype = float)
L = np.zeros((n,n),dtype = float)
np.array(A)
np.array(b)
def mult_matrix(M,N):
    MN = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            for k in range(0,n):
                MN[i][j]+=M[i][k]*N[k][j]
    return MN
def pivot(M):
    id_mat = [[float(i ==j) for i in range(n)] for j in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(M[i][j]))
        if j is not row:
            id_mat[j], id_mat[row] = id_mat[row], id_mat[j]
    return id_mat
P = pivot(A)
np.array(P)
PA = mult_matrix(P,A)
np.array(PA)
for i in range(n):
    U[0][i] = A[0][i]
    L[i][0] = A[i][0]/U[0][0]
for j in range(n):
    L[j][j] = 1.0
    for i in range(j+1):
        U[i][j] = PA[i][j]
        for k in range(i):
            U[i][j] -= (L[i][k]*U[k][j])
    for i in range(j,n):
        L[i][j] = PA[i][j]/U[j][j]
        for k in range(j):
            L[i][j] -= (U[k][j]*L[i][k]/U[j][j])
X_usual_sol = np.linalg.solve(A,b)
pl, u = lu(A, permute_l=True)
Y = np.zeros(n)
for m, q in enumerate(b):
    Y[m] = q
        # skip for loop if m == 0
    if m:
        for i in range(m):
            Y[m] -= Y[i] * pl[m,i]
    Y[m] /= pl[m, m]

    # backward substitution to solve for y = Ux
X = np.zeros(n)
for midx in range(n):
    m = n - 1 - midx  # backwards index
    X[m] = Y[m]
    if midx:
        for nidx in range(midx):
            i = n - 1 - nidx
            X[m] -= X[i] * u[m,i]
    X[m] /= u[m, m]
l_1 = np.linalg.norm(X-X_usual_sol,1)
print(L)
print(U)
print(u)
print(X)
print('The norm of difference between solutions is:', l_1)
