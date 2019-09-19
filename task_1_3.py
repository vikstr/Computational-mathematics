import sys
import numpy as np
import scipy
import scipy.linalg
n=4
A = []
for i in range(n):
    row = input().split()
    for j in range(n):
        row[j]=int(row[j])
    A.append(row)
np.array(A)
b = list()
for k in range(n):
    x = input()
    b.append(int(x))
np.array(b)
print(A)
print(b)
P, L, U = scipy.linalg.lu(A)
X_usual_sol = np.linalg.solve(A,b)
np.array(L)
np.array(U)
np.array(P)
Y = np.linalg.solve(L,b)
print(Y)
X = np.linalg.solve(U,Y)
l_1 = np.linalg.norm(X-X_usual_sol,1)
print(X)
print(P)
print(L)
print(U)
print(X_usual_sol)
print(l_1)
