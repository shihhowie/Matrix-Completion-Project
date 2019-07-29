# Generating a matrix with corrupted rows/columns
import numpy as np
from numpy import linalg as LA
from random import *
from math import *

def LowRank_Sparse(m, n, sparse_depth, byRow=True):
    X = np.random.normal(0, 1, (m, n))
    Y = np.random.normal(0, 1, (m, n))
    L_0 = X@Y.T
    S_0 = np.zeros((m, m))
    x = sample(range(m), k=ceil(sparse_depth*m))
    if byRow==True:
        S_0[x, :] = np.random.normal(20, 5, m)
    else:
        S_0[:, x] = np.random.normal(20, 5, m)
    M = L_0+S_0
    return M, L_0, S_0

def LowRank_Sparse_Block(m, n, sparse_depth):
    X = np.random.normal(0, 1, (m, n))
    Y = np.random.normal(0, 1, (m, n))
    L_0 = X@Y.T
    S_0 = np.zeros((m, m))
    x = ceil(m/2)
    a = ceil(x+(sparse_depth*m))
    for i in range(x, a):
        for j in range(x, a):
            S_0[i, j] = np.random.normal(20, 5)
    M = L_0+S_0
    return M, L_0, S_0

def is_all_zero(C):
    n = C.shape[0]
    count = 0
    for i in range(n):
        for j in range(n):
            if C[i, j] != 0:
                count=count+1
            else:
                count=count
    return count


# Sparse depth is an int for the number of nonzero entries in S
# c > 0
# Setting used http://jmlr.org/papers/volume20/18-022/18-022.pdf on page 11
def l_s(m, n, sparse_depth, c):
    X = np.random.normal(0, 1, (m, n))
    Y = np.random.normal(0, 1, (m, n))
    L_0 = X@Y.T
    S_0 = np.zeros((m, m))
    rows = np.random.choice(m, sparse_depth)
    cols = np.random.choice(m, sparse_depth)
    for i in range(sparse_depth):
        S_0[rows[i], cols[i]] = np.random.uniform(-c*np.abs(L_0[rows[i], cols[i]]), c*np.abs(L_0[rows[i], cols[i]]))

    M = L_0 + S_0
    return M, L_0, S_0

    



