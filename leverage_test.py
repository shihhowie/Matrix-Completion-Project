import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
import cvxpy as cp
from HelperFunctions import *


pct_obs =0.8
m,n,k = 100,100,5
alpha = 0

pcts = np.arange(0.1, 1, 0.05)
alphas = np.arange(0, 1.1, 0.25)

fig, ax = plt.subplots(2,1)

r_lev = []
l_lev = []
for alpha in alphas:
	M = construct_matrix(m,n,k,alpha)
	M = M/np.linalg.norm(M,"fro")
	u,s,v = np.linalg.svd(M)
	r_lev_full, c_lev_full = leverage(M)
	r_lev_approx, c_lev_approx =approx_leverage(M, 6)
	plt.subplot(3,2,1)
	plt.plot(np.arange(len(r_lev_full)), np.sort(r_lev_full), label=str(alpha), alpha=0.7)
	plt.subplot(3,2,2)
	plt.plot(np.arange(len(c_lev_full)), np.sort(c_lev_full), label=str(alpha), alpha=0.7)
	plt.subplot(3,2,3)
	plt.plot(np.arange(len(r_lev_approx)), r_lev_approx, label=str(alpha), alpha=0.7)
	plt.subplot(3,2,4)
	plt.plot(np.arange(len(c_lev_approx)), c_lev_approx, label=str(alpha), alpha=0.7)

	rcoh = np.linalg.norm(u@np.diag(s), axis=1)
	ccoh = np.linalg.norm(np.diag(s)@v, axis=0)
	plt.subplot(3,2,5)
	plt.plot(np.arange(len(rcoh)), rcoh, label=str(alpha), alpha=0.7)
	plt.subplot(3,2,6)
	plt.plot(np.arange(len(ccoh)), ccoh, label=str(alpha), alpha=0.7)


plt.legend()
plt.show()
# for alpha in alphas:
# 	r_errs = []
# 	l_errs = []
# 	for pct in pcts:
# 		r_err = []
# 		l_err = []
# 		for i in range(20):
# 			M = construct_matrix(m,n,k,alpha)
# 			M_obs = construct_partial_matrix(M, pct)
# 			r_lev_full, l_lev_full = leverage(M)
# 			r_lev_partial, l_lev_partial = leverage(np.nan_to_num(M_obs))
# 			r_err.append(np.linalg.norm(r_lev_full-r_lev_partial))
# 			l_err.append(np.linalg.norm(l_lev_full-l_lev_partial))
# 		r_errs.append(np.mean(r_err))
# 		l_errs.append(np.mean(l_err))
# 	plt.subplot(2,1,1)
# 	plt.plot(pcts, r_errs, label = "r,"+str(np.round(alpha,2))+","+str(pct))
# 	plt.subplot(2,1,2)
# 	plt.plot(pcts, l_errs, label = "l,"+str(np.round(alpha,2))+","+str(pct))

# plt.show()
