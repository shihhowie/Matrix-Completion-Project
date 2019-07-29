import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from HelperFunctions import *
import cvxpy as cp
import multiprocessing as mp


def coherent_mc(M, k, lambda_, max_iter=20, method="uniform", alpha=1):
	obs_entries = ~np.isnan(M)
	m,n = M.shape
	R = np.random.normal(0,1,(k,n))
	errs = []
	lam = cp.Parameter(nonneg=True)
	p_c = cp.Parameter(m, nonneg=True)
	p_r = cp.Parameter(n, nonneg=True)
	lam.value = lambda_
	if method == "uniform":
		p_c.value = np.ones(m)/m
		p_r.value = np.ones(n)/n
	elif method == "empirical":
		p_c.value = np.sum(obs_entries, axis=1)/np.sum(obs_entries)
		p_r.value = np.sum(obs_entries, axis=0)/np.sum(obs_entries)
	pbar = ProgressBar()
	for i in pbar(range(max_iter)):
		if i%2 == 0:
			L = cp.Variable((m,k))
		else:
			R = cp.Variable((k,n))
		M_est = L*R
		error = cp.multiply(obs_entries, np.nan_to_num(M))-cp.multiply(obs_entries, M_est)
		if i%2 == 0:
			objective = cp.Minimize(cp.square(cp.norm(error, "fro")) + lam*cp.norm(p_c*cp.square(L)))
		else:
			objective = cp.Minimize(cp.square(cp.norm(error, "fro")) + lam*cp.norm(cp.square(R)*p_r))
		try:
			prob = cp.Problem(objective)
			prob.solve(solver=cp.SCS, use_indirect=False)
			if i % 2 == 0:
				L = L.value
			else:
				R = R.value
			err = np.multiply(obs_entries, np.nan_to_num(M))-np.multiply(obs_entries, L@R)
			errs.append(np.linalg.norm(err))
		except:
			if i % 2 == 0:
				L = np.random.normal(0,1,(m,k))
			else:
				R = np.random.normal(0,1,(k,n))
			err = np.nan
			errs.append(err)
		
	return(L,R, errs)

#Experiment on sampling uniformly vs. 

pct_obs = np.arange(0.6,1,0.1)
alphas = np.arange(0,1.1,0.25)
rep = 10
m,n,k = 40,10,7
lambda_ = np.sqrt(1/max(m,n))
max_iter = 20
rep = 10 

uni_error = []
emp_error = []
for i in range(rep):
	M = construct_matrix(m,n,k,alphas[4])
	M_obs = construct_partial_matrix(M,pct_obs[2],method="uniform")
	M_est_uni = coherent_mc(M_obs, 10, lambda_, max_iter, "uniform")
	M_est_emp = coherent_mc(M_obs, 10, lambda_, max_iter, "empirical")
	# relative_error_uni = completion_error(M,M_est_uni[0]@M_est_uni[1])
	# relative_error_emp = completion_error(M,M_est_emp[0]@M_est_emp[1])
	uni_error.append(M_est_uni[2])
	emp_error.append(M_est_emp[2])

plt.plot(np.array(uni_error).T)
plt.show()

plt.plot(np.array(emp_error).T)
plt.show()

def relative_error(alpha, pct, method):
	M = construct_matrix(m,n,k,alpha)
	M_obs = construct_partial_matrix(M,pct,method=method)
	M_est = coherent_mc(M_obs, 10, lambda_, max_iter)
	relative_error = completion_error(M,M_est[0]@M_est[1])
	return(relative_error)

def mc_ex1(method):
	completions = []
	pbar = ProgressBar()
	for j in pbar(range(len(alphas))):
		alpha = alphas[j]
		completion = []
		for i in range(len(pct_obs)):
			pct = pct_obs[i]
			trial = []
			for j in range(rep):
				trial.append(relative_error(alpha, pct, method))
			completion.append(trial)
		completions.append(completion)
	
	return(completions)

# incoherent_completions = mc_ex1("incoherent")

def plot_completion_rate(completions):
	pct_completion = np.mean(np.array(completions)<0.1,axis=2)
	fig, ax = plt.subplots(len(alphas),1)
	for i in range(len(alphas)):
		plt.subplot(len(alphas),1,i+1)
		# plt.plot(pct_obs, pct_completion_incoherent[i], label=str(alphas[i])+" incoherent")
		plt.plot(pct_obs, pct_completion[i], label=str(alphas[i]*100)+"pct coherent")
		plt.legend()

	plt.show()

plot_completion_rate(incoherent_completions)

