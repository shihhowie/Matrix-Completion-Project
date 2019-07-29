import pandas as pd
import numpy as np

def construct_matrix(m,n,k,alpha):
	U = np.random.normal(0,1,(m,k))
	V = np.random.normal(0,1,(n,k))
	Dl = np.diag(np.arange(1,m+1)**(-alpha))
	Dr = np.diag(np.arange(1,n+1)**(-alpha))
	M = Dl @ U @ V.T @ Dr
	return(M)

def completion_error(M, M_est):
	error = np.linalg.norm(M-M_est, "fro")
	relative_error = error/np.linalg.norm(M,"fro")
	return(relative_error)

def local_coherence(M):
	u,s,v = np.linalg.svd(M, full_matrices=False)
	rcoh = np.linalg.norm(u@np.diag(s), axis=1)
	ccoh = np.linalg.norm(np.diag(s)@v, axis=0)
	return(rcoh, ccoh)

def approx_leverage(M, k):
	m,n = M.shape
	u,s,v = np.linalg.svd(M, full_matrices=False)
	r_lev = np.linalg.norm(u[:,:k], axis=1)
	c_lev = np.linalg.norm(v[:k,:], axis=0)
	return(r_lev, c_lev)

def leverage(M):
	m,n = M.shape
	u,s,v = np.linalg.svd(M, full_matrices=False)
	r_lev = np.linalg.norm(u, axis=1)
	c_lev = np.linalg.norm(v, axis=0)
	return(r_lev, c_lev)

def construct_partial_matrix(M, pct, method="uniform", alpha=1, k=10):
	m,n = M.shape
	n_samples = int(m*n*pct)
	if(method=="uniform"):
		p = np.ones(n*m)/(n*m)
	elif(method=="leverage"):
		pr, pc = approx_leverage(M,k)
		p = np.outer(pr, pc).reshape(-1)
		p = p/np.sum(p)
	elif(method=="coherent"):
		pr, pc = local_coherence(M)
		p = np.outer(pr, pc).reshape(-1)
		p = p/np.sum(p)
	else:
		pr = np.random.dirichlet(np.ones(m)*alpha)
		pc = np.random.dirichlet(np.ones(n)*alpha)
		p = np.outer(pr, pc).reshape(-1)
		p = p/np.sum(p)
	M_obs = np.nan*np.ones(m*n)
	obs_indices = np.random.choice(np.arange(m*n), p=p, size=n_samples, replace=False)
	flattened_M = np.copy(M).reshape(-1)
	M_obs[obs_indices] = flattened_M[obs_indices]
	M_obs = M_obs.reshape(m,n)
	return(M_obs)

