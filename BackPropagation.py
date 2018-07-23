import numpy as np
from scipy.linalg import block_diag
from activation import activation as act
from cost import cost as cost

def BackProp(X,y,theta,gamma):
	n,m = X.shape
	a = act(X,theta)
	delta = [np.asmatrix(a[-1] - y)]
	r = len(theta)
	X_ = np.concatenate((np.ones((n,1)), X), axis = 1 )
	theta_ = np.array([np.delete(theta[i],0,axis = 1) for i in np.arange(len(theta))])
	theta_reg = [np.copy(theta[i]) for i in np.arange(len(theta))]
	for i in np.arange(len(theta)):
		theta_reg[i][:,0]=0
	
	a = np.ndarray.tolist(a)
	a.insert(0,X_.T)
	a = np.array(a)
	for i in np.arange(r-1,0,-1):
		delta_ = np.delete(np.multiply(np.multiply(np.matmul(theta[i].T, delta[(r-1) - i]),a[i]), 1 - a[i]),0,axis=0)
		delta.append(delta_)
	delta = np.array(delta)
	a_ = [a[i].T for i in np.arange(len(delta)-1,-1,-1)]


	D = [(1 / n ) * (np.matmul(delta[i], a_[i]) + gamma * theta_reg[(len(delta) - 1) - i ]) for i in np.arange(len(delta)-1,-1,-1)]
	D = np.array([np.asarray(D[i]) for i in np.arange(len(D))])
	return D

def GradientChecking(X,y,theta,gamma,epsilon):
	eps = np.array([np.zeros(theta[i].shape) for i in np.arange(len(theta))])
	D_approx = np.array([np.zeros(theta[i].shape) for i in np.arange(len(theta))])
	for k in np.arange(len(theta)):
		n,m = theta[k].shape
		for i in np.arange(n):
			for j in np.arange(m):
				eps[k][i,j] = epsilon
				D_approx[k][i,j] = (1 / (2 * epsilon)) *  (cost(X,y,theta + eps, 1) - cost(X,y,theta -eps,1))
				eps[k][i,j] = 0
	return D_approx
