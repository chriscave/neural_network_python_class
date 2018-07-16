import numpy as np
from scipy.linalg import block_diag
from decomp_bdiag import decomp_bdiag as db
from activation import activation as act
from cost import cost as cost




def actrowct(rowct):
	r = len(rowct)
	arowct = np.array([1])
	for i in np.arange(1,r):
	 	k = arowct[i-1] + (rowct[i] - rowct[i-1]) + 1
	 	arowct = np.append(arowct,k)
	return arowct

def BackProp(X,y,theta,gamma):
	rowct, colct =db(theta)
	r = len(rowct)
	arowct = actrowct(rowct)
	n,m = X.shape

	a = act(X,theta)
	D = np.array([])
	delta = np.asmatrix(a[arowct[r-2]-1]- y)
	for i in np.arange(r-3,-1,-1):
		Delta = np.matmul(delta,a[arowct[i]-1:arowct[i+1]-1].T)
		theta_ = theta[rowct[i+1]:rowct[i+2],colct[i+1]+1:colct[i+2]]

		D_ = Delta[:,1::] + gamma * theta_
		d_ = Delta[:,0]
		D_ = (1/n) * np.hstack((d_,D_))
		D = (block_diag(D_,D))

		A = np.matmul(theta_.T,delta)
		H = a[arowct[i]:arowct[i+1]-1]
		delta = np.multiply(np.multiply(A,H), 1 - H)

	X_ = np.concatenate((np.ones((n,1)), X), axis = 1 )
	Delta = np.matmul(delta,X_)
	theta_ =  theta[rowct[0]:rowct[1],colct[0]+1:colct[1]]
	D_ = Delta[:,1::] + gamma * theta_
	d_ = Delta[:,0]
	D_ = (1/n) * np.hstack((d_,D_))
	D = block_diag(D_,D)
	D = np.delete(D,-1,0)
	return D

def GradientChecking(X,y,theta,gamma,epsilon):
	rowct,colct = db(theta)
	n,m = theta.shape
	D_approx = np.zeros((n,m))
	eps = np.zeros((n,m))
	r = len(rowct)
	for k in np.arange(r-1):
		for i in np.arange(rowct[k],rowct[k+1]):
			for j in np.arange(colct[k],colct[k+1]):
				eps[i,j] = epsilon
				approx = (1 / (2 * epsilon)) *  (cost(X,y,theta + eps, gamma) - cost(X,y,theta-eps,gamma))
				D_approx[i,j] = approx
				eps = np.zeros((n,m))
	return D_approx


if __name__ == "__main__":
    import sys
    BackProp(int(sys.argv[1]))