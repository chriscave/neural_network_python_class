import numpy as np 
from decomp_bdiag import decomp_bdiag as db
from activation import activation as act

def xentropy(p,q):
	return np.sum(np.multiply(-p, np.log(q)))

def cost(X,y,theta,gamma):
	n,m = X.shape
	q_ = act(X,theta).T[:,-1]
	q = np.concatenate((q_, 1 - q_)).reshape(2,n).T
	p = np.concatenate((y,1-y)).reshape(2,n).T
	rowct, colct = db(theta)
	theta_reg = np.delete(theta,colct,1)
	return (1/n) * (xentropy(p,q)) + (gamma / (2 * n)) * (np.sum(np.square(theta_reg)))
	
	
	if __name__ == "__main__":
	    import sys
	    cost(int(sys.argv[1]))