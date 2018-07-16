import numpy as np 
from initalize_theta import initalize_theta as itheta
from decomp_bdiag import decomp_bdiag as db
from cost import cost as cost
from BackPropagation import BackProp as BackProp


def TrainDNN(X,y,hl,alpha,gamma,num_iters):
	n,m = X.shape
	m = np.array([m])
	theta = itheta(m,hl)
	rowct, colct = db(theta)
	Cost_history = np.zeros(num_iters)
	for i in np.arange(num_iters):
		D = BackProp(X,y,theta,gamma)
		theta = theta - (alpha * D)
		Cost_history[i] = cost(X,y,theta,gamma)
	return theta, Cost_history


if __name__ == "__main__":
    import sys
    TrainDNN(int(sys.argv[1]))