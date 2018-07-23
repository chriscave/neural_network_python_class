import numpy as np 
from initalize_theta import initalize_theta as itheta
from cost import cost as cost
from BackPropagation import BackProp as BackProp


def TrainDNN(X,y,hl,alpha,gamma,num_iters):
	n,m = X.shape
	m = np.array([m])
	theta = itheta(m,hl)
	Cost_history = np.zeros(num_iters + 1)
	Cost_history[0] = cost(X,y,theta,gamma)
	for i in np.arange(num_iters):
		D = BackProp(X,y,theta,gamma)
		theta = theta - (alpha * D)
		Cost_history[i+1] = cost(X,y,theta,gamma)
	return theta, Cost_history
