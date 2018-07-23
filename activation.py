import numpy as np 

def sigmoid(z):
	return 1 / (1 + np.exp(-np.array(z, dtype = 'float64')))

def activation(X,theta): #X does not include the bias nodes
	n,m = X.shape
	X = np.concatenate((np.ones((n,1)), X), axis = 1 )
	r = len(theta)
	act = X.T
	H = []
	for i in np.arange(r):
		weights = theta[i]
		c = sigmoid(np.matmul(weights,act))
		act = np.concatenate((np.ones((1,n)),c))
		H.append(act)
	H = np.array(H)
	H[-1] = np.delete(H[-1], 0, axis = 0)
	return H
