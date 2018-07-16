import numpy as np 
from decomp_bdiag import decomp_bdiag as db

def sigmoid(z):
	return 1 / (1 + np.exp(-np.array(z)))

def activation(X,theta): #X does not include the bias nodes
	rowct, colct = db(theta)
	n,m = X.shape
	X = np.concatenate((np.ones((n,1)), X), axis = 1 )
	r = len(rowct)
	H = np.array([])
	act = X.T
	for i in np.arange(r - 2):
		weights = theta[rowct[i]:rowct[i+1], colct[i]:colct[i+1]]
		c = sigmoid(np.matmul(weights,act))
		act = np.concatenate((np.ones((1,n)),c))
		H = np.append(H,act)
	weights = theta[rowct[r-2]:rowct[r-1], colct[r-2]:colct[r-1]]
	H = np.append(H, sigmoid(np.matmul(weights,act)))
	H = H.reshape((rowct[r-1]+len(rowct) - 2,n))
	return H

if __name__ == "__main__":
    import sys
    activation(int(sys.argv[1]))