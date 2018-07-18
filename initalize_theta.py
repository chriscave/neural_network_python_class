import numpy as np 
from scipy.linalg import block_diag


def initalize_theta(m,hl):  #m does not include biases
	l = np.concatenate((m,hl))
	l = np.concatenate((l,np.array([1])))
	theta = []
	for i in np.arange(len(l)-1):
		epsilon = np.sqrt(6) / (np.sqrt(l[i+1]) + np.sqrt(l[i]))
		theta.append((np.random.rand(l[i+1],l[i]+1) * (2 * epsilon) - epsilon))
	return np.array(theta)

if __name__ == "__main__":
    import sys
    initalize_theta(int(sys.argv[1]))