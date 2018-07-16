import numpy as np 
#from scipy.linalg import block_diag
def decomp_bdiag(theta):
	A = np.matmul(theta, theta.T)
	B = np.matmul(theta.T, theta)
	n = A.shape[0]
	m = B.shape[0]
	
	p = np.zeros(n)
	p[0] = 1
	rows = []
	for i in np.arange(n-1):
		if np.array_equal(np.matmul(A, np.diag(p)),np.matmul(np.diag(p),A)):
			rows.append(p)
			p = np.zeros(n)
			p[i+1] = 1
		else:
			p[i+1] = 1
	rows.append(p)

	q = np.zeros(m)
	q[0] = 1
	cols = []
	for i in np.arange(m-1):
		if np.array_equal(np.matmul(B, np.diag(q)),np.matmul(np.diag(q),B)):
			cols.append(q)
			q = np.zeros(m)
			q[i+1] = 1
		else:
			q[i+1] = 1
	cols.append(q)
	#return rows, cols
	rowct = []
	colct = []
	k = len(rows)
	for i in np.arange(k):
		rowct.append(np.nonzero(rows[i])[0][0])
	rowct.append(theta.shape[0])

	l = len(cols)
	for i in np.arange(l):
		colct.append(np.nonzero(cols[i])[0][0])
	colct.append(theta.shape[1])
	return rowct, colct

if __name__ == "__main__":
    import sys
    decomp_bdiag(int(sys.argv[1]))