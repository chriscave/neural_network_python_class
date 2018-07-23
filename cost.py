import numpy as np 
from activation import activation as act
import unittest

def xentropy(p,q):
	return np.sum(np.multiply(-p, np.log(q)))


def cost(X,y,theta,gamma):
	n,m = X.shape
	q_ = act(X,theta)[-1].T
	q = np.concatenate((q_, 1 - q_), axis = 1)
	p = np.concatenate((y.T,1-y.T),axis = 1) #here y needs to be a matrix not a numpy series
	theta_reg = np.array([np.delete(theta[i],0,axis = 1) for i in np.arange(len(theta))])
	return (1/n)* xentropy(p,q) + (gamma / (2*n)) * np.sum([np.sum(np.square(theta_reg)[i]) for i in np.arange(len(theta))])

