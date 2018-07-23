import numpy as np 
from activation import activation as act
from TrainDNN import TrainDNN as TrainDNN
from activation import sigmoid as sigmoid
from cost import cost as cost
	
class NeuralNetwork:
	
	def __init__(self, hiddenLayers, learningRate, numIters,reg):
		self.hiddenLayers = np.array(hiddenLayers)
		self.learningRate = learningRate
		self.numIters = numIters
		self.reg = reg

	def fit(self, X,y):
		n,m = X.shape
		y = np.matrix(y)
		self.features = np.array([m])
		self.amountOfExamples = np.array([n])
		self.weights, self.costHistory = TrainDNN(X, y, self.hiddenLayers,self.learningRate,self.reg,self.numIters)

	def prob_predict(self,X):
		return act(X,self.weights)[-1]
	
	def predict(self,X):
		a = self.prob_predict(X)
		return np.array([1 if a[0][i] > 0.5 else 0 for i in np.arange(len(a[0]))])

