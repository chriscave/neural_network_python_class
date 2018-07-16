import numpy as np 
from scipy.linalg import block_diag
from decomp_bdiag import decomp_bdiag as db
from activation import activation as act
from initalize_theta import initalize_theta as itheta
from cost import cost as cost
from BackPropagation import BackProp as BackProp
from BackPropagation import GradientChecking as GradientChecking
from TrainDNN import TrainDNN as TrainDNN
	
class NeuralNetwork:
	
	def __init__(self, hiddenLayers, learningRate, numIters,reg):
		self.hiddenLayers = np.array(hiddenLayers)
		self.learningRate = learningRate
		self.numIters = numIters
		self.reg = reg

	def fit(self, X,y):
		n,m = X.shape
		self.features = np.array([m])
		self.amountOfExamples = np.array([n])
		theta = itheta(self.features,self.hiddenLayers)
		self.weights, self.costHistory = TrainDNN(X, y, self.hiddenLayers,self.learningRate,self.reg,self.numIters)


	



hl = np.array([2,3])
X = np.array([[1,2,3,4],[5,6,7,8], [9,10,11,12]])
y = np.array([1,0,0])
n,m = X.shape
m = np.array([m])
theta = itheta(m,hl)

rowct, colct = db(theta)

a = act(X,theta)

r = len(rowct)
num_iters = 500
#print(np.arange(num_iters))
#theta, Cost_history = TrainDNN(X,y,hl,0.03,1,500)
DNN = NeuralNetwork(hl,0.03,200,1)

#print(DNN.__dict__)
DNN.fit(X,y)


print(DNN.weights)