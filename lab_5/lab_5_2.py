import numpy as np

def sigmoid(x):
	# Наша функція активації: f(x) = 1 / (1 + e^(-x)) 
	return 1 / (1 + np.exp(-x))

class Neuron:
	def init (self, weights, bias): 
		self.weights = weights 
		self.bias = bias
	
	def feedforward(self, inputs):
		# Вхідні дані про вагу, додавання зміщення # і подальше використання функції активації
		total = np.dot(self.weights, inputs) + self.bias 
		return sigmoid(total)
weights = np.array([0, 1]) # w1 = 0, w2 = 1 
bias = 4 # b = 4
n = Neuron(weights, bias)
x = np.array([2, 3]) # x1 = 2, x2 = 3 
print(n.feedforward(x))
# ... Тут код із попереднього завдання

class KhokhlovNeuralNetwork: 
	def init (self): 
		weights = np.array([0, 1]) 
		bias = 0 

		# Класс Neuron із попереднього завдання 
		self.h1 = Neuron(weights, bias)
		self.h2 = Neuron(weights, bias) 
		self.o1 = Neuron(weights, bias) 
	
	def feedforward(self, x): 
		out_h1 = self.h1.feedforward(x) 
		out_h2 = self.h2.feedforward(x) 

		# Входи для о1 є виходами h1 и h2 
		out_o1 = self.o1.feedforward(np.array([out_h1, out_h2])) 
		return out_o1 

network = KhokhlovNeuralNetwork() 
x = np.array([2, 3]) 
print(network.feedforward(x)) # 0.7216325609518421