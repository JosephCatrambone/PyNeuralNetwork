#!/usr/bin/env python
# Author: Joseph Catrambone <jo.jcat@gmail.com>
# Obtained from https://gist.github.com/JosephCatrambone/b8a6509384d3858974c2
# License:
# The MIT License (MIT)
#
# Copyright (c) 2015 Joseph Catrambone
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function, division
import numpy
import math

def sanity_check(val, val_limit=float('inf')):
	return False # Uncomment to disable
	if numpy.any(numpy.isnan(val)) or numpy.any(numpy.isinf(val)) or numpy.any(val > val_limit) or numpy.any(val < -val_limit):
		print("Breaking")
		import pdb; pdb.set_trace()

# Activation functions
# Delta activation functions da(y) expects y to be a(x).
# dActivation function da(y) expects y to be just x.
# Note the distinction in naming.
def linear(x):
	return x
def delta_linear(x):
	return numpy.ones(x.shape, dtype=numpy.float)

def sigmoid(x):
	return 1.0/(1.0+numpy.exp(-x))
def delta_sigmoid(x):
	# Or, if x = sig(x), then x*(1-x)
	return numpy.multiply(sigmoid(x), (1-sigmoid(x)))
	
def tanh(x):
	return numpy.tanh(x)
def delta_tanh(x):
	return 1 - numpy.power(tanh(x), 2)
	
def softplus(x): # Smooth Relu
	return numpy.log(1 + numpy.exp(x))
def delta_softplus(x):
	return sigmoid(x) # Coincidence

class NeuralNetwork(object):

	def __init__(self, layers, activation_function_names, weight_range=0.1):
		"""Construct a neural network.
		Layers should be an array of sizes. [2, 4, 10, 1] will have an input of two and an output of 1.
		activation_functions should be an array of functions which take an array and return an array.
		delta_activation_functions should take an array and return f'(x) for each x in the array, NOT f'(f(x))."""

		self.weights = list()
		self.biases = list()
		self.activation_functions = list()
		self.delta_activation_functions = list()

		for afn in activation_function_names:
			af, daf = self._activation_lookup(afn)
			self.activation_functions.append(af)
			self.delta_activation_functions.append(daf)

		#for l1, l2 in zip(layers, layers[:1]):
		for index in range(len(layers)-1):
			l1 = layers[index]
			l2 = layers[index+1]
			self.weights.append(numpy.random.uniform(low=-weight_range, high=weight_range, size=(l1,l2)))

		for layer_size in layers:
			self.biases.append(numpy.zeros((1, layer_size))) # Need to dup

	def _add_bias(self, data, bias):
		return data + bias.repeat(data.shape[0], axis=0)

	def _activation_lookup(self, name):
		"""This method maps from a string to the activation function OR, if a tuple of act/dact is provided, returns that."""
		if not isinstance(name, str):
			# Not a string, see if it's a tuple of functions.
			if not (isinstance(name, tuple) and hasattr(name[0], '__call__')):
				raise Exception("Invalid activation function specified: {}".format(name))
			return name[0], name[1]
		else:
			if name.lower() == "linear":
				return linear, delta_linear
			elif name.lower() == "logit" or name.lower() == "logistic" or name.lower() == "sigmoid":
				return sigmoid, delta_sigmoid
			elif name.lower() == "tanh":
				return tanh, delta_tanh
			elif name.lower() == "relu" or name.lower() == "softplus":
				return softplus, delta_softplus

	def predict(self, examples):
		activities, activations = self.forward_propagate(examples)
		return self.activation_functions[-1](activities[-1])

	def forward_propagate(self, examples):
		"""Returns the values and the activations."""
		activities = list() # Preactivations
		activations = list() # After act-func.
		# Populate input
		activities.append(examples)
		activations.append(self.activation_functions[0](examples))
		# Forward prop
		for weight, bias, func in zip(self.weights, self.biases[1:], self.activation_functions):
			preactivation = self._add_bias(numpy.dot(activations[-1], weight), bias)
			activities.append(preactivation)
			activations.append(func(preactivation))
		return activities, activations

	def backward_propagate(self, expected, activities, activations):
		"""Given the expected values and the activities, return the delta_weight and delta_bias arrays."""
		# From Bishop's book,
		# Forward propagate to get activities (a) and activations (z)
		# Evaluate dk for all outputs with dk = yk - yk (basically, get error at output)
		# Backpropagate error dk using dj = deltaAct(aj) * sum(wkj * dk)
		# Use dEn/dwji = dj*zi 
		expected = numpy.atleast_2d(expected)
		
		delta_weights = list()
		delta_biases = list()

		# Calculate blame/error
		last_error = expected - activations[-1] # Linear loss.
		delta = numpy.multiply(last_error, self.delta_activation_functions[-1](activities[-1]))
		delta_biases.append(delta.mean(axis=0))

		# (Dot of weight and delta) * gradient at activity
		for k in range(len(activities)-2, -1, -1):
			layer_error = numpy.dot(delta, self.weights[k].T)
			delta_weights.append(numpy.dot(activations[k].T, delta)) # Get weight change before calculating blame at this level.
			delta = numpy.multiply(layer_error, self.delta_activation_functions[k](activities[k]))
			delta_biases.append(delta.mean(axis=0))

		delta_biases.reverse()
		delta_weights.reverse()

		return delta_weights, delta_biases

	def fit(self, examples, labels, epochs=1000, shuffle_data=True, batch_size=10, learning_rate=0.01, momentum=0.0, early_cutoff=0.0, update_every=0, update_func=None):
		"""Train the neural network on the given examples and labels.
		epochs is the maximum number of iterations that should be spent training the data.
		batch_size is the number of examples which should be used at a time.
		learning_rate is the amount by which delta weights are downsamples.
		momentum is the amount by which old changes are applied.
		early_cutoff is the amount of error which, if a given epoch undershoots, training will cease.
		After 'update_every' epochs (k%up == 0), update_func (if not null) will be called with the epoch and error. """
		# To calculate momentum, maintain the last changes.
		# We prepopulate this list with a bunch of zero matrices since there are no changes at the start.
		last_delta_weights = list()
		last_delta_biases = list()
		for i in range(len(self.weights)):
			last_delta_weights.append(numpy.zeros(self.weights[i].shape))
		for i in range(len(self.biases)):
			last_delta_biases.append(numpy.zeros(self.biases[i].shape))

		for k in range(epochs):
			# Randomly select examples in the list
			samples = numpy.random.randint(low=0, high=examples.shape[0], size=[batch_size,])
			x = numpy.atleast_2d(examples[samples])
			y = numpy.atleast_2d(labels[samples])

			# Forward propagate
			activities, activations = self.forward_propagate(x)

			# Backprop errors
			dws, dbs = self.backward_propagate(y, activities, activations)
			
			# Apply deltas
			for i, dw in enumerate(dws):
				last_delta_weights[i] = last_delta_weights[i]*momentum + ((learning_rate*dw)/float(batch_size))*(1.0-momentum)
				self.weights[i] += last_delta_weights[i]
			for i, db in enumerate(dbs):
				last_delta_biases[i] = last_delta_biases[i]*momentum + ((learning_rate*db)/float(batch_size))*(1.0-momentum)
				self.biases[i] += last_delta_biases[i]
			
			# Calculate error
			error = numpy.sum(numpy.abs(y - self.activation_functions[-1](activities[-1])))
			
			# Check to see if we should call the user's progress function
			if update_every != 0 and update_func is not None and k%update_every == 0:
				update_func(k, error)

			# Break early
			if error < early_cutoff:
				return

if __name__=="__main__":
	examples = numpy.asarray([
		[0.0, 0.0],
		[1.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.1]
	])

	labels = numpy.asarray([
		[0.0,],
		[1.0,],
		[1.0,],
		[0.0,]
	])

	#nn = NeuralNetwork([2, 3, 1], [sigmoid, sigmoid, sigmoid], [delta_sigmoid, delta_sigmoid, delta_sigmoid], weight_range=1.0)
	nn = NeuralNetwork([2, 3, 3, 1], ["tanh", "tanh", "tanh", "linear"])

	#import pdb; pdb.set_trace()

	nn.fit(examples, labels, epochs=100000, learning_rate=0.9, momentum=0.3, update_every=100, update_func=lambda i,x : print("Iteration {}: {}".format(i,x)))

	print(nn.predict(examples))

