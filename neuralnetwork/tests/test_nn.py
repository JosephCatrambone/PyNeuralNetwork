
import sys, os
import tempfile
import pickle

from unittest import TestCase
import numpy

import neuralnetwork.neuralnetwork as nn

class TestNet(TestCase):
	def test_xor(self):
		net = nn.NeuralNetwork([2, 3, 1], ['tanh', 'tanh', 'tanh'])
		examples = numpy.asarray([ [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.1] ])
		labels = numpy.asarray([ [0.0,], [1.0,], [1.0,], [0.0,] ])

		net.fit(examples, labels, epochs=10000, learning_rate=0.9, momentum=0.3)

		prediction = net.predict(examples)
		self.assertTrue(prediction[0] < 0.1)
		self.assertTrue(prediction[1] > 0.9)
		self.assertTrue(prediction[2] > 0.9)
		self.assertTrue(prediction[3] < 0.1)

	def test_pickle(self):
		# Verify a model can be trained, saved to disk, and loaded back.
		net = nn.NeuralNetwork([2, 3, 1], ['sigmoid', 'sigmoid', 'sigmoid'])
		examples = numpy.asarray([ [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.1] ])
		labels = numpy.asarray([ [0.0,], [1.0,], [1.0,], [0.0,] ])
		net.fit(examples, labels, epochs=200000, learning_rate=0.9, momentum=0.3) # Sigmoid needs more time.

		# Open temp file and pickle network.
		fid, filename = tempfile.mkstemp()
		os.close(fid)
		fout = open(filename, 'w')
		pickle.dump(net, fout, -1)
		fout.close()

		# Reload network from file and remove temp file.
		fin = open(filename, 'r')
		net = pickle.load(fin)
		fin.close()
		os.remove(filename)

		# Verify predictions.
		prediction = net.predict(examples)
		print(prediction)
		self.assertTrue(prediction[0] < 0.1)
		self.assertTrue(prediction[1] > 0.9)
		self.assertTrue(prediction[2] > 0.9)
		self.assertTrue(prediction[3] < 0.1)

	def test_functions(self):
		self.assertTrue(nn.sigmoid(0) == 0.5)
		self.assertTrue(nn.sigmoid(-1e10) < 0.1)
		self.assertTrue(nn.sigmoid(1e10) > 0.9)

		self.assertTrue(nn.tanh(0) == 0)
		self.assertTrue(nn.tanh(-1e10) < 0.1)
		self.assertTrue(nn.tanh(1e10) > 0.9)
		
	def test_gradients(self, places=5, epsilon=1.0e-6):
		step = 0.01
		ran = 10.0 # Area to cover.

		# f'(x) = f(x+h)-f(x)/h
		# f'(x) = f(x+h)-f(x-h)/2h
		def derivative(fun, x, epsilon):
			return (fun(x+epsilon)-fun(x-epsilon))/float(2.0*epsilon)

		def gradient_order(p, q):
			if numpy.max(p) == 0 and numpy.max(q) == 0:
				return 0
			else:
				return numpy.abs(p - q)/max(numpy.max(p), numpy.max(q))

		nums = numpy.arange(-ran, ran, step)
		#self.assertAlmostEqual(nn.delta_sigmoid(nums), derivative(nn.sigmoid, nums, epsilon), places=places)
		self.assertTrue(numpy.all(gradient_order(nn.delta_sigmoid(nums), derivative(nn.sigmoid, nums, epsilon)) < 0.1**places))
		self.assertTrue(numpy.all(gradient_order(nn.delta_tanh(nums), derivative(nn.tanh, nums, epsilon)) < 0.1**places))
		self.assertTrue(numpy.all(gradient_order(nn.delta_linear(nums), derivative(nn.linear, nums, epsilon)) < 0.1**places))
		self.assertTrue(numpy.all(gradient_order(nn.delta_softplus(nums), derivative(nn.softplus, nums, epsilon)) < 0.1**places))


