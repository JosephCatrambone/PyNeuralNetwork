
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
