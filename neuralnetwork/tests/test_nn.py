import sys, os
import pickle
from unittest import TestCase

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
