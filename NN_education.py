import sys
import numpy as np
import math
import numpy as np
import sys
import copy
from scipy import stats
class Data(object):
	def __init__(self):
		self.data = []
		self.normalized = []
		self.normalizedLabel = []
		self.normalizedTestData = []
		self.sampleNum = 0
		self.dimension = 0

	def parser(self, file, label, testFile):
		with open(file, 'r') as f:
			f.readline()
			for line in f:
				tempData = line.strip().split(',')
				for i in xrange(len(tempData)):
					tempData[i] = float(tempData[i])
				self.data.append(tempData)
		x = np.array(self.data)
		self.normalized = stats.zscore(x)

		tempLabel = []
		with open(label, 'r') as f:
			for line in f:
				tempLabel.append(float(line.strip()))
		labels = np.array(tempLabel)
		self.normalizedLabel = labels
		self.sampleNum = self.normalized.shape[0]
		self.dimension = self.normalized.shape[1]

		testData = []
		with open(testFile, 'r') as tf:
			tf.readline()
			for line in tf:
				tempTest = line.strip().split(',')
				for i in xrange(len(tempTest)):
					tempTest[i] = float(tempTest[i])
				testData.append(tempTest)
		testing = np.array(testData)
		self.normalizedTestData = stats.zscore(testing)

class NeuralNetwork(object):
	def __init__(self, data):
		# Data object
		self.dataObject = data
		self.trainingData = data.normalized
		self.trainingLabels = data.normalizedLabel
		self.normalizedTestData = data.normalizedTestData
		self.root = None
		self.weightsInputToHidden = []
		self.weightsHiddenToOutput = []
		self.weightInit(3, data.dimension)
		self.MSE = 0

	def weightInit(self, num, dim):
		# Make the random generated "fixed"
		np.random.seed(30)
		self.weightsInputToHidden = np.random.uniform(
											# low = -1/(dim ** 0.5), 
											# high = 1/(dim ** 0.5), 
											low = -4 * ((6 / (dim + num)) ** 0.5),
											high = 4 * ((6 / (dim + num)) ** 0.5),
											size = (num, dim))
		self.weightsHiddenToOutput = np.random.uniform(
											low = -4 * ((6 / (1 + num)) ** 0.5), 
											high = 4 * ((6 / (1 + num)) ** 0.5),
											size = (1, num))


	def forward(self):
		prevMSE = float('inf')
		# 5000 1:20 MSE = 3585
		for iteration in xrange(5001):
		# while prevMSE >= self.MSE and prevMSE - self.MSE > 0.00001:
			W_2 = np.array([[0.0], [0.0], [0.0]])
			W_1 = np.array([[0.0 for _ in xrange(3)] for _ in xrange(5)])

			for i in xrange(len(self.trainingData)):
				first = np.dot(self.weightsInputToHidden, self.trainingData[i])
				# tempFirst is the values that hasn't been "sigmoided"
				tempFirst = first.copy()
				for j in xrange(len(first)):
					first[j] = self.sigmoid(first[j])

				# first stores the data that has been tranformed
				second = np.dot(self.weightsHiddenToOutput, first)

				self.MSE += ((second - self.trainingLabels[i]) ** 2)
				delta_1 = -2 * (self.trainingLabels[i] - second)
				for k in xrange(len(tempFirst)):
					W_2[k] += delta_1 * first[k]

				tempDelta = [[0.0, 0.0, 0.0]]
				for w in xrange(len(self.weightsHiddenToOutput)):
					tempDelta[0][w] = delta_1 * self.weightsHiddenToOutput[w] * self.sigmoid(tempFirst[w]) * (1 - self.sigmoid(tempFirst[w]))
				t = np.array(tempDelta[0][0])
				new = np.reshape(t, (-1, 3))
				trainingData = np.reshape(self.trainingData[i], (5, 1))
				gradient_1 = trainingData.dot(new)
				W_1 += gradient_1
 
				# if prevMSE < self.MSE:
				# 	raise ValueError("BOOOOOOOOOOOOOOOOOOOOM")
			W_1 /= 400.0
			W_2 /= 400.0
			prevMSE = self.MSE
			if iteration < 201:
				learningRate = 0.01
			elif iteration < 3301:
				learningRate = 0.005
			else:
				learningRate = 0.003

			self.weightsInputToHidden -= learningRate * W_1.transpose()
			self.weightsHiddenToOutput -= learningRate * W_2.transpose()
			if iteration % 100 == 0:
				print iteration, self.MSE
				print 
				if iteration == 10000:
					print second, self.trainingLabels[i]
			self.MSE = 0
		
	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))

	def startTraining(self):
		self.forward()
		print 
		print self.weightsInputToHidden
		print 
		print self.weightsHiddenToOutput
		print "TRAINING COMPLETED! NOW PREDICTING."
		self.startPredicting()

	def startPredicting(self):
		tempLabel = []
		with open('education_dev_keys.txt', 'r') as f:
			for line in f:
				tempLabel.append(float(line.strip()))
		labels = np.array(tempLabel)
		testMSE = 0
		for i in xrange(len(self.normalizedTestData)):
			first = np.dot(self.weightsInputToHidden, self.normalizedTestData[i])
			# tempFirst is the values that hasn't been "sigmoided"
			tempFirst = first.copy()
			for j in xrange(len(first)):
				first[j] = self.sigmoid(first[j])

			# first stores the data that has been tranformed
			second = np.dot(self.weightsHiddenToOutput, first)
			print second, labels[i]
			testMSE += ((second - labels[i]) ** 2)
		print testMSE


if __name__ == '__main__':
	trainingFile = sys.argv[1]
	labelFile = sys.argv[2]
	testFile = sys.argv[3]
	parser = Data()
	parser.parser(trainingFile, labelFile, testFile)
	NN = NeuralNetwork(parser)
	NN.startTraining()
