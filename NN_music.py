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
					try:
						tempData[i] = float(tempData[i])
					except:
						tempData[i] = 1 if tempData[i] == "yes" else 0
				self.data.append(tempData)
		x = np.array(self.data)
		trainingMean = np.mean(x, axis = 0)
		trainingStd = np.std(x, axis = 0)
		self.normalized = stats.zscore(x)
		# print self.normalized
		# self.normalized = x
		max1, max2, max3, max4 = self.normalized.max(axis=0)
		# min1, min2, min3, min4 = self.normalized.min(axis=0)
		# print max1, max2, max3, max4, "this is maxs"

		for j in xrange(len(self.normalized)):
			self.normalized[j, 2] = 1 if abs(self.normalized[j][2] - max3) < 1e-02 else 0
			self.normalized[j, 3] = 1 if abs(self.normalized[j][3] - max4) < 1e-02 else 0
		# print self.normalized

		
		tempLabel = []
		with open(label, 'r') as f:
			for line in f:
				tempLabel.append(1 if line.strip() == "yes" else 0)
				# tempLabel.append(float(line.strip()))
		labels = np.array(tempLabel)
		self.normalizedLabel = labels
		self.sampleNum = self.normalized.shape[0]
		self.dimension = self.normalized.shape[1]
		# print self.normalizedLabel
		# print self.sampleNum
		# print self.dimension
		testData = []
		with open(testFile, 'r') as tf:
			tf.readline()
			for line in tf:
				tempTest = line.strip().split(',')
				for i in xrange(len(tempTest)):
					try:
						tempTest[i] = float(tempTest[i])
					except:
						tempTest[i] = 1 if tempTest[i] == "yes" else 0
				testData.append(tempTest)
		testing = np.array(testData)
		# print "training paras" , trainingMean, trainingStd

		self.normalizedTestData = (testing - trainingMean) / trainingStd
		tmax1, tmax2, tmax3, tmax4 = self.normalizedTestData.max(axis=0)
		# print self.normalizedTestData.max(axis=0)
		# print self.normalizedTestData.min(axis=0)

		for k in xrange(len(self.normalizedTestData)):
			self.normalizedTestData[k, 2] = 1 if abs(self.normalizedTestData[k][2] - tmax3) < 1e-10 else 0
			self.normalizedTestData[k, 3] = 1 if abs(self.normalizedTestData[k][3] - tmax4) < 1e-10 else 0
		# print self.normalizedTestData

class NeuralNetwork(object):
	def __init__(self, data):
		# Data object
		self.dataObject = data
		# print data.dimension, "asd"
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
											# low = -1/(float(dim) ** 0.5), 
											# high = 1/(float(dim) ** 0.5), 
											low = -4.0 * ((6.0 / (dim + num)) ** 0.5),
											high = 4.0 * ((6.0 / (dim + num)) ** 0.5),
											size = (num, dim))
		self.weightsHiddenToOutput = np.random.uniform(
											low = -4.0 * ((6.0 / (1 + num)) ** 0.5), 
											high = 4.0 * ((6.0 / (1 + num)) ** 0.5),
											size = (1, num))

		# print self.weightsInputToHidden
		# print self.weightsHiddenToOutput


	def forward(self):
		prevMSE = float('inf')
		# learningRate = 0.05
		# 5000 1:20 MSE = 3585
		for iteration in xrange(3400):
		# while prevMSE >= self.MSE and prevMSE - self.MSE > 0.00001:
			W_2 = np.array([[0.0], [0.0], [0.0]])
			W_1 = np.array([[0.0 for _ in xrange(3)] for _ in xrange(4)])

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
				trainingData = np.reshape(self.trainingData[i], (4, 1))
				gradient_1 = trainingData.dot(new)
				W_1 += gradient_1
 
				if prevMSE < self.MSE:
					raise ValueError("BOOOOOOOOOOOOOOOOOOOOM")
			W_1 /= 100.0
			W_2 /= 100.0
			# if prevMSE < self.MSE:
				# learningRate /= 5.0
			prevMSE = self.MSE
			# if iteration < 400:
			# 	learningRate = 0.1
			if iteration < 301:
				learningRate = 0.1
			elif iteration < 1401:
				learningRate = 0.005
			elif iteration < 3201:
				learningRate = 0.003
			else:
				learningRate = 0.001

			self.weightsInputToHidden -= learningRate * W_1.transpose()
			self.weightsHiddenToOutput -= learningRate * W_2.transpose()
			# if iteration % 100 == 0:
				# print iteration, self.MSE
				# print 
			print iteration, 0.5 * self.MSE[0]

			self.MSE = 0
		
	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))

	def startTraining(self):
		self.forward()
		# print 
		# print self.weightsInputToHidden
		# print 
		# print self.weightsHiddenToOutput
		print "TRAINING COMPLETED! NOW PREDICTING."
		self.startPredicting()

	def startPredicting(self):
		tempLabel = []
		with open('music_dev_keys.txt', 'r') as f:
			for line in f:
				tempLabel.append(1 if line.strip() == "yes" else 0)
		
		labels = np.array(tempLabel)
		total = len(labels)
		error = 0
		testMSE = 0

		for i in xrange(len(self.normalizedTestData)):
			first = np.dot(self.weightsInputToHidden, self.normalizedTestData[i])
			# tempFirst is the values that hasn't been "sigmoided"
			tempFirst = first.copy()
			for j in xrange(len(first)):
				first[j] = self.sigmoid(first[j])

			# first stores the data that has been tranformed
			second = np.dot(self.weightsHiddenToOutput, first)
			# print second, 1 if second > 0.5 else 0, labels[i]
			print "yes" if second > 0.5 else "no"
			error += abs((1 if second > 0.5 else 0) - labels[i])
			testMSE += ((second - labels[i]) ** 2)
		# print "total", total
		# print "error", error
		# print "error rate", error/float(total) 
		# print testMSE


if __name__ == '__main__':
	trainingFile = sys.argv[1]
	labelFile = sys.argv[2]
	testFile = sys.argv[3]
	parser = Data()
	parser.parser(trainingFile, labelFile, testFile)
	NN = NeuralNetwork(parser)
	NN.startTraining()
