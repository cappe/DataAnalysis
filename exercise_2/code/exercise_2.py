import csv
import random
import math
import operator
from collections import OrderedDict

# Loads the given dataset to trainingSet list
def loadDataset(filename):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		return list(lines)

# Calculates the euclidean distance between the two instances
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

# Returns a list of the closest neighbors for the given k
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getCrossValidationNeighbors(trainingSet, testInstance, testIndexes, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		if (x < testIndexes[0]) or (x > testIndexes[1]):
			dist = euclideanDistance(testInstance, trainingSet[x], length)
			distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

# Returns the majority vote of the closest neighbors
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

# Returns the accuracy in percentage between the predicted and actual values
def getAccuracy(actual_values, predictions):
	correct = 0
	confusionMatrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
	for x in range(len(predictions)):
		confusionMatrixEvaluation(actual_values[x][-1], predictions[x], confusionMatrix)

	precision = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FP'])
	recall = confusionMatrix['TP'] / (confusionMatrix['TP'] + confusionMatrix['FN'])
	f_score = 2 * ((precision * recall) / (precision + recall))

	print confusionMatrix
	print 'Precision: ' + repr(precision)
	print 'Recall: ' + repr(recall)

	return f_score

def confusionMatrixEvaluation(actualValue, predictionValue, matrix):
	if(actualValue == 1):
		if(predictionValue == 1):
			# True positive
			matrix['TP'] += float(1)
		else:
			# False negative
			matrix['FN'] += float(1)
	else:
		if(predictionValue == 1):
			# False positive
			matrix['FP'] += float(1)
		else:
			# True negative
			matrix['TN'] += float(1)

def replaceStrings(trainingSet, lineNumber, classes = []):
	classifiers = OrderedDict()
	for line in trainingSet:
		instance = line[lineNumber]
		if instance in classifiers:
			classifiers[instance] += 1
		else:
			classifiers[instance] = 1

	if (len(classes) == 0):
		for key in classifiers:
			classes.append(key)

	for line in trainingSet:
		length = len(classes)
		for index in range(0, length):
			key = classes[index]
			if (line[lineNumber] == key):
				line[lineNumber] = index

def convertDataToFloat(dataSet):
	for x in range(len(dataSet)):
		for y in range(42):
			dataSet[x][y] = float(dataSet[x][y])

def kNN(trainingSet, testSet):
	for k in range(1, 11):
		print 'k = ' + repr(k)
		predictions = []
		for x in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[x], k)
			result = getResponse(neighbors)
			predictions.append(result)

		accuracy = getAccuracy(testSet, predictions)
		print 'F-Score: ' + repr(accuracy)
		print ''

def crossValidation(trainingSet):
	instances = len(trainingSet)
	foldSize = instances / 10
	for k in range(1, 11):
		print 'k = ' + repr(k)
		testIndexes = [0, foldSize]
		predictions = []
		for x in range(0, 10):
			for y in range(testIndexes[0], testIndexes[1]):
				neighbors = getCrossValidationNeighbors(trainingSet, trainingSet[y], testIndexes, k)
				result = getResponse(neighbors)
				predictions.append(result)

			testIndexes[0] += foldSize
			testIndexes[1] += foldSize
		accuracy = getAccuracy(trainingSet, predictions)
		print 'F-Score for 10-fold cross validation on the training data: ' + repr(accuracy)
		print ''

# The main function to run the program
def main():
	trainingSet = loadDataset('TrainData.csv')
	testSet = loadDataset('TestData.csv')
	portClasses = []
	transferClasses = []
	auxiliaryClasses = []
	
	replaceStrings(trainingSet, 1, portClasses)
	replaceStrings(trainingSet, 2, transferClasses)
	replaceStrings(trainingSet, 3, auxiliaryClasses)
	
	replaceStrings(testSet, 1, portClasses)
	replaceStrings(testSet, 2, transferClasses)
	replaceStrings(testSet, 3, auxiliaryClasses)

	convertDataToFloat(trainingSet)
	#convertDataToFloat(testSet)

	crossValidation(trainingSet)
	#kNN(trainingSet, testSet)

main()
