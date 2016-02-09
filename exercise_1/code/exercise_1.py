import csv
import random
import math
import operator

# Loads the given dataset to trainingSet list
def loadDataset(filename, trainingSet=[]):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
			trainingSet.append(dataset[x])

# Calculates the euclidean distance between the two instances
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

# Returns a list of the closest neighbors for the given k
def getNeighbors(trainingSet, testInstance, k, round):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		if x != round:
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
	for x in range(len(actual_values)):
		if actual_values[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(actual_values))) * 100.0

# The main function to run the program
def main():
	# Prepare the data and init variables
	trainingSet = []
	testInstance = []

	best_k = 0
	best_accuracy = 0

	loadDataset('iris.data', trainingSet)

	# Handle data
	for k in range(1, 150):
		predictions = []
		actual_values = []

		for round in range(0, 149):
			testInstance = trainingSet[round]

			neighbors = getNeighbors(trainingSet, testInstance, k, round)
			result = getResponse(neighbors)
			predictions.append(result)
			actual_values.append(testInstance)
	
		accuracy = getAccuracy(actual_values, predictions)

		if accuracy > best_accuracy:
			best_k = k
			best_accuracy = accuracy

		# Print all the key and their accuracies values
		print 'k = ' + repr(k) + ', Accuracy: ' + repr(accuracy) + '%'

	# Print the best key value and it's accuracy
	print 'Best k: ' + repr(best_k) + ' with the accuracy of: ' + repr(best_accuracy)

main()

