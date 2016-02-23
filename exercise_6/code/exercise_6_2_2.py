import csv
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import random
from scipy import stats

def generateDataset(sampleSize):
	dataset = np.zeros((sampleSize, 1001))
	for col in range(1000):
		column = []
		for row in range(sampleSize):
			column.append(random.randint(0, 100))
		normalized_column = stats.zscore(column)
		for row in range(sampleSize):
			dataset[row][col] = normalized_column[row]
	for row in range(len(dataset)):
		if (row < sampleSize/2):
			dataset[row][-1] = 0
		else:
			dataset[row][-1] = 1
	np.random.shuffle(dataset)
	return dataset

# Calculates the euclidean distance between the two instances
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

# Return a list of the closest neighbors for the given k
def getNeighbors(dataset, testInstance, k, testInstance_row):
	distances = []
	length = len(testInstance)-1
	for row in range(len(dataset)):
		if(row != testInstance_row):
			dist = euclideanDistance(testInstance, dataset[row], length)
			distances.append((dataset[row], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for neighbor in range(k):
		neighbors.append(distances[neighbor][0])
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

# Calculate C-index
def Cindex(actual_values, predictions):
	n = 0
	h_num = 0
	for i in range(0, len(actual_values)):
		t = actual_values[i]
		p = predictions[i]
		for j in range(i+1, len(actual_values)):
			nt = actual_values[j]
			np = predictions[j]
			if (t != nt):
				n += 1
				if (p < np and t < nt) or (p > np and t > nt):
					h_num += 1
				elif (p < np and t > nt) or (p > np and t < nt):
					h_num = h_num
				elif (p == np):
					h_num += 0.5
	c_index = h_num / n
	return c_index

def selectFeatures(dataset, select_count, testInstance_row):
	correlation_values = []
	for col in range(len(dataset[0])-1):
		attribute_col = []
		label_col = []
		for row in range(len(dataset)):
			if(row != testInstance_row):
				attribute_col.append(dataset[row][col])
				label_col.append(dataset[row][-1])
		tau, p_value = stats.kendalltau(attribute_col, label_col)
		correlation_values.append([abs(tau), col])
	
	reduced_dataset = np.zeros((50, select_count + 1))
	correlation_values.sort(key=operator.itemgetter(0), reverse=True)	
	for row in range(len(dataset)):
		for col in range(0, select_count):
			reduced_dataset[row][col] = dataset[row][correlation_values[col][1]]
	for row in range(len(reduced_dataset)):
		reduced_dataset[row][-1] = dataset[row][-1]
	return reduced_dataset

def main():

	for k in range(0, 6):
		dataset = generateDataset(50)

		predictions = []
		actual_labels = []

		for row in range(0, len(dataset)):

			reduced_dataset = selectFeatures(dataset, 10, row)
			testInstance = reduced_dataset[row]

			neighbors = getNeighbors(reduced_dataset, testInstance, 3, row)
			result = getResponse(neighbors)

			predictions.append(result)
			actual_labels.append(testInstance[1])

		c_index = Cindex(actual_labels, predictions)
		
		print ''
		print 'Round: ' + repr(k)
		print ("C-index: " + repr(c_index))
	print ''

main()