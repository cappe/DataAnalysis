import csv
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
from scipy import stats

# Load the given dataset
def loadDataset(filename, cols, labeling = False):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for col in range(cols):
			for row in range(len(dataset)):
				dataset[row][col] = float(dataset[row][col])
		
		firstValue = 1
		secondValue = 1

		if (labeling):
			for row in range(len(dataset)):
				dataset[row].append(firstValue)
				dataset[row].append(secondValue)
				if(secondValue < 20):
					secondValue += 1
				else:
					firstValue += 1
					secondValue = 1
		return dataset

# Remove the neighbors inside the dead zone
def removeSameMembers(input_dataset, labels, testInstance, modified):
	reducedDataset = []
	reducedLabels = []
	testInstance_first_value = testInstance[-2]
	testInstance_second_value = testInstance[-1]
	for row in range(len(input_dataset)):
		first_value = input_dataset[row][-2]
		second_value = input_dataset[row][-1]
		if(modified):
			if ((first_value != testInstance_first_value and first_value != testInstance_second_value) and (second_value != testInstance_first_value and second_value != testInstance_second_value)):
					reducedDataset.append((input_dataset[row]))
					reducedLabels.append((labels[row]))
		else:
			if ((first_value != testInstance_first_value or second_value != testInstance_second_value) and (first_value != testInstance_second_value or second_value != testInstance_first_value)):
					reducedDataset.append((input_dataset[row]))
					reducedLabels.append((labels[row]))

	return [reducedDataset, reducedLabels]

# Calculates the euclidean distance between the two instances
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

# Return a list of the closest neighbors for the given k
def getNeighbors(dataset, labels, testInstance, k):
	distances = []
	length = len(testInstance)-2
	for row in range(len(dataset)):
		dist = euclideanDistance(testInstance, dataset[row], length)
		distances.append((labels[row], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for neighbor in range(k):
		neighbors.append(distances[neighbor][0])
	return neighbors

# Returns the majority vote of the closest neighbors
def getResponse(neighbors):
	classVotes = {}
	print neighbors
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

def main():
	input_dataset = loadDataset('proteins.features', 41, True)
	labels = loadDataset('proteins.labels', 1)


	print ''
	
	modified = False
	for i in range(2):
		c_indexes = []
		for k in range(1, 21):
			predictions = []
			actual_labels = []
			for row in range(0, len(input_dataset)):
				testInstance = input_dataset[row]
				reducedData = removeSameMembers(input_dataset, labels, testInstance, modified)
				reducedDataset = reducedData[0]
				reducedLabels = reducedData[1]

				neighbors = getNeighbors(reducedDataset, reducedLabels, testInstance, k)
				result = getResponse(neighbors)

				predictions.append(result)
				actual_labels.append(labels[row])

			c_index = Cindex(actual_labels, predictions)
			c_indexes.append(c_index)

			print 'K: ' + repr(k)
			print 'C-index: ' + repr(c_index)
			print ''
		
		if(modified):
			plt.title('Modified leave-one-out cross-validation')
		else:
			plt.title('Unmodified leave-one-out cross-validation')
		plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], c_indexes)
		plt.axis([1, 20, 0.5, 1.0])
		plt.ylabel('C-index')
		plt.xlabel('K-value')
		plt.show()
		modified = True

main()