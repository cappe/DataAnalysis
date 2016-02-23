import csv
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
import random
from scipy import stats

def generateDataset(sampleSize):
	dataset = []
	normalized_column = []
	for row in range(sampleSize):
		col = []
		col.append(random.randint(0, 100))
		normalized_column.append(col[0])
		if (row < sampleSize/2):
			col.append(0)
		else:
			col.append(1)
		dataset.append(col)

	normalized_column = stats.zscore(normalized_column)
	for row in range(sampleSize):
		dataset[row][0] = normalized_column[row]
	random.shuffle(dataset)

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

def main():

	samples = [10, 50, 100, 500, 2000]

	print ''

	for k in range(len(samples)):
		sample = samples[k]
		c_indexes_over_0_7 = 0
		c_indexes = []
		for x in range(0, 100):
			dataset = generateDataset(sample)
			predictions = []
			actual_labels = []
			for row in range(0, len(dataset)):
				testInstance = dataset[row]

				neighbors = getNeighbors(dataset, testInstance, 3, row)
				result = getResponse(neighbors)

				predictions.append(result)
				actual_labels.append(testInstance[1])

			c_index = Cindex(actual_labels, predictions)
			c_indexes.append(c_index)

		c_index_mean = np.mean(c_indexes)
		c_index_variance = np.var(c_indexes)

		for x in range(0, len(c_indexes)):
			if (c_indexes[x] > 0.7):
				c_indexes_over_0_7 += 1

		c_indexes_percentage_over_0_7 = (float(c_indexes_over_0_7) / len(c_indexes)) * 100

		print ("Sample size: " + repr(sample))
		print ("Mean: " + repr(c_index_mean))
		print ("Variance: " + repr(c_index_variance))
		print ("Percentage of C-indexes over 0.7: " + repr(c_indexes_percentage_over_0_7) + " %")
		print ''
		plt.title("Sample size: " + repr(sample))
		plt.plot(c_indexes)
		plt.axis([0, 100, 0, 1])
		plt.ylabel('C-index')
		plt.xlabel('Round')
		plt.show()

main()