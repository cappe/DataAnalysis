import csv
import numpy as np
import math
import operator
from scipy import stats

# Load the given dataset
def loadDataset(filename, cols, normalize=False):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		normalized_data = []
		for col in range(cols):
			column = []
			for row in range(len(dataset)):
				dataset[row][col] = float(dataset[row][col])
				column.append(dataset[row][col])
			if normalize:
				normalized_column = stats.zscore(column)
				for row in range(len(dataset)):
					dataset[row][col] = normalized_column[row]
		return dataset

# Calculate the euclidean distance between the two instances
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

# Remove the neighbors inside the dead zone
def removeNearest(radius, input_dataset, output_dataset, coordinates, test_instance_row):
	reducedDataset = []
	reducedOutput = []
	length = 2
	for row in range(len(input_dataset)):
		if row != test_instance_row:
			dist = euclideanDistance(coordinates[test_instance_row], coordinates[row], length)
			if dist > radius:
				reducedDataset.append((input_dataset[row]))
				reducedOutput.append((output_dataset[row]))
	return [reducedDataset, reducedOutput]

# Return a list of the closest neighbors for the given k
def getNeighbors(dataset, output, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for row in range(len(dataset)):
		dist = euclideanDistance(testInstance, dataset[row], length)
		distances.append((output[row], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for neighbor in range(k):
		neighbors.append(distances[neighbor][0])
	return neighbors

# Return the average of neighbors
def getResponse(neighbors):
	return np.mean(neighbors)

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
	input_dataset = loadDataset('INPUT.csv', 95, True)
	output_dataset = loadDataset('OUTPUT.csv', 1)
	coordinates = loadDataset('COORDINATES.csv', 2)
	
	radius = 0
	while (radius <= 200):
		predictions = []
		actual_values = []
		for row in range(1, len(input_dataset)):
			testInstance = input_dataset[row]
			reducedData = removeNearest(radius, input_dataset, output_dataset, coordinates, row)
			neighbors = getNeighbors(reducedData[0], reducedData[1], testInstance, 5)
			result = getResponse(neighbors)

			predictions.append(result)
			actual_values.append(output_dataset[row])
		
		c_index = Cindex(actual_values, predictions)

		print 'C-index: ' + repr(c_index) + ' when radius: ' + repr(radius)

		radius += 10
		
main()