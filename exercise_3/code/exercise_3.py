import csv
import numpy as np
import math
import operator
from scipy import stats

def loadDataset(filename):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		normalized_data = []
		for col in range(6):
			column = []
			for row in range(len(dataset)):
				dataset[row][col] = float(dataset[row][col])
				column.append(dataset[row][col])
			normalized_column = stats.zscore(column)
			for row in range(len(dataset)):
				dataset[row][col] = normalized_column[row]
		return dataset

# Calculates the euclidean distance between the two instances
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

# Returns a list of the closest neighbors for the given k
def getNeighbors_leave_one(dataset, testInstance, k, test_instance_row):
	distances = []
	length = 3
	for row in range(len(dataset)):
		if row != test_instance_row:
			dist = euclideanDistance([testInstance[3], testInstance[4], testInstance[5]], [dataset[row][3], dataset[row][4], dataset[row][5]], length)
			distances.append((dataset[row], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getNeighbors_leave_three(dataset, testInstance, k, round):
	distances = []
	length = 3
	for row in range(len(dataset)):
		if row < round or row > round + 2:
			dist = euclideanDistance([testInstance[3], testInstance[4], testInstance[5]], [dataset[row][3], dataset[row][4], dataset[row][5]], length)
			distances.append((dataset[row], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

# Returns the majority vote of the closest neighbors
def getResponse(neighbors):
	c_average = []
	cd_average = []
	pb_average = []
	
	for row in range(len(neighbors)):
		c_average.append(neighbors[row][0])
		cd_average.append(neighbors[row][1])
		pb_average.append(neighbors[row][2])
		
	c_average = np.mean(c_average)
	cd_average = np.mean(cd_average)
	pb_average = np.mean(pb_average)

	return [c_average, cd_average, pb_average]

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

def construct_mean_value(testInstances):
	testInstance_mean = [testInstances[0][0], testInstances[0][1], testInstances[0][2]]
	for col in range(3, len(testInstances)+3):
		column = []
		for row in range(len(testInstances)):
			column.append(testInstances[row][col])
		testInstance_mean.append(np.mean(column))
	return testInstance_mean

def leave_one_out():
	dataset = loadDataset('Water_data.csv')

	best_k_c_total = 0
	best_k_cd = 0
	best_k_pb = 0

	best_c_index_c_total = 0
	best_c_index_cd = 0
	best_c_index_pb = 0

	for k in range(1, 100):
		predictions_c_total = []
		predictions_cd = []
		predictions_pb = []

		actual_values_c_total = []
		actual_values_cd = []
		actual_values_pb = []

		for row in range(0, 201):
			testInstance = dataset[row]
			neighbors = getNeighbors_leave_one(dataset, testInstance, k, row)
			result = getResponse(neighbors)

			predictions_c_total.append(result[0])
			predictions_cd.append(result[1])
			predictions_pb.append(result[2])

			actual_values_c_total.append(testInstance[0])
			actual_values_cd.append(testInstance[1])
			actual_values_pb.append(testInstance[2])

		c_index_c_total = Cindex(actual_values_c_total, predictions_c_total)
		c_index_cd = Cindex(actual_values_cd, predictions_cd)
		c_index_pb = Cindex(actual_values_pb, predictions_pb)

		if (c_index_c_total > best_c_index_c_total):
			best_c_index_c_total = c_index_c_total
			best_k_c_total = k

		if (c_index_cd > best_c_index_cd):
			best_c_index_cd = c_index_cd
			best_k_cd = k

		if (c_index_pb > best_c_index_pb):
			best_c_index_pb = c_index_pb
			best_k_pb = k

	print ''
	print ''
	print 'Results for leave one out:'
	print '------------------------------------------------'
	print 'C-index (c_total): ' + repr(best_c_index_c_total) + ' when k: ' + repr(best_k_c_total)
	print 'C-index (Cd): ' + repr(best_c_index_cd) + ' when k: ' + repr(best_k_cd)
	print 'C-index (Pb): ' + repr(best_c_index_pb) + ' when k: ' + repr(best_k_pb)

def leave_three_out():
	dataset = loadDataset('Water_data.csv')

	best_k_c_total = 0
	best_k_cd = 0
	best_k_pb = 0

	best_c_index_c_total = 0
	best_c_index_cd = 0
	best_c_index_pb = 0

	for k in range(1, 100):
		predictions_c_total = []
		predictions_cd = []
		predictions_pb = []

		actual_values_c_total = []
		actual_values_cd = []
		actual_values_pb = []
		
		round = 0
		while (round < 201):
			testInstance = construct_mean_value([dataset[round], dataset[round+1], dataset[round+2]])
			neighbors = getNeighbors_leave_three(dataset, testInstance, k, round)
			result = getResponse(neighbors)

			predictions_c_total.append(result[0])
			predictions_cd.append(result[1])
			predictions_pb.append(result[2])

			actual_values_c_total.append(testInstance[0])
			actual_values_cd.append(testInstance[1])
			actual_values_pb.append(testInstance[2])

			round += 3

		c_index_c_total = Cindex(actual_values_c_total, predictions_c_total)
		c_index_cd = Cindex(actual_values_cd, predictions_cd)
		c_index_pb = Cindex(actual_values_pb, predictions_pb)

		if (c_index_c_total > best_c_index_c_total):
			best_c_index_c_total = c_index_c_total
			best_k_c_total = k

		if (c_index_cd > best_c_index_cd):
			best_c_index_cd = c_index_cd
			best_k_cd = k

		if (c_index_pb > best_c_index_pb):
			best_c_index_pb = c_index_pb
			best_k_pb = k

	print ''
	print ''
	print 'Results for leave three out:'
	print '------------------------------------------------'
	print 'C-index (c_total): ' + repr(best_c_index_c_total) + ' when k: ' + repr(best_k_c_total)
	print 'C-index (Cd): ' + repr(best_c_index_cd) + ' when k: ' + repr(best_k_cd)
	print 'C-index (Pb): ' + repr(best_c_index_pb) + ' when k: ' + repr(best_k_pb)
	print ''
	print ''

leave_one_out()
leave_three_out()