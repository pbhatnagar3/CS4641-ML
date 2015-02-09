from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from pybrain.utilities import percentError
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint
import json
import sys


class InstanceCollection():
	def __init__(self, x=None, y=None):
		self.x = x # list of instances
		self.y = y # list of corresponding labels


class DataSet():
	def __init__(self, raw_data=None, train_data=None, test_data=None):
		self.raw = raw_data
		self.train = InstanceCollection()
		self.test = InstanceCollection()


print "Generating instances for Heart Data"
heart = DataSet()
f = open('heart-data/SPECT.train', 'r')
set_one_train = [ [int(n) for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()
f = open('heart-data/SPECT.test', 'r')
set_one_test = [ [int(n) for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()
heart.train.x = [ a[1:] for a in set_one_train ]
heart.train.y = [ a[0] for a in set_one_train ]
heart.test.x = [ a[1:] for a in set_one_test ]
heart.test.y = [ a[0] for a in set_one_test ]
heart.raw = set_one_train + set_one_test


print "Generating instances for Inonosphere data\n"
sphere = DataSet()
f = open('ionosphere-data/ionosphere.json', 'r')
data = json.loads(f.read())
# Test and train data for dataset-2 (Ionosphere)
sphere.train.x = [ a[:-1] for a in data['train'] ]
sphere.train.y = [ a[-1] for a in data['train'] ]
sphere.test.x = [ a[:-1] for a in data['test'] ]
sphere.test.y = [ a[-1] for a in data['test'] ]
# sphere.raw = 


def generate_report(predicted, real, target_names=['class0', 'class1']):
	print classification_report(real, predicted, target_names=target_names)
	print "Confusion Matrix:"
	print confusion_matrix(real, predicted)
	print '.....................'


def test_decision_trees(ds):
	depths = [None, 2, 4, 6, 8]
	for depth in depths:
		print "Depth:", depth
		dt = tree.DecisionTreeClassifier(max_depth=depth)
		dt.fit(ds.train.x, ds.train.y)
		pred_y = dt.predict(ds.test.x)
		print "Score:", dt.score( ds.test.x, ds.test.y)
		generate_report(pred_y, ds.test.y)

print "DECISION TREES\n"
print "Heart SPECT"
test_decision_trees(heart)

# print "Ionosphere\n"
# dt = tree.DecisionTreeClassifier()
# dt.fit(two_train_x, two_train_y)
# two_pred_y = dt.predict( two_test_x )
# print "Score:", dt.score( two_test_x, two_test_y )
# generate_report( dt, two_pred_y, two_test_y )