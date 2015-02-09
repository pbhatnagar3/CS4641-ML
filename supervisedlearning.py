from sklearn import tree
from sklearn import svm, cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
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


def generate_report(predicted, real):
	print classification_report(real, predicted)
	print "Confusion Matrix:"
	print confusion_matrix(real, predicted)
	print '.....................'


def test_decision_trees(ds):
	depths = [2, 4, 6, 8, None]
	for depth in depths:
		print "Depth:", depth
		dt = tree.DecisionTreeClassifier(max_depth=depth)
		dt.fit(ds.train.x, ds.train.y)
		pred_y = dt.predict(ds.test.x)
		print "Score:", dt.score( ds.test.x, ds.test.y)
		generate_report(pred_y, ds.test.y)

# print "DECISION TREES\n"
# print "Heart SPECT"
# test_decision_trees(heart)
# print "Ionosphere"
# test_decision_trees(sphere)

def test_neural_nets(ds):

	def plot_errors(x, train_err, test_err):
		plt.plot(x, train_err, label='Training accuracy')
		plt.plot(x, test_err, marker='o', linestyle='..', color='r', label='Test Accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.title('Training and test accuracies')
		plt.legend()
		plt.show()

	input_size = len(ds.train.x[0]) # no. of attributes
	target_size = 1
	hidden_size = 10
	iterations = 1000
	train_nnds = SupervisedDataSet(input_size, target_size)
	train_nnds.setField('input', ds.train.x)
	one_train_reshaped = np.array(ds.train.y).reshape(-1,1) 
	train_nnds.setField('target', one_train_reshaped)
	net = buildNetwork( input_size, hidden_size, target_size, bias = True )
	trainer = BackpropTrainer( net, train_nnds )
	epochs, train_acc, test_acc = [], [], []
	
	for i in xrange(iterations):
		trainer.train()
		train_pred_y = []
		# Compute percent training error
		for row in ds.train.x:
			p = int( round( net.activate(row)[0] ) )
			if p >= 1: p = 1 
			else: p = 0 # sometimes rounding takes us to 2 or -1
			train_pred_y.append(p)
		train_error = percentError(train_pred_y, ds.train.y)

		if i%100 == 0 or i==iterations-1:
			epochs.append(i)
			train_acc.append(100. - train_error)
			print "Train error", train_error
			# Compute percent test error
			pred_y = []
			for row in ds.test.x:
				p = int( round( net.activate(row)[0] ) )
				if p >= 1: p = 1 
				else: p = 0 # sometimes rounding takes us to 2 or -1
				pred_y.append(p)
			print "Test error: " + str(percentError(pred_y, ds.test.y))
			test_acc.append(100. - percentError(pred_y, ds.test.y))
	
	plot_errors(epochs, train_acc, test_acc)
	generate_report(pred_y, ds.test.y)

# print "NEURAL NETS"
# test_neural_nets(heart)
# nn_sphere = DataSet() # modify sphere to work with neural nets
# nn_sphere.train.x = sphere.train.x
# nn_sphere.train.y = [ 1 if a=='g' else 0 for a in sphere.train.y ]
# nn_sphere.test.x = sphere.test.x
# nn_sphere.test.y = [ 1 if a=='g' else 0 for a in sphere.test.y ]
# test_neural_nets(nn_sphere)

def test_boosting(ds, algorithm="SAMME"):
	estimators = [10,20,30,50]
	depths = [2,4,6,None]
	print "depth | # estimators | precision | recall | f1-score"
	for depth in depths:
		for e in estimators:
			bdt = AdaBoostClassifier(
		    	tree.DecisionTreeClassifier(max_depth=2),
			    n_estimators=e,
			    algorithm=algorithm)
			bdt.fit(ds.train.x, ds.train.y)
			pred_y = [ bdt.predict(x)[0] for x in ds.test.x ]
			print "Depth:", depth, "Number of estimators:", e
			print bdt.score( ds.test.x, ds.test.y )
			generate_report(pred_y, ds.test.y)
		print "........."
			

# print "BOOSTING"
# print "Heart SPECT"
# test_boosting(heart, algorithm="SAMME")
# print "Ionosphere"
# test_boosting(sphere, algorithm="SAMME.R")


def test_svm(ds):
	kernels = ['linear', 'poly', 'rbf']
	for k in kernels:
		if k != 'linear':
			for gamma in [0.1, 0.5, 1.]:
				svc = svm.SVC(kernel=k, gamma=gamma)
				svc.fit(ds.train.x, ds.train.y)
				pred_y = [ svc.predict(x)[0] for x in ds.test.x ]
				print "Kernel:", k, "Gamma:", gamma
				print svc.score( ds.test.x, ds.test.y )
				generate_report(pred_y, ds.test.y)
		else:
			svc = svm.SVC(kernel=k)
			svc.fit(ds.train.x, ds.train.y)
			pred_y = [ svc.predict(x)[0] for x in ds.test.x ]
			print "Kernel:", k
			print svc.score( ds.test.x, ds.test.y )
			generate_report(pred_y, ds.test.y)


print "SVM"
print "Heart SPECT"
test_svm(heart)
print "Ionosphere"
test_svm(sphere)
print '*'*50




