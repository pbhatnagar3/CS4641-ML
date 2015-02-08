from pprint import pprint
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle
import numpy as np
from pybrain.utilities import percentError
from sklearn.neighbors import KNeighborsClassifier
import sys


######## GENERATE INSTANCES ########
print "Generating instances for Heart Data"
f = open('heart-data/SPECT.train', 'r')
set_one_train = [ [int(n) for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()
f = open('heart-data/SPECT.test', 'r')
set_one_test = [ [int(n) for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()
# Test and train data for dataset-1 (Heart SPECT)
one_train_x = [ a[1:] for a in set_one_train ]
one_train_y = [ a[0] for a in set_one_train ]
one_test_x = [ a[1:] for a in set_one_test ]
one_test_y = [ a[0] for a in set_one_test ]

f = open('ionosphere-data/ionosphere.data', 'r')
print "Generating instances for Inonosphere data\n"
set_two_raw = [ [float(n) if n not in 'gb' else n for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()
shuffle(set_two_raw) # randomly arranges the instances so they can be split
split_at = int(0.70*len(set_two_raw))
# Test and train data for dataset-2 (Ionosphere)
two_train_x = [ a[:-1] for a in set_two_raw[:split_at] ]
two_train_y = [ a[-1] for a in set_two_raw[:split_at] ]
two_test_x = [ a[:-1] for a in set_two_raw[split_at:] ]
two_test_y = [ a[-1] for a in set_two_raw[split_at:] ]


def generate_report(classifier, predicted, real):
	print classification_report(real, predicted, target_names=['class0', 'class1'])
	print "Confusion Matrix:"
	print confusion_matrix(real, predicted)
	print


######## DECISION TREES ########
print "DECISION TREES\n"
print "Heart SPECT\n"
dt = tree.DecisionTreeClassifier()
dt.fit(one_train_x, one_train_y)
one_pred_y = dt.predict(one_test_x)
print "Score:", dt.score( one_test_x, one_test_y )
generate_report(dt, one_pred_y, one_test_y)
print "Ionosphere\n"
dt = tree.DecisionTreeClassifier()
dt.fit(two_train_x, two_train_y)
two_pred_y = dt.predict( two_test_x )
print "Score:", dt.score( two_test_x, two_test_y )
generate_report( dt, two_pred_y, two_test_y )
print "*"*50


######## NEURAL NETS ########
# TODO: Very bad code. Write a function and call for the two datasets
print "NEURAL NETS"
input_size = len(one_train_x[0]) # no. of attributes
target_size = 1
hidden_size = 5
iterations = 2000

# print "Heart SPECT"
# train_nnds = SupervisedDataSet(input_size, target_size)
# train_nnds.setField('input', one_train_x)
# one_train_reshaped = np.array(one_train_y).reshape(-1,1) 
# train_nnds.setField('target', one_train_reshaped)
# net = buildNetwork( input_size, hidden_size, target_size, bias = True )
# trainer = BackpropTrainer( net, train_nnds )
# for i in xrange(iterations):
# 	train_error = trainer.train()
# 	if i%100 == 0 or i==iterations-1:
# 		print "Train error", train_error
# 		one_pred_y = []
# 		for row in one_test_x:
# 			p = int( round( net.activate(row)[0] ) )
# 			if p >= 1: p = 1 # sometimes rounding takes us to 2
# 			one_pred_y.append(p)
# 		print "Test error: " + str(percentError(one_pred_y, one_test_y))

# print set(one_pred_y), len(one_pred_y)
# print set(one_test_y), len(one_pred_y)
# generate_report(trainer, one_pred_y, one_test_y)

# converting g/b classes to numbers 0/1
temp_train_y, temp_test_y = two_train_y, two_test_y
two_train_y = [ 1 if a=='g' else 0 for a in temp_train_y ]
two_test_y = [ 1 if a=='g' else 0 for a in temp_test_y ]
print "Ionosphere"
input_size = len(two_train_x[0])
print input_size
train_nnds = SupervisedDataSet(input_size, target_size)
train_nnds.setField('input', two_train_x)
two_train_reshaped = np.array(two_train_y).reshape(-1,1) 
train_nnds.setField('target', two_train_reshaped)
net = buildNetwork( input_size, hidden_size, target_size, bias = True )
trainer = BackpropTrainer( net, train_nnds )
for i in xrange(iterations):
	train_error = trainer.train()
	if i%100 == 0 or i==iterations-1:
		print "Train error", train_error
		two_pred_y = []
		for row in two_test_x:
			p = int( round( net.activate(row)[0] ) )
			if p >= 1: p = 1 
			else: p = 0 # sometimes rounding takes us to 2 or -1
			two_pred_y.append(p)
		print "Test error: " + str(percentError(two_pred_y, two_test_y))

print set(two_pred_y), len(two_pred_y)
print set(two_test_y), len(two_pred_y)
generate_report(trainer, two_pred_y, two_test_y)

print "*"*50


######## BOOSTING ########
print "BOOSTING"
bdt_discrete = AdaBoostClassifier(
    tree.DecisionTreeClassifier(),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")
bdt_discrete.fit(one_train_x, one_train_y)
one_pred_y = [ bdt_discrete.predict(x)[0] for x in one_test_x ]
print bdt_discrete.score( one_test_x, one_test_y )
print classification_report(one_test_y, one_pred_y, target_names=['class0', 'class1'])
print "*"*50


######## SUPPORT VECTOR MACHINES ########
print "SUPPORT VECTOR MACHINES"
svc = svm.SVC()
svc.fit(one_train_x, one_train_y)
one_pred_y = [ svc.predict(x)[0] for x in one_test_x ]
print svc.score( one_test_x, one_test_y )
print classification_report(one_test_y, one_pred_y, target_names=['class0', 'class1'])
print "*"*50


######## kNN ########
print "k-NEAREST NEIGHBORS"
neighbors = 7
neigh = KNeighborsClassifier(n_neighbors=neighbors)
neigh.fit(one_train_x, one_train_y)
one_pred_y = [neigh.predict(x)[0] for x in one_test_x]
print neigh.score(one_test_x, one_test_y)
print classification_report(one_test_y, one_pred_y, target_names=['class0', 'class1'])


if __name__ == '__main__':
	print sys.argv
	if len(sys.argv) > 1:
		classifier = sys.argv[2]
