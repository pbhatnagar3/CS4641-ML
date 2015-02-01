from pprint import pprint
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import numpy as np
from pybrain.utilities import percentError
from sklearn.neighbors import KNeighborsClassifier
import sys


######## GENERATE INSTANCES ########
f = open('heart-data/SPECT.train', 'r')
set_one_train = [ [int(n) for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()

f = open('heart-data/SPECT.test', 'r')
set_one_test = [ [int(n) for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()

one_train_x = [ a[1:] for a in set_one_train ]
one_train_y = [ a[0] for a in set_one_train ]
one_test_x = [ a[1:] for a in set_one_test ]
one_test_y = [ a[0] for a in set_one_test ]


def generate_report(classifier, predicted, real):
	print classification_report(real, predicted, target_names=['class0', 'class1'])
	print confusion_matrix(real, predicted)


######## DECISION TREES ########
print "DECISION TREES"
print "training decision tree"
dt = tree.DecisionTreeClassifier()
dt.fit(one_train_x, one_train_y)
one_pred_y = [ dt.predict(x)[0] for x in one_test_x ]
print one_pred_y
print dt.score( one_test_x, one_test_y )
generate_report(dt, one_pred_y, one_test_y)
print "*"*50


######## NEURAL NETS ########
print "NEURAL NETS"
input_size = 22
target_size = 1
train_nnds = SupervisedDataSet(input_size, target_size)
train_nnds.setField('input', one_train_x)
# print one_train_y
one_train_reshaped = np.array(one_train_y).reshape(-1,1) 
train_nnds.setField('target', one_train_reshaped)
# train_nnds.setField('target', one_train_y)

hidden_size = 5

net = buildNetwork( input_size, hidden_size, target_size, bias = True )
trainer = BackpropTrainer( net, train_nnds )

print "training the neural net ..."
# trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )
trainer.trainEpochs( 1000 )
print "Predicting with the neural network"
one_pred_y = [ int( round( net.activate(row)[0] ) ) for row in one_test_x ]
print "Test error: " + str(percentError(one_pred_y, one_test_y))
print confusion_matrix(one_test_y, one_pred_y)
try:
	print classification_report(one_test_y, one_pred_y, target_names=['class0', 'class1'])
except:
	print one_test_y
	print one_pred_y
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
