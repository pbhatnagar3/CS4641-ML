import json
import numpy as np
import matplotlib.pyplot as plt

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError


class InstanceCollection():
	def __init__(self, x=None, y=None):
		self.x = x # list of instances
		self.y = y # list of corresponding labels


class DataSet():
	def __init__(self, raw_data=None, train_data=None, test_data=None):
		self.raw = raw_data
		self.train = InstanceCollection()
		self.test = InstanceCollection()


print "Generating instances for Inonosphere data\n"
sphere = DataSet()
f = open('ionosphere-data/ionosphere.json', 'r')
data = json.loads(f.read())
# Test and train data for dataset-2 (Ionosphere)
sphere.train.x = [ a[:-1] for a in data['train'] ]
sphere.train.y = [ a[-1] for a in data['train'] ]
sphere.test.x = [ a[:-1] for a in data['test'] ]
sphere.test.y = [ a[-1] for a in data['test'] ]

# train with all 351 instances
sphere.train.x += sphere.test.x
sphere.train.y += sphere.test.y


def test_neural_nets(ds):

	def plot_errors(x, train_err, test_err):
		plt.plot(x, train_err, label='Training error')
		plt.xlabel('Epochs')
		plt.ylabel('Error')
		plt.title('Training error using backpropagation')
		plt.legend()
		plt.show()

	input_size = len(ds.train.x[0]) # no. of attributes
	target_size = 1
	hidden_size = 5
	iterations = 1000

	n = FeedForwardNetwork()
	in_layer = LinearLayer(34)
	hidden_layer = [SigmoidLayer(20), SigmoidLayer(20), SigmoidLayer(20)]
	out_layer = LinearLayer(1)

	n.addInputModule(in_layer)
	for layer in hidden_layer:
		n.addModule(layer)
	n.addOutputModule(out_layer)
	in_to_hidden = FullConnection(in_layer, hidden_layer[0])
	h1 = FullConnection(hidden_layer[0], hidden_layer[1])
	h2 = FullConnection(hidden_layer[1], hidden_layer[2])
	hidden_to_out = FullConnection(hidden_layer[2], out_layer)

	n.addConnection(in_to_hidden)
	n.addConnection(h1)
	n.addConnection(h2)
	n.addConnection(hidden_to_out)

	n.sortModules()

	print n

	train_nnds = SupervisedDataSet(input_size, target_size)
	train_nnds.setField('input', ds.train.x)
	one_train_reshaped = np.array(ds.train.y).reshape(-1,1) 
	train_nnds.setField('target', one_train_reshaped)

	trainer = BackpropTrainer( n, train_nnds )
	epochs, train_acc, test_acc = [], [], []
	
	for i in xrange(iterations):
		trainer.train()
		train_pred_y = []
		# Compute percent training error
		for row in ds.train.x:
			p = int( round( n.activate(row)[0] ) )
			if p >= 1: p = 1 
			else: p = 0 # sometimes rounding takes us to 2 or -1
			train_pred_y.append(p)
		train_error = percentError(train_pred_y, ds.train.y)

		if i%25 == 0 or i==iterations-1:
			epochs.append(i)
			train_acc.append(train_error)
			print "Train error", train_error
	
	plot_errors(epochs, train_acc, test_acc)


print "NEURAL NETS"
nn_sphere = DataSet() # modify sphere to work with neural nets
nn_sphere.train.x = sphere.train.x
nn_sphere.train.y = [ 1 if a=='g' else 0 for a in sphere.train.y ]
nn_sphere.test.x = sphere.test.x
nn_sphere.test.y = [ 1 if a=='g' else 0 for a in sphere.test.y ]
test_neural_nets(nn_sphere)



