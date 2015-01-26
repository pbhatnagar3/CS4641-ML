from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import os
import sys

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
__location__ =os.path.join(__location__, 'DataSet1');

ds = SupervisedDataSet(22, 1);

f = open(os.path.join(__location__, 'SPECT.train'), "r");
for line in f:
	l =  line.replace("\r\n", "").split(",")
	print l
	current_input = tuple(l[:len(l)-1])
	current_output = tuple(l[len(l)-1])
	# print "here is the current input", current_input, "and its length", len(current_input)
	# print "here is the current output", current_output, "and its length", len(current_output)
	ds.addSample(current_input, current_output);	
	# print len(l)

print len(ds)
#net = buildNetwork(2, 3, 1)
#ds = SupervisedDataSet(2, 1)

