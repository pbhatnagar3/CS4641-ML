from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import os
import sys

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
__location__ =os.path.join(__location__, 'DataSet1');

ds = SupervisedDataSet(22, 1);

f = open(os.path.join(__location__, 'SPECT.test'), "r");
for line in f:
	my_list =  line.replace("\r\n", "").split(",")
	# print l
	# print len(my_list)
	current_input = tuple(my_list[:len(my_list)-1])
	current_output = tuple(my_list[len(my_list)-1])
	# print "here is the current input", current_input, "and its length", len(current_input)
	# print "here is the current output", current_output, "and its length", len(current_output)
	ds.addSample(current_input, current_output);	
	# print len(l)

#if you want to check out the length of the training set, then uncomment the last line
#print len(ds)

net = buildNetwork(22, 3, 1);
trainer = BackpropTrainer(net, ds)
x = trainer.trainUntilConvergence()
# print x
# print len(x)
print net.params
#net = buildNetwork(2, 3, 1)
#ds = SupervisedDataSet(2, 1)
trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
