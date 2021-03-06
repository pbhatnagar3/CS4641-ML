# These are all the header files that are stated on the pybrain website
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
# For the GRAPHICAL OUTPUT

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

# making and printing some clean datasets
#we are making some mean values
means = [(-1,0),(2,4),(3,1)]
print "Contents of means", means

#creating some covariance matrix
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
print "Contents of cov", cov

# creating the Dataset to add all the data
#arguments:
	# input, output, nb_classes=3

alldata = ClassificationDataSet(2, 1, nb_classes=3)
print "initial contents of the data", alldata
for n in xrange(400):
    for klass in range(3):
    	# print "value of klass", klass
    	#so we are choosing some value of mean and some value of variance and then 
    	#then adding all the data to the sample. 
        input = multivariate_normal(means[klass],cov[klass])
        # print "here is the input", input
        alldata.addSample(input, [klass])

print "the length of the final dataset is ", len(alldata)
#I am finding the length of the entire dataset and that is expected to be 1200 which it is. The
# the thing that I am more concerned about is that when you print out the alldata, you get a wierd 
# number like 2056 or 2048
print "here is the whole data "
print "*"*20
print alldata
# splitting the data between the trainings_et and testing_set
tstdata_temp, trndata_temp = alldata.splitWithProportion( 0.25 )
# tstdata, trndata = alldata.splitWithProportion( 0.25 )

#here are are just copying the data from the temp variables to the actual vaiables that I will be using
tstdata = ClassificationDataSet(2, 1, nb_classes=3)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(2, 1, nb_classes=3)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

print "here is the type of data", type(trndata)
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

print "here is the modified training data", trndata

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)


ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy
for i in range(20):
	trainer.trainEpochs( 1 )
	trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
	tstresult = percentError( trainer.testOnClassData(dataset=tstdata ),tstdata['class'] )
	print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
	out = fnn.activateOnDataset(griddata)
	out = out.argmax(axis=1)  # the highest output activation gives the class
	out = out.reshape(X.shape)

	figure(1)
	ioff()  # interactive graphics off
	clf()   # clear the plot
	hold(True) # overplot on
	for c in [0,1,2]:
		here, _ = where(tstdata['class']==c)
		plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
	if out.max()!=out.min():  # safety check against flat field
		contourf(X, Y, out)   # plot the contour

ioff()
show()