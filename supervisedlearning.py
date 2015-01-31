from pprint import pprint
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report


######## GENERATE INSTANCES ########
f = open('heart-data/SPECT.train', 'r')
set_one_train = [ [int(n) for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()

f = open('heart-data/SPECT.test', 'r')
set_one_test = [ [int(n) for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()

one_train_x = [ a[1:] for a in set_one_train ]
one_train_y = [ a[0] for a in set_one_train ]
print "one train_x", one_train_x
one_test_x = [ a[1:] for a in set_one_test ]
one_test_y = [ a[0] for a in set_one_test ]

def generate_report(classifier, predicted, real):
	pass


######## DECISION TREES ########
print "training decision tree"
dt = tree.DecisionTreeClassifier()
dt.fit(one_train_x, one_train_y)

one_pred_y = [ dt.predict(x)[0] for x in one_test_x ]

print dt.score( one_test_x, one_test_y )
print classification_report(one_test_y, one_pred_y, target_names=['class0', 'class1'])

######## NEURAL NETS ########

######## BOOSTING ########
bdt_discrete = AdaBoostClassifier(
    tree.DecisionTreeClassifier(),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")
bdt_discrete.fit(one_train_x, one_train_y)
one_pred_y = [ bdt_discrete.predict(x)[0] for x in one_test_x ]
print bdt_discrete.score( one_test_x, one_test_y )
print classification_report(one_test_y, one_pred_y, target_names=['class0', 'class1'])


######## SUPPORT VECTOR MACHINES ########
svc = svm.SVC()
svc.fit(one_train_x, one_train_y)
one_pred_y = [ svc.predict(x)[0] for x in one_test_x ]
print svc.score( one_test_x, one_test_y )
print classification_report(one_test_y, one_pred_y, target_names=['class0', 'class1'])


######## kNN ########



