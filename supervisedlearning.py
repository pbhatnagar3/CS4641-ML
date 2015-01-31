from pprint import pprint
from sklearn import tree
from sklearn.metrics import classification_report


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


# for i in range(len(one_train_x)):
# 	print data_set_one[i]
# 	print one_train_y[i], one_train_x[i]

print "training decision tree"
dt = tree.DecisionTreeClassifier()
dt.fit(one_train_x, one_train_y)

one_pred_y = [ dt.predict(x)[0] for x in one_test_x ]
print one_pred_y

print dt.score( one_test_x, one_test_y )
print classification_report(one_test_y, one_pred_y, target_names=['class0', 'class1'])


