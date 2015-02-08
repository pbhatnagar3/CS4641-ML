# Script To create test and training data for this dataset
from random import shuffle
import json


percent_split = 0.70

f = open('ionosphere.data', 'r')
set_two_raw = [ [float(n) if n not in 'gb' else n for n in line.rstrip().split(',')] for line in f.readlines() ]
f.close()
shuffle(set_two_raw) # randomly arranges the instances so they can be split
split_at = int(percent_split*len(set_two_raw))
# Test and train data for dataset-2 (Ionosphere)
data = {'test': None, 'train' : None}
data['train'] = set_two_raw[:split_at]
data['test'] = set_two_raw[split_at:]

f = open('ionosphere.json', 'w')
f.write( json.dumps(data) )
f.close()