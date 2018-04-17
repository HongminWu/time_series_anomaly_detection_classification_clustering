import numpy as np
import ipdb
import csv

data_file = "lp1.data"
fp = open(data_file)
labels = []
ddict  = {}
line = 'init'
while line:
    line   = fp.readline()
    label  = line.strip()
    labels.append(label)
    array = []
    for i in range(0, 15):
        line = fp.readline()
        array.append(map(float, line.strip('\t\n').replace('\t',',').split(',')))
    if label in labels and labels.count(label)>1:
        ddict[label] = np.concatenate((np.array(ddict[label]), np.array(array)), axis=0)
    else:
        ddict[label] = np.array(array)
    line = fp.readline()
    line = fp.readline()
print ddict
