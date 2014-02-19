#/usr/bin/python

import pandas as pd                                                         #import pandas
import numpy as np                                                          #import numpy

train = pd.read_csv("data/train.csv")                                       #load train.csv as a pandas frame

print "Number of Rows: " + str(len(train))                                  #prints the total number of rows
print "Number of Unique Users: " + str(len(train['user'].unique()))         #prints the number of unique users...
print "Number of Unique SKU: " + str(len(train['sku'].unique()))
print "Number of Unique Categories: " + str(len(train['category'].unique()))
