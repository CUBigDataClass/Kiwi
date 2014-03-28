#/usr/bin/python

import pandas as pd                                                         #import pandas
import numpy as np                                                          #import numpy

train = pd.read_csv("../data/train.csv")                                       #load train.csv as a pandas frame

print "Number of Rows:\t\t\t" + str(len(train))                                  #prints the total number of rows
print "Number of Unique Users:\t\t" + str(len(train['user'].unique()))         #prints the number of unique users...
print "Number of Unique SKU:\t\t" + str(len(train['sku'].unique()))
print "Number of Unique Categories:\t" + str(len(train['category'].unique()))

testisnull = pd.isnull(train['sku'])

for i in testisnull:
  if (i == True):
    print "null"

