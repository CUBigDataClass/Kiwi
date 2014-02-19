#/usr/bin/python

import pandas as pd
import numpy as np
train = pd.read_csv("data/train.csv")
print "Number of Unique Users: " + str(len(train['user'].unique()))
print "Number of Unique SKU: " + str(len(train['sku'].unique()))
print "Number of Unique Categories: " + str(len(train['category'].unique()))
