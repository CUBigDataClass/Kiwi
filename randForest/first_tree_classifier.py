import pandas as pd
import numpy as np
import re
import string

from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn import tree
from sklearn import preprocessing

#load train.csv as a pandas frame
train = pd.read_csv("../data/train.csv")

#prints basic stats
totalnum = len(train)
print "Number of Rows: " + str(len(train))
print "Number of Unique Users: " + str(len(train['user'].unique()))
print "Number of Unique SKU: " + str(len(train['sku'].unique()))
print "Number of Unique Categories: " + str(len(train['category'].unique()))

#add column for formatted time difference
train['Tdiff'] = (pd.to_datetime(train['click_time']) - pd.to_datetime(train['query_time'])).apply(lambda x: x / np.timedelta64(1, 's'))

#clean up query
train['query'] = train['query'].apply(lambda x: x.upper(),1)
train['query'] = train['query'].apply(lambda x: x.translate(string.maketrans("",""), string.punctuation),1)
train['query'] = train['query'].apply(lambda x: re.sub('(\s+)(a|an|and|the)(\s+)', '\1\3', x),1)
train['query'] = train['query'].apply(lambda x: re.sub(' +',' ',x),1)
train['query'] = train['query'].apply(lambda x: ' '.join(sorted(x.split())),1)  #doesn't seem to help in my train (first 8k) test (next 2k) trial

#store SKU number (answer)
Y = train['sku'].values

#drop uneeded rows
train.drop(['click_time','query_time', 'sku'],inplace=True,axis=1)


# In[44]:

#vectorize data...in actual case we will have to vectorize the training and test data sets together!
vec = DictVectorizer()
train_vec = vec.fit_transform(train[0:10000].T.to_dict().values()) #for now I am only taking a small slice of the actual data
#substitute vectorized with hash trick
#substitute vectorized with LableEncoder (see naieveBays/nnb.py)

r_max = 8000
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_vec.toarray()[0:r_max], Y[0:r_max])

Z = clf.predict(train_vec.toarray()[8001:10000])

#count matches
count = 0
for i in range(0,len(Z)):
    if Z[i]==Y[8001+i]:
        #print str(i) + "\t" + str(Z[i]) + "\t" + str(Y[8001+i])
        count+=1
print count

#accuracy
print float(count) / len(Z)

#accuracy
print float(count) / len(Z)
