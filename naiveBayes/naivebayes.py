#/usr/bin/python

import pandas as pd                                                         #import pandas
import numpy as np                                                          #import numpy
import nltk
from sklearn import cross_validation
from nltk.probability import *

train = pd.read_csv("../data/train.csv")                                       #load train.csv as a pandas frame

totalnum = len(train)
print "Number of Rows: " + str(len(train))                                  #prints the total number of rows
print "Number of Unique Users: " + str(len(train['user'].unique()))         #prints the number of unique users...
print "Number of Unique SKU: " + str(len(train['sku'].unique()))
#print "Number of Unique Categories: " + str(len(train['category'].unique()))

datUser = train['user']
datSKU = train['sku']
datCat = train['category']

datQuery = train['query']
datQuery = datQuery.apply(lambda x: x.upper())

datClickT = pd.to_datetime(train['click_time'])
datQueryT = pd.to_datetime(train['query_time'])
datDiff = (datClickT - datQueryT).apply(lambda x: x / np.timedelta64(1, 's'))  # change date time to float
datDiff = datDiff / 300.0   # normalize to 1
def nClassify(x,values):
    if x < values[0]:
        return 0
    else:    
        for i in range(1,len(values) - 1,1):
            if (x < values[i+1] and x > values[i]):
                return i
        return (len(values)-1)

def nDivide(dat,n):
    vs = np.zeros(n-1)
    perc = np.linspace(0,1,n+1)
    perc = perc[1:-1]
    for i in range(len(perc)):
        vs[i] = datDiff.quantile(perc[i])
    datD = dat.apply(nClassify,values=vs)         
    return datD
    
datTime = nDivide(datDiff,4)
datTime.name = 'time'

newData = pd.concat([datUser,datQuery,datTime],axis=1)
features = newData.columns
featuresE = enumerate(features)

d = [(dict([(colname,row[i])
    for i,colname in enumerate(features)
    ]),
    datSKU[j]) 
    for j,row in enumerate(newData.values)
    ]  

labeled_f = d
#gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)
classifier = nltk.NaiveBayesClassifier.train(labeled_f,estimator=MLEProbDist)
#estimator choices:ELEProbDist, LaplaceProbDist,LidstoneProbDist,MLEProbDist,ConditionalProbDist,

testDat = d[0:1000]
print nltk.classify.accuracy(classifier,testDat)  
#print classifier.show_most_informative_features()        
#def lidstone(gamma):
#    return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)
#est = lidstone(0.1)
#classifier = nltk.NaiveBayesClassifier.train(labeled_f,estimator=est)

cv = cross_validation.KFold(n=len(labeled_f),n_folds=10,indices=True,shuffle=True, random_state=None, k=None)

scores=[]
for traincv, testcv in cv:
#    classif = nltk.NaiveBayesClassifier.train(labeled_f[traincv[0]:traincv[len(traincv)-1]],
#                                               estimator=MLEProbDist)
#    print 'accuracy:', nltk.classify.accuracy(classif, labeled_f[testcv[0]:testcv[len(testcv)-1]])
    classif = nltk.NaiveBayesClassifier.train([labeled_f[i] for i in traincv], estimator = LaplaceProbDist)
    score = nltk.classify.accuracy(classif, [labeled_f[i] for i in testcv])
    scores.append(score)
    print 'accuracy:',  score 

print sum(scores)/len(scores)