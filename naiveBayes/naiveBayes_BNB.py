#/usr/bin/python

import pandas as pd                                                         #import pandas
import numpy as np                                                          #import numpy
import sys
#import nltk

from sklearn import cross_validation
from sklearn.naive_bayes import *
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.probability import *
import BNB as BNClassifier
import re
import timeit 

def printInfo(train):
	totalnum = len(train)
	print "Number of Rows: " + str(len(train))                                  #prints the total number of rows
	print "Number of Unique Users: " + str(len(train['user'].unique()))         #prints the number of unique users...
	print "Number of Unique SKU: " + str(len(train['sku'].unique()))
	print "Number of query: " + str(len(train['query'].unique()))
	print "Number of Unique Categories: " + str(len(train['category'].unique()))

def preprocessQuery(train):
##### change all the letters to upper case
	dat = train['query'].apply(lambda x: x.upper())
	dat.name = 'query'

	for ind, item in enumerate(dat):
		#litem = item.split()
		#litem.sort()
		#llitem = re.findall(r"[a-zA-Z]+|\d+", item)
		#llitem = re.findall(r"[a-zA-Z0-9]+",item)

		r = re.compile("[0-9]+[a-zA-Z]+[0-9]+|[a-zA-Z]+|[0-9]+")
		llitem = r.findall(item)
		llitem.sort()
		ix = -1
		i360 = -1
		for i, c in enumerate(llitem):
			if c == 'XBOX':
				ix = i
			if c == '360':
				i360 = i

		indlist = []
		indlist = [i for i in xrange(len(llitem)) if i != ix and i != i360]

		ss = ' '.join(llitem[i] for i in indlist)
		if (i360 != -1):
			ss = ss + ' XBOX' + ' 360'
		elif (ix != -1):
			ss = ss + ' XBOX'
		#ss = ' '.join(ss,)
		print ss
		dat[ind] = ss
	return dat

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
        vs[i] = dat.quantile(perc[i])
    datD = dat.apply(nClassify,values=vs)         
    return (datD,vs)
    
####################
# binning the data into numbins bins
####################
def binDivide(dat,numbins):
	bins = np.linspace(dat.min(),dat.max(),numbins+1)
	perc = bins[1:]
	datD = dat.apply(nClassify, values = perc)         
	return (datD,perc)

def preprocessTime(train):
	datClickT = pd.to_datetime(train['click_time'])
	datQueryT = pd.to_datetime(train['query_time'])
	datDiff = (datClickT - datQueryT).apply(lambda x: x / np.timedelta64(1, 's'))  # change date time to float
	#datDiff = datDiff / 300.0   # normalize to 1
	numbins = 100
	#(datTime,vs) = nDivide(datDiff,numbins) # divide into equal bins
	(datTime,vs) = binDivide(datDiff,numbins)
	datTime.name = 'time'
	return datTime

    


def encodeAsInt(dat):
# encode labels as integer
	le = preprocessing.LabelEncoder()
	le.fit(dat)

	datnew = le.transform(dat)
#	for i in xrange(len(le.classes_)):
#		print (np.bincount(datnew))[i],i

	#for i in xrange(len(dat)):
	#	print datnew[i],dat[i]

#	print list(le.classes_)
#	print "# of" ,"is", len(le.classes_)

	return datnew


if __name__=='__main__':

	start = timeit.default_timer()

	train = pd.read_csv("../data/train.csv")                                       #load train.csv as a pandas frame

	train = train.iloc[:40000,:]
	printInfo(train)

#	datUser = train['user']
#	datSKU = train['sku']
#	datCat = train['category']

	print "preprocessing data"
	train['time'] = preprocessTime(train)
	train['query'] = preprocessQuery(train)

	#features = ['user','time','query']
	features = ['category','time','query']
	label = 'sku'
	#Xt = pd.concat([datUser,datTime,datQuery],axis=1)
	Xt = pd.concat([train[features]],axis=1)
	yt = train[label]

	Xtrain = Xt.copy()
	for item in features:
		print "# of " + item +  " is " + str(len(Xtrain[item].unique()))  
#		Xtrain[item] = encodeAsInt(Xt[item]).copy()

#	print Xtrain.columns
#	print Xt.values[1]
#	print Xtrain.values[1]
	
	print "# of " + yt.name +  " is " + str(len(yt.unique()))  
	yt = encodeAsInt(yt)
#	print list(yt)

#le = preprocessing.LabelEncoder()
#le.fit(Yt)
#yt = le.transform(Yt)

#	mnb = BernoulliNB(alpha=0.5,binarize=0.0) 
#	mnb = BernoulliNB(alpha=0.1, binarize=1.0, class_prior=None, fit_prior=True)
#	bnb = BernoulliNB(alpha=0.5, fit_prior=True)
	bnb = BNClassifier.BNB(alpha=0.1)

#	mnb = BernoulliNB(alpha=0.1) 
	
	bnb.fit(Xtrain.values, yt.values)
	ytp =  bnb.predict(Xtrain.values)

	#print [(yt[i],ytp[i]) for i in xrange(len(yt))]	
	acc = (1 - (1. / len(ytp) * sum( yt != ytp )))
	print 'prediction accuracy: %.4f' % acc

	stop = timeit.default_timer()
	print 'time is', stop - start


#gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)

#classifier = nltk.NaiveBayesClassifier.train(labeled_f,estimator=MLEProbDist)
#estimator choices:ELEProbDist, LaplaceProbDist,LidstoneProbDist,MLEProbDist,ConditionalProbDist,

#testDat = d[0:1000]
#print nltk.classify.accuracy(classifier,testDat)  
#print classifier.show_most_informative_features()        
#def lidstone(gamma):
#    return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)
#est = lidstone(0.1)
#classifier = nltk.NaiveBayesClassifier.train(labeled_f,estimator=est)

#cv = cross_validation.KFold(n=len(labeled_f),n_folds=10,indices=True,shuffle=True, random_state=None, k=None)

#scores=[]
#for traincv, testcv in cv:
##    classif = nltk.NaiveBayesClassifier.train(labeled_f[traincv[0]:traincv[len(traincv)-1]],
##                                               estimator=MLEProbDist)
##    print 'accuracy:', nltk.classify.accuracy(classif, labeled_f[testcv[0]:testcv[len(testcv)-1]])
#    classif = nltk.NaiveBayesClassifier.train([labeled_f[i] for i in traincv], estimator = LaplaceProbDist)
#    score = nltk.classify.accuracy(classif, [labeled_f[i] for i in testcv])
#    scores.append(score)
#    print 'accuracy:',  score 
#
#print sum(scores)/len(scores)
