#/usr/bin/python


from PreprocessData import *
import pandas as pd                                                         #import pandas
import numpy as np                                                          #import numpy
import sys
#import nltk

from sklearn import cross_validation
from sklearn.naive_bayes import *
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.probability import *
import MNB as MNClassifier
import timeit 
    

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

	train = train.iloc[:20000,:]
	printInfo(train)

#	datUser = train['user']
#	datSKU = train['sku']
#	datCat = train['category']

	print "preprocessing data"
	train['time'] = preprocessTime(train)
	train['query'] = preprocessQuery(train)

	#features = ['user','time','query']
	features = ['category','time','query']
	#features = ['user','category','time','query']
	label = 'sku'
	#Xt = pd.concat([datUser,datTime,datQuery],axis=1)
	Xt = pd.concat([train[features]],axis=1)
	yt = train[label]

	Xtrain = Xt.copy()
	print "features are", features
	for item in features:
		print "# of " + item +  " is " + str(len(Xtrain[item].unique()))  
		Xtrain[item] = encodeAsInt(Xt[item])

#	print Xtrain.columns
#	print Xt.values[1]
#	print Xtrain.values[1]
	
	print "Labels are", label
	print "# of " + yt.name +  " is " + str(len(yt.unique()))  
	yt = encodeAsInt(yt)
#	print list(yt)

#le = preprocessing.LabelEncoder()
#le.fit(Yt)
#yt = le.transform(Yt)

#	mnb = BernoulliNB(alpha=0.5,binarize=0.0) 
#	mnb = BernoulliNB(alpha=0.1, binarize=1.0, class_prior=None, fit_prior=True)
#	bnb = BernoulliNB(alpha=0.5, fit_prior=True)
	bnb = MNClassifier.MNB(alpha=0.1)

#	mnb = BernoulliNB(alpha=0.1) 
	
	bnb.fit(Xtrain.values[0:18000], yt.values[0:18000])
	ytp =  bnb.predict(Xtrain.values[18000:,:])

	#print [(yt[18000+i],ytp[i])  for i in xrange(len(ytp))]	
	acc = (1 - (1. / len(ytp) * sum( yt[18000:] != ytp )))
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
