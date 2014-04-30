#/usr/bin/python

import pandas as pd                                                         #import pandas
import numpy as np                                                          #import numpy
import nltk
from sklearn import cross_validation
from nltk.probability import *
from collections import defaultdict
import timeit 


def preprocessQuery(train):
    train['query'] = train['query'].apply(lambda x: x.upper())
    return train

def nClassify(x,values):
    if x < values[0]:
        return 0
    else:    
        for i in range(1,len(values) - 1,1):
            if (x < values[i+1] and x > values[i]):
                return i
        return (len(values)-1)

def binDivide(dat,numbins):
	print dat.min()
	print dat.max()
	bins = np.linspace(dat.min(),dat.max(),numbins+1)
	perc = bins[1:]
	datD = dat.apply(nClassify, values = perc)         
	return (datD,perc)

def nDivide(dat,n):
    vs = np.zeros(n-1)
    perc = np.linspace(0,1,n+1)
    perc = perc[1:-1]
    for i in range(len(perc)):
        vs[i] = dat.quantile(perc[i])
    datD = dat.apply(nClassify,values=vs)         
    return (datD,vs)
    
def preprocessTime(train):
    datClickT = pd.to_datetime(train['click_time'])
    datQueryT = pd.to_datetime(train['query_time'])
    datDiff = (datClickT - datQueryT).apply(lambda x: x / np.timedelta64(1, 's'))  # change date time to float
    datDiff = datDiff / 300.0   # normalize to 1
    (datTime,vs) = binDivide(datDiff,100)
#    datTime.name = 'time'
    train['time'] = datTime
    return train

class ModNaiveBayesClassifier(nltk.NaiveBayesClassifier):
    @staticmethod   
    def train(labeled_featuresets, estimator=ELEProbDist):
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()
        
        # Count up how many times each feature value occurred, given
        # the label and featurename.
        for featureset, label in labeled_featuresets:
            label_freqdist.inc(label)
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                feature_freqdist[label, fname].inc(fval)
                # Record that fname can take the value fval.
                feature_values[fname].add(fval)
                # Keep a list of all feature names.
                fnames.add(fname)
        
        # If a feature didn't have a value given for an instance, then
        # we assume that it gets the implicit value 'None.'  This loop
        # counts up the number of 'missing' feature values for each
        # (label,fname) pair, and increments the count of the fval
        # 'None' by that amount.
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                feature_freqdist[label, fname].inc(None, num_samples-count)
                feature_values[fname].add(None)
        
        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist,bins = len(label_freqdist))
        
        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname])) 
            feature_probdist[label,fname] = probdist
        
        return ModNaiveBayesClassifier(label_probdist, feature_probdist)
 
def lidstone(gamma):
    return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)               
        
def main():
	start = timeit.default_timer()		
	train = pd.read_csv("../data/train.csv")                                       #load train.csv as a pandas frame
	train = preprocessQuery(train)

	train = preprocessTime(train)

	features=['time','user','query']
	label = 'sku'
#newData = pd.concat([datUser,datQuery,datTime],axis=1)
	featuresE = set((j,item) for j,item in enumerate(train.columns) if item in features)

	labeled_f = [(dict([(colname, row[i])
	for i,colname in featuresE
	]),
	train.loc[j,label]) 
	for j,row in enumerate(train.values)
	]  
	classifier = ModNaiveBayesClassifier.train(labeled_f,estimator=MLEProbDist)
##estimator choices:ELEProbDist, LaplaceProbDist,LidstoneProbDist,MLEProbDist,ConditionalProbDist,
#
	testDat = labeled_f
	print "using MLE, accuracy", nltk.classify.accuracy(classifier,testDat)  
#print classifier.show_most_informative_features(5)        

#    est = lidstone(1.0)
#    classifier = ModNaiveBayesClassifier.train(labeled_f,estimator=est)
#    print "using lidstone(1.0), accuracy", nltk.classify.accuracy(classifier,testDat) 
#    print classifier.show_most_informative_features(5)     
#    #
#    cv = cross_validation.KFold(n=len(labeled_f),n_folds=10,indices=True,shuffle=False, random_state=None, k=None)
#    #
#    scores=[]
#    for traincv, testcv in cv:
#        classif = ModNaiveBayesClassifier.train([labeled_f[i] for i in traincv], estimator = est)
#        score = nltk.classify.accuracy(classif, [labeled_f[i] for i in testcv])
#        scores.append(score)
#        print 'accuracy:',  score 
#    
#    print sum(scores)/len(scores)

	stop = timeit.default_timer()
	print 'time is', stop - start

####################
# run main() as the main function
if __name__=='__main__':
    main()
