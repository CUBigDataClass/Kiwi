#/usr/bin/python

from PreprocessData import *
import pandas as pd                                                         #import pandas
import numpy as np                                                          #import numpy
import ModifiedNaiveBayesClassifier as mod_nb
import nltk
from sklearn import cross_validation
from nltk.probability import *


 
def lidstone(gamma):
    return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)               
        
def main():
	train = pd.read_csv("../data/train.csv")                                       #load train.csv as a pandas frame
	train = train.iloc[:20000]
	train['query'] = preprocessQuery(train)

	train['time'] = preprocessTime(train)

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
#    classifier = ModNaiveBayesClassifier.train(labeled_f,estimator=MLEProbDist)
#    ##estimator choices:ELEProbDist, LaplaceProbDist,LidstoneProbDist,MLEProbDist,ConditionalProbDist,
#    #
	trainDat = labeled_f[:10000]
	testDat = labeled_f[10000:]
#    print "using MLE, accuracy", nltk.classify.accuracy(classifier,testDat)  
#    print classifier.show_most_informative_features(5)        
#
	est = lidstone(0.1)
#	est = WittenBellProbDist
	classifier = mod_nb.ModNaiveBayesClassifier.train(labeled_f[:10000],estimator=est)
	
	xt = [dict([(colname, row[i]) for i,colname in featuresE]) for row in train.values] 
	print xt

	ytp =  classifier.batch_classify(xt)
	acc = (1 - (1. / len(ytp) * sum( train.loc[10000:,label] != ytp )))
	print 'prediction accuracy: %.4f' % acc

	print ytp
	print "using lidstone(0.1), accuracy", nltk.classify.accuracy(classifier,testDat) 
	print classifier.show_most_informative_features(5)     
#
#   cv = cross_validation.KFold(n=len(labeled_f),n_folds=10,indices=True,shuffle=False, random_state=None, k=None)
#
#   scores=[]
#   for traincv, testcv in cv:
#       classif = ModNaiveBayesClassifier.train([labeled_f[i] for i in traincv], estimator = est)
#       score = nltk.classify.accuracy(classif, [labeled_f[i] for i in testcv])
#       scores.append(score)
#       print 'accuracy:',  score 

#	print sum(scores)/len(scores)

####################
# run main() as the main function
if __name__=='__main__':
    main()
