#/usr/bin/python

import numpy as np
import math
from scipy import sparse
import sys


def countFreq(kk,dd):
	if (kk in dd):
		dd[kk] += 1
	else:
		dd[kk] = 1

# count freq of word in dic[tag]
# if tag is not in dic, create a new subdictory for dic[tag]
def countDF(word,tag,dic):
	if tag not in dic:
			dic[tag]={}

	countFreq(word,dic[tag])


### smoothing
#def smoothData(dic,totalnum, featurenum, alpha = 1.0):
#	denom = (float(totalnum) + alpha * float(featurenum))
#	sdic = {}
#	for item in dic:
#		val = dic[item]
#		sval = (float(val) + alpha) / denom
#		sdic[item] = val
#		return sdic

def log(x):
	if ( x > 0.0 ):
			return math.log(x)
	else:
			return -99999999999.0 

##################################################
### bernoulli classifier
##################################################
class BNB:
	def __init__(self,alpha=1.0):
		self.alpha = alpha # smoothing parameter
		self.fn = 0 # num of features
		self.ln = 0 # num of unique classes
		self.isDataFit = False # where the classifier is fitted
		self.unkfeature = '<UNK_F>' # for unknown words
		self.unklabel = '<UNK_L>' # for unknown labels, unused now
		self.labels = [] # list fot unique labels
		self.dictlist = {} # diclist for P(feature|given label)
		self.pidic = {} # prior dic for P(label)

	##################################################
	## init prior matrix with normalization
	##################################################
	def init_pidic(self,labels):
		pidic = {}
		labeldic = {}
		numlabels = len(labels)
		for j in xrange(numlabels):
			tag = labels[j]
			countFreq(tag, labeldic)
		####################
		# normalize
		####################
		for item in labeldic:
			pidic[item] = float(labeldic[item]) / float(numlabels)

		return (labeldic,pidic)

	##################################################
	## init feature-label pairing with smoothing
	## smoothing factor = self.alpha
	##################################################
	def init_emit_1v1(self,features,labels,labelcount):
		classdic = {}
		for j in xrange(len(features)):
			val = features[j]
			tag = labels[j]

			countDF(val,tag,classdic)

		####################
		# smoothing
		####################
		fnum = len(np.unique(features))
		for label in classdic:
			denom = (float(labelcount[label]) + self.alpha * float(fnum))
			for feature in classdic[label]:
				val = classdic[label][feature]
				classdic[label][feature] = (float(val) + self.alpha)/denom
		###unknown
			classdic[label][self.unkfeature] = self.alpha / denom
			#sys.exit()

		return classdic

	##################################################
	## fit features to labels
	##################################################
	def fit(self,features,labels):
		# num of features
		self.isDataFit = True
		#features = np.asarray(features)
		#labels = np.asarray(labels)
		if (np.ndim(features) == 1):
			datanum = features.shape[0]
			self.fn = 1
			features = np.reshape(features,(datanum,1))
		else:
			datanum, self.fn = features.shape

		print datanum, self.fn
		
		####################
		## check the size of the data
		####################
		if (datanum != len(labels)):
			print "Error: length of labels not equal to number of data"
			sys.exit()

		self.labels, indices = np.unique(labels,return_inverse=True)
		self.ln = len(self.labels)

		####################
		#### generate prior matrix
		####################
		labelcount,self.pidic = self.init_pidic(labels)

		####################
		# generate dictlist
		####################
		self.dictlist = {}
		for i in xrange(self.fn):
			self.dictlist[i] = self.init_emit_1v1(features[:,i],labels,labelcount)

		#	print self.dictlist[i][self.labels[1]]

	def readTagProb(self,label,tag):
		val =  log(self.pidic[label])
		for j in xrange(len(tag)):
			item = tag[j]
			if item in self.dictlist[j][label]:
				val += log(self.dictlist[j][label][item])
			else:
				val += log(self.dictlist[j][label][self.unkfeature])

		return val

	##################################################
	## predict a single feature
	##################################################
	def predictOne(self,tag):
		maxprob = log(0.0)
		maxlabel = []
		for k in xrange(self.ln):
			label = self.labels[k]
			cutprob =  self.readTagProb(label,tag)

			if maxprob < cutprob:
				maxprob = cutprob
				maxlabel = label

		return maxlabel

	##################################################
	## predict features
	##################################################
	def predict(self,feature):
		if not self.isDataFit:
			print "Classes not fit, cannot predict!"
			sys.exit()

		#feature = np.asarray(feature)
		if (np.ndim(feature) == 1):
			dataLen = feature.shape[0]
			numTF = 1
			feature = np.reshape(feature,(dataLen,1))
		else:
			dataLen, numTF = feature.shape
### check whether the feature is the same as training set
		if ( numTF != self.fn):
			print "Error: Size of test feature is different from training set!"
			sys.exit()

		pLabel = np.empty(dataLen,dtype = type([self.labels[0]]))
		dicLabel = {}
		for i in xrange(dataLen):
			tag = feature[i,:]
			label = self.predictOne(tag)
			pLabel[i] = label

		return pLabel


	def score(self,features,yt):
		yt = np.asarray(yt)
		ytp = self.predict(features)
		acc = (1 - (1. / len(ytp) * sum( yt != ytp )))
		print 'prediction accuracy: %.4f' % acc





