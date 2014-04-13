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
	def __init__(self,alpha=0.0):
		self.freqdic=[]
		self.alpha = alpha
		self.fn = 0
		self.cn = 0
		self.isDataFit = False
		self.unkfeature = '<UNK_F>' # for unknown words
		self.unklabel = '<UNK_L>' # for unknown words

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
		print fnum
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
		self.datanum, self.fn = features.shape
		
		####################
		## check the size of the data
		####################
		if (self.datanum != len(labels)):
			print "length of labels not equal to number of data"
			sys.exit()

		self.labels, indices = np.unique(labels,return_inverse=True)
		self.cn = len(self.labels)

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

	def predict(self,feature):
		if self.isDataFit:
			return 1.0

		else:
			print "Classes not fit, cannot predict!"
			sys.exit()

	def score(self,features,yt):
		ytp = self.predict(features)
		acc = (1 - (1. / len(ytp) * sum( yt != ytp )))
		print 'prediction accuracy: %.4f' % acc





