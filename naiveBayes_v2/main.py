#/usr/bin/python
##################################################
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
## all the good packages
	import pandas as pd

import numpy as np
import sys
from PreprocessData import preprocess
import timeit
from sklearn import naive_bayes
#import nltk
import collections
from collections import defaultdict
from operator import itemgetter, attrgetter

##################################################
##
## function definitions
##
####################
# print the info of data
####################
def printInfo(train):
	totalnum = len(train)
	#prints the total number of rows
	print "Number of Rows: " + str(len(train))                                  
	#prints the number of unique users...
	print "Number of Unique Users: " + str(len(train['user'].unique()))         	
	
	print "Number of Unique SKU: " + str(len(train['sku'].unique()))
	print "Number of query: " + str(len(train['query'].unique()))
	print "Number of Unique Categories: " + str(len(train['category'].unique()))

####################
## group data by category 
####################
def groupByCat(dataroot):
#### return a dictionary of the grouped data by category

	## read file into pandas data frame
	train = pd.read_csv(dataroot)
#	printInfo(train)

	print "group by category"
## dictionary of grouping categories
	grouped = train.groupby('category').groups
	groupedcat= dict.fromkeys(grouped.keys())
	
	for cat in grouped:
		groupedcat[cat] = train.ix[grouped[cat]].drop('category',axis=1)
	
	return groupedcat


####################
## get the feature sets
####################
def getFeatureSet(dat,n=5):	
	## count frequency of sku

## choose the most frequent 5 skus
	skugd = dat.groupby('sku').groups
	sku_counts =  dat['sku'].value_counts()
	sku_counts.sort(ascending=False)
	sku_above = []

	if len(sku_counts) < n:
		n = len(sku_counts)
	
	for i in xrange(n):
		sku_above.append(sku_counts.index[i])

	sku_dic = dict.fromkeys(sku_above,list)

	wordlist = list()
	num_samples = 0
	for sku in sku_above:
		num_samples += len(skugd[sku])
		queries = dat['query'][skugd[sku]].values.tolist()
		sku_dic[sku] = queries
		wordlist.extend(sum(queries,[]))

	wordlist = set(wordlist)

	map_vocab = dict([item,i] for i,item in enumerate(wordlist))
	num_features = len(wordlist)
	fsets = np.zeros((num_samples,num_features),dtype=np.uint8)
	y = np.zeros(num_samples)

	j = 0
	for sku in sku_dic:
		words4sku= sku_dic[sku]
		for words in words4sku:
			y[j] = sku
			for word in words:
				try:
					fsets[j,map_vocab[word]] = 1
				except:
					continue

			j += 1
####store the information
	sku_info = dict()
	sku_info['pop_skus'] = sku_above
	sku_info['num_features'] = num_features
	sku_info['map_vocab'] = map_vocab
	#print sku_info,fsets,y

	return (sku_info,fsets,y)

def get_test_featuresets(catdat,skuinfo):
	num_features = skuinfo['num_features']
	map_vocab = skuinfo['map_vocab']
	num_samples = len(catdat)

	testfsets = np.zeros((num_samples,num_features),dtype=np.uint8)
	for j in xrange(num_samples):
		words = catdat['query'].iloc[j]
		for word in words:
			try:
				testfsets[j,map_vocab[word]] = 1
			except:
				continue
	return testfsets
##################################################
## main function
##################################################
def main():
	start = timeit.default_timer()
	print "read train data"
	dataroot = "../data/train_part.csv"
	gcat_dic = groupByCat(dataroot)

	cat_list = gcat_dic.keys()
####################
## preprocess the data
####################

####################
## feature selections
####################
	cat_info = dict.fromkeys(cat_list,{})
	for cat in cat_list:
		#print "preprocessing data"
		catdat = preprocess(gcat_dic[cat])

		#print "feature selections"
		sku_info,fset,skus = getFeatureSet(catdat)
		cat_info[cat]['sku_info'] = sku_info
	
##### method 1 #####
## choose the most frequent 5 skus

##### method 2 #####
## or choose the skus with frequency > n, n is user specified

## the final feature set is a matrix X, (n_samples, n_skus)
##	and a column of sku, (n_skus)
####################
## train NB classifiers
####################
		#print "training data"
		cls = naive_bayes.MultinomialNB(alpha=0.5)
		cls.fit(fset,skus)
		cat_info[cat]['cls'] = cls
#
### release the memory
#	gcat_dic = dict()
####################
## predict
####################
## preprocess test data
	print "read test data"
	dataroot = "../data/test_part.csv"
	testdat = groupByCat(dataroot)

## predict by nb_dic
	for cat in testdat:
		try:
			catdat = preprocess(testdat[cat])
			catdic = cat_info[cat]["sku_info"]
			testfsets = get_test_featuresets(catdat,catdic)
			cls = cat_info[cat]['cls']
			yt = cls.predict(testfsets)
			print yt
		except KeyError:
			print "Category %s is unseen!" % str(cat)
			sys.exit()
		

####################
## compute elapsed CPU time
####################
	stop = timeit.default_timer()

	print 'time is', stop - start
	
##################################################
if __name__=='__main__':
	main()
##################################################



