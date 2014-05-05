#/usr/bin/python
##################################################
## all the good packages
import numpy as np
import pymongo
from pymongo import MongoClient
from pymongo import Connection
from bson.son import SON
import sys
from nltk.stem import WordNetLemmatizer
import re
import itertools
import cPickle as pickle
from sklearn import naive_bayes
from bson.binary import Binary
import timeit
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter, attrgetter
from scipy import sparse
import math
from sklearn.feature_extraction.text import TfidfVectorizer

from Best_Buy_MAP5 import *
import os
import errno
##################################################
def make_sure_path_exists(path):
	path = os.getcwd()+'/'+path
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

####################
## tokenize words
####################
wnl = WordNetLemmatizer()
lemmatize = wnl.lemmatize
def ptWords(words):
	try:
		words = re.sub(r'[^\w\s]','', words.lower())
	except AttributeError:
		words = re.sub(r'[^\w\s]','', str(words).lower())
	r = re.compile("[a-zA-Z]+|[0-9]+")
	litem = r.findall(words)
	litem = [lemmatize(item) for item in litem]
	return litem

def tokenizeWords(words):
	r = re.compile("[a-zA-Z]+|[0-9]+")
	litem = r.findall(words)
	litem = [lemmatize(item) for item in litem]
	return litem

####################
## preprocess words
####################
def preprocessWords(words):
	try:
		words = re.sub(r'[^\w\s]','', words.lower())
	except AttributeError:
		words = re.sub(r'[^\w\s]','', str(words).lower())
		
	return words

####################
## train data
####################
def trainData(collection,col4cls,clsdir):
###
###find the distinct categories in the training set
###
	catlist = collection.distinct('category')

#### find the popular keys by frequency and return a list of popular skus
### only look at sku and query term
### sort count descending and _id descending ## 1-ascending, -1 - descending
### choose frequency >= 20
### return a list of words for each sku
	for cat in catlist:
		resultdic = collection.aggregate([
		{"$match":{"category":cat}},
		{"$project":{"sku":1, "query":1}},
		{"$group":{"_id":"$sku","count":{"$sum":1},"wordlist":{"$push":"$query"}}},
		{"$sort":SON([("count",-1),("_id",-1)])},
		{"$match":{"count":{"$gte":20}}},
		{"$project":{"_id":0,"sku":"$_id","word_list":"$wordlist"}}
		])

#####
## if the list of words are too short, return the most popular 10 skus
#####
		if resultdic['result'] == [] or len( resultdic['result']) < 10:
			resultdic = collection.aggregate([
			{"$match":{"category":cat}},
			{"$project":{"sku":1,"query":1}},
			{"$group":{"_id":"$sku","count":{"$sum":1},"wordlist":{"$push":"$query"}}},
			{"$sort":SON([("count",-1),("_id",-1)])},
			{"$limit":10}, ## limit the number of input documents
			{"$project":{"_id":0,"sku":"$_id","word_list":"$wordlist"}}
			])

		sku_wordlist = resultdic['result']
		resultdic = {}

		## compute number of samples
		num_samples = 0
		total_wordlist=[]
		extend=total_wordlist.extend
		y = []
		for item in sku_wordlist:
			num_samples += len(item['word_list'])
			extend(item['word_list'])
			y.extend([item['sku']] * len(item['word_list']))

		y = np.array(y)
		
#####
## compute td-idf on the words and their bigrams
#####
		#vec = CountVectorizer(tokenizer=preprocessWords,binary=False,lowercase=False,
	#							max_features=10000)
#		vec = TfidfVectorizer(tokenizer=ptWords,
#							binary=False,lowercase=False,
#					ngram_range=(1, 2),max_features=10000)
		vec = TfidfVectorizer(preprocessor =preprocessWords,
							tokenizer=tokenizeWords,
							binary=False,lowercase=False,
					ngram_range=(1, 2),max_features=10000)
		data = sparse.csr_matrix(vec.fit_transform(total_wordlist))
		total_wordlist=[]
###
###classification
###
		cls = naive_bayes.MultinomialNB(alpha=0.01)
		cls.fit(data,y)

###
# output vectorizor and classifier to file
# and store the filename and category in mongo
###
		clsdic = {}
		clsdic['countvectorizor'] = vec
		clsdic['cls'] = cls
		filename = clsdir+cat+".txt"
		try:
			with open(filename,'wb') as w:
				pickle.dump(clsdic,w, protocol=2)
		except:
			print "cannot open file!"
			raise IOError

		outdic = {}
		outdic['_id'] = cat
		outdic['cls'] = filename
##### store id and filename to mongodb
		col4cls.save(outdic)
		print cat
		outdic = {}

##############################
## test data
##############################
def testData(col4cls,coltest):
	ybest_all=[]
	append = ybest_all.append
	for doc in coltest.find():
		cat = doc['category']
		words = doc['query']
		
		clsout= col4cls.find_one({"_id":cat})
		#cls = pickle.loads(clsout['cls'])
		try:
			with open(clsout['cls'],'rb') as r:
				cls = pickle.load(r)
		except IOError:
			print "%s doesn't exist!" % clsout['cls']
			raise IOError
			
		vec = cls['countvectorizor']
		fset = vec.transform([words])

		yclasses = cls['cls'].classes_
		yall = cls['cls'].predict_log_proba(fset)
		ysort = np.argsort(-yall)

		if len(yclasses) == 1:
			yout = ysort[:,:1] ## only one class	
		elif len(yclasses) < 5:
			yout = ysort
		else:
			yout = ysort[:,:5]
		
		ybest = yclasses[yout]
		#ybest = ybest.flatten().tolist()
		ybest = tuple(ybest.flatten()) 
		append(ybest)
	return ybest_all

def getTrueSKU(coltest):
	skulist = []
	append = skulist.append
	for doc in coltest.find():
		append(doc['sku'])
	
	return skulist

####################
## main function
####################
def main():
	start = timeit.default_timer()

	client = MongoClient()
	connection = Connection()

	db = client['bigdata']

	#clsinfo = "clsinfo" ## unigram and bigram without google refine
	#clsinfo = "clsinfo_googlerefine" ## unigram and bigram
	train_name = "train_cv"
	#test_name = "test_part"
	test_name = "test_cv"

	clsinfo = "clsinfo_cv"
	clsdir = 'cls_cv/'
	make_sure_path_exists(clsdir)

	col4cls = db[clsinfo] ## collection for storing info for classifiers
#######################
##### training data
#######################

	col4train = db[train_name]
	#collection = db['train_googlerefine']
	#clsdir = 'cls/'
	trainData(col4train,col4cls,clsdir)

######################
#### testing data
######################
##### test ####
	col4test = db[test_name]	 ## collection for test data
	ybest = testData(col4cls,col4test)
	#ybest = testData(col4cls,col4train)

	#ytrue = getTrueSKU(col4train)
	ytrue = getTrueSKU(col4test)
#######################
##### check MAP
#######################
#	for i in xrange(len(ytrue)):
#		yb = ybest[i]
#		yt = ytrue[i]
#		print yb,yt,Mean_Average_Precision([yt],[yb]),Max_Score([yt],[yb])
#
	mapscore = Mean_Average_Precision(ytrue,ybest)
	maxscore = Max_Score(ytrue,ybest)
	print "Total MAP score is", mapscore
	print "Total MAX score is", maxscore
#######################
##### time
#######################
	stop = timeit.default_timer()

	print 'time is', stop - start

##################################################
if __name__=='__main__':
	main()
##################################################

