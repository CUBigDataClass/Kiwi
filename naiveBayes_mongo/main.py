#/usr/bin/python
##################################################
## all the good packages
import pandas as pd
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
from sklearn.decomposition import FastICA 
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter, attrgetter
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import math
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from Best_Buy_MAP5 import *

##################################################
####################
## tokenize words
####################
def preprocessWords(words):
	wnl = WordNetLemmatizer()
	lemmatize = wnl.lemmatize

	try:
		words = re.sub(r'[^\w\s]','', words.lower())
	except AttributeError:
		words = re.sub(r'[^\w\s]','', str(words).lower())
		
	r = re.compile("[a-zA-Z]+|[0-9]+")
	litem = r.findall(words)
	litem = [lemmatize(item) for item in litem]
	return litem

####################
## main function
####################
def main():
	start = timeit.default_timer()

	client = MongoClient()
	connection = Connection()

	db = client['bigdata']
	collection = db['train']
	#clsinfo = "clsinfo_subdir"
	clsinfo = "clsinfo"
	col4cls = db[clsinfo]
#
####################
## find the distinct categories in the training set
####################
	catlist = collection.distinct('category')

	for cat in catlist:
#### find the popular keys by frequency and return a list of popular skus
		resultdic = collection.aggregate([
		{"$match":{"category":cat}},
### only look at sku and query term
		{"$project":{"sku":1, "query":1}},
		{"$group":{"_id":"$sku","count":{"$sum":1},"wordlist":{"$push":"$query"}}},
### sort count descending and _id descending ## 1-ascending, -1 - descending
		{"$sort":SON([("count",-1),("_id",-1)])},
### choose frequency >= 20
	{"$match":{"count":{"$gte":20}}},
##		{"$limit":10}, ## limit the number of input documents
		### return a list of words for each sku
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
		vec = TfidfVectorizer(tokenizer=preprocessWords,binary=False,lowercase=False,
					ngram_range=(1, 2),max_features=10000)
		data = sparse.csr_matrix(vec.fit_transform(total_wordlist))
		total_wordlist=[]
####################
### classification
####################
		cls = naive_bayes.MultinomialNB(alpha=0.01)
		cls.fit(data,y)
#		print cls.score(data,y)
		clsdic = {}
###################
# output vectorizor and classifier to file
# and store the filename and category in mongo
###################
		clsdic['countvectorizor'] = vec
		clsdic['cls'] = cls
		filename = "cls/"+cat+".txt"
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
		outdic = {}
		print cat
		#col4cls.insert(outdic)
######################
#### testing data
######################
##### test ####
	test_name = "test_part"
	coltest = db[test_name]
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
		yall = cls['cls'].predict_proba(fset)
		ysort = np.argsort(-yall)

		try:
			yout = ysort[:,:5]
		except IndexError:
			yout = ysort[:,:len(yclasses)] ## only one class	
		
		ybest = yclasses[yout]
		ybest = tuple(ybest.flatten().tolist()) 
		print ybest,doc['sku'],Mean_Average_Precision([doc['sku']],[ybest]),Max_Score([doc['sku']],[ybest])

	stop = timeit.default_timer()

	print 'time is', stop - start

##################################################
if __name__=='__main__':
	main()
##################################################

