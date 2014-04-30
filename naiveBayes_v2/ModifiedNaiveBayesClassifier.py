import nltk
from collections import defaultdict

from nltk.probability import *


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
					#print probdist
					feature_probdist[label,fname] = probdist
			
			return ModNaiveBayesClassifier(label_probdist, feature_probdist)

