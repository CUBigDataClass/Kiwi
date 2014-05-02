import jellyfish
import string
import re
from collections import defaultdict

class Query_Summary(object):
  def __init__(self, words):
    self.word_list = set(words)
    self.freq = {}
    for w in words:
      if w in self.freq:
        self.freq[w] += 1
      else:
        self.freq[w] = 1

  def make_gold_list(self, min_freq):
    self.gold_list = []
    for w in self.word_list:
      if self.freq[w] >= min_freq:
        self.gold_list.append(w)
  
  def correct_gold_list(self, max_distance):
    self.edit_dict = {}
    check = [x for x in self.word_list if x not in self.gold_list]
    for w in check:
      edit = ''
      for c in self.gold_list:
        diff = jellyfish.levenshtein_distance(w,c)
        if diff <= max_distance:
          if not edit:
            edit = c
            min_diff = diff
          elif diff < min_diff:
            edit = c
            min_diff = diff
      if edit:
        self.edit_dict[w] = edit

def Spell_Check(check_word, known_word, max_distance):
  diff = jellyfish.levenshtein_distance(check_word, known_word)
  print diff
  if diff <= max_distance:
    return known_word
  else:
    return check_word


import pandas as pd
import sys
train = pd.read_csv("../data/train.csv")

#clean up query
train['query'] = train['query'].apply(lambda x: x.upper(),1)
train['query'] = train['query'].apply(lambda x: x.translate(string.maketrans("",""), string.punctuation),1)
train['query'] = train['query'].apply(lambda x: re.sub('(\s+)(A|AN|AND|THE|OF)(\s+)', '\1\3', x),1)
train['query'] = train['query'].apply(lambda x: re.sub(' +',' ',x),1)

#creates a dictionary with key = sku and value is list of all words that come up in all queries for that sku
big_dict = defaultdict(list)
for i in range(len(train)):
  sku = train.ix[i,'sku']
  txt = train.ix[i,'query']
  for w in txt.split():
    big_dict[sku].append(w)

#creates a dictionary of Query_Summary objects
dict2 = {}
for sku in big_dict:
  dict2[sku] = Query_Summary(big_dict[sku])
  dict2[sku].make_gold_list(2)    #makes a golden list based on a min frequency of 2
print sys.getsizeof(dict2)
