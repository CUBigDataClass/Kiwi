
# coding: utf-8

# In[1]:

import json
import pysolr

#read in data
json_data=open('product_description.json')
documents = json.load(json_data)
json_data.close()


# In[3]:

#connect to Solr server
solr = pysolr.Solr('http://localhost:8983/solr/', timeout=10)
solr.delete(q='*:*')
solr.add(documents)


# In[6]:

#rough idea of how to query

#nb_results = ('id:7705281' OR 'id:9412619') #figure out how to do an or filter
results = solr.search("text:'HP tablet'", rows=10)
for result in results:


# In[ ]:



