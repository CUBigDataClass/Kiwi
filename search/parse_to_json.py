import glob
import pysolr
import json
from bs4 import BeautifulSoup

documents = []
filelist = glob.glob('../data/product_data/products/*')
for filename in filelist:
    f       = open(filename, 'r')
    soup    = BeautifulSoup(f.read())
    products = soup.findAll('product')
    for product in products:
        sku  = int(product.find('sku').text)
        name = product.find('name').text
        ld   = product.find('longdescription')
        sd   = product.find('shortdescription')
        if ld:
            ld = ld.text
        if sd:
            sd = sd.text
        #ADD CATEGORY!
        document = {"id":sku,
                    "name":name,
                    "long_description":ld,
                    "short_description":sd}
        documents.append(document)

with open('data.txt', 'w') as outfile:
  json.dump(documents, outfile, sort_keys=True, indent=4, separators=(',', ': '))

#solr = pysolr.Solr('http://localhost:8983/solr/', timeout=10)
#solr.delete(q='*:*')
#solr.add(documents)

#nb_results = ('id:7705281' OR 'id:9412619') #figure out how to do an or filter
#results = solr.search("text:'iPod tuner'", fq=nb_results, rows=10)
#for result in results:
        #print result['id']
        #print result['name']
        #print result
