import glob
import json
from bs4 import BeautifulSoup

documents = []                                          # fill up list with dictionary
filelist = glob.glob('../data/product_data/products/*') # grab all files in directory
for filename in filelist:                               # loop over files
    f       = open(filename, 'r')
    soup    = BeautifulSoup(f.read())                   # parse file html
    products = soup.findAll('product')                  # list of all product fields
    for product in products:                            # loop over products
        sku  = int(product.find('sku').text)            # get sku
        name = product.find('name').text                # get name
        ld   = product.find('longdescription')          # get long description
        sd   = product.find('shortdescription')         # get short description
        catL = product.find('categorypath')             # grab structure containing category info
        cat_id   = []                                   # list for category id
        cat_name = []                                   # list for category name
        if catL:                                        
            cat_html = catL.findAll('category')         # loop over categories
            for cat in cat_html:
                cat_id.append(cat.find('id').text)      # add category id to list
                cat_name.append(cat.find('name').text)  # add category name to list
        if ld:
            ld = ld.text
        if sd:
            sd = sd.text
        document = {"id":sku,                           # make dictionary of fields
                    "cat_id":cat_id,
                    "cat_name":cat_name,
                    "name":name,
                    "long_description":ld,
                    "short_description":sd}
        documents.append(document)                      # add dictionary to list

with open('data.txt', 'w') as outfile:                  # print list of dictionary to json
  json.dump(documents, outfile, sort_keys=True, indent=4, separators=(',', ': '))
