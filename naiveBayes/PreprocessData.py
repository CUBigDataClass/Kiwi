import numpy as np
import pandas as pd                                                         #import pandas
import re

####################
# print the info of data
####################
def printInfo(train):
	totalnum = len(train)
	print "Number of Rows: " + str(len(train))                                  #prints the total number of rows
	print "Number of Unique Users: " + str(len(train['user'].unique()))         #prints the number of unique users...
	print "Number of Unique SKU: " + str(len(train['sku'].unique()))
	print "Number of query: " + str(len(train['query'].unique()))
	print "Number of Unique Categories: " + str(len(train['category'].unique()))


####################
# preprocess query item
####################
def preprocessQuery(train):
##### change all the letters to upper case
	dat = train['query'].apply(lambda x: x.upper())
	dat.name = 'query'

	for ind, item in enumerate(dat):
		#litem = item.split()
		#litem.sort()
		#llitem = re.findall(r"[a-zA-Z]+|\d+", item)
		#llitem = re.findall(r"[a-zA-Z0-9]+",item)

		r = re.compile("[0-9]+[a-zA-Z]+[0-9]+|[a-zA-Z]+|[0-9]+")
		llitem = r.findall(item)
		llitem.sort()
		ix = -1
		i360 = -1
		for i, c in enumerate(llitem):
			if c == 'XBOX':
				ix = i
			if c == '360':
				i360 = i

		indlist = []
		indlist = [i for i in xrange(len(llitem)) if i != ix and i != i360]

		ss = ' '.join(llitem[i] for i in indlist)
		if (i360 != -1):
			ss = ss + ' XBOX' + ' 360'
		elif (ix != -1):
			ss = ss + ' XBOX'
		#ss = ' '.join(ss,)
		#print ss
		dat[ind] = ss
	return dat

####################
# classify x wrt values
####################
def nClassify(x,values):
    if x < values[0]:
        return 0
    else:    
        for i in range(1,len(values) - 1,1):
            if (x < values[i+1] and x > values[i]):
                return i
        return (len(values)-1)

####################
# divide the data into n equal intervals
####################
def nDivide(dat,n):
    vs = np.zeros(n-1)
    perc = np.linspace(0,1,n+1)
    perc = perc[1:-1]
    for i in range(len(perc)):
        vs[i] = dat.quantile(perc[i])
    datD = dat.apply(nClassify,values=vs)         
    return (datD,vs)
    
####################
# binning the data into numbins bins
####################
def binDivide(dat,numbins):
	bins = np.linspace(dat.min(),dat.max(),numbins+1)
	perc = bins[1:]
	datD = dat.apply(nClassify, values = perc)         
	return (datD,perc)

####################
# preprocess time
####################
def preprocessTime(train):
	datClickT = pd.to_datetime(train['click_time'])
	datQueryT = pd.to_datetime(train['query_time'])
	datDiff = (datClickT - datQueryT).apply(lambda x: x / np.timedelta64(1, 's'))  # change date time to float
	#datDiff = datDiff / 300.0   # normalize to 1
	numbins = 100
	#(datTime,vs) = nDivide(datDiff,numbins) # divide into equal bins
	(datTime,vs) = binDivide(datDiff,numbins)
	datTime.name = 'time'
	return datTime

