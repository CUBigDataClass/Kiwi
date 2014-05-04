Work with databse to reduce the requirement on memory

To run:
first import csv into mongo db
1 open terminal window and type mongod
2 open another terminal window and go to the directory where you store the data for BestBuy
3 type:
	mongoimport -d bigdata -c train --type csv --file train.csv --headerline
	mongoimport -d bigdata -c test --type csv --file test.csv --headerline
it may take a while to import the data to mongodb

4 go to the naiveBayes_mongo/ and type:
	python main_mongo.py
	
