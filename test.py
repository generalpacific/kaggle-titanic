import csv as csv
import numpy as np


#open file

train = csv.reader(open('train.csv','rb'))
header = train.next();

#print header

data=[]
for row in train:
	data.append(row)
data = np.array(data)


#calculate statistics number of survived and total number of passengers
numberOfPassengers = np.size(data[0::,1].astype(np.float))
#print("Number of passengers ",numberOfPassengers)

numOfSurvived = np.sum(data[0::,1].astype(np.float))
#print("Number of survived passengers ",numOfSurvived)

test = csv.reader(open('test.csv','rb'))
writeFile = csv.writer(open("genderbased.csv","wb"))

for row in test:
	if row[3] == 'female':
		row.insert(0,'1')
		writeFile.writerow(row)
	else:
		row.insert(0,'0')
		writeFile.writerow(row)
