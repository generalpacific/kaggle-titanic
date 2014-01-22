
# This code will compare different algorithms for the dataset
# To add a new algorithm use the preprocessed data and write a new method and call the method in the end.

import re	
import csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Add time along with the log
def log(logname,string):
        print str(datetime.now()) + "\t"  + logname + "\t" + string

##################################################
# METHODS FOR PREPROCESSING THE DATA
##################################################

# Convert the gender
def convertGender(gender):
	if gender == 'female':
		gender = 0
	if gender == 'male':
		gender = 1
	return gender

# Convert the embarked field
def convertEmbarked(embarked):
	if embarked == 'C':
		embarked = 0
	if embarked == 'Q':
		embarked = 1
	if embarked == 'S':
		embarked = 2
	else:
		embarked = '2'
	return embarked

# return title
def getTitle(name):
	for word in name.split():
		if word.endswith('.'):
			title=word
			break
	return title

# convert title to hash 
# TODO need to improve
def getTitleHash(title,gender):
	has = ord(title[0]) + len(title) + int(gender)
	return has

# returns one if the passenger had a family
def getFamily(sibsp,parch):
	if int(sibsp) + int(parch) > 0:
		family = 1
	else:
		family = 0
	return family

# Pull out the dept from the ticket number
def getTicketCode(ticket):
    deptName = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", ticket)
    if len(deptName) == 0:
        deptName = 'none'
    deptCode = ord(deptName[0]) + len(deptName)
    return deptCode

# Return the same fare if it is non-empty.
# Else return the average fro the given 'ticket class'
# The average fare is already calculated in a spreadsheet
def getFare(fare,ticketclass):
	if fare != '':
		return fare
	if ticketclass == 1:
		return '94'
	if ticketclass == 2:
		return '22'
	if ticketclass == '3':
		return '12'

##################################################
# METHODS FOR DIFFERENT ALGORITHMS
#
# Tips to add new algorithm:
#	1. Copy the following random forest code. 
#	2. Change the place holders accordingly
##################################################

def randomforest(trainfeatures,trainlabels,testfeatures):
	RandomForest = RandomForestClassifier(n_estimators = 1000)
	return runalgorithm(RandomForest,trainfeatures,trainlabels,testfeatures)

def decisiontree(trainfeatures,trainlabels,testfeatures):
	tree = DecisionTreeClassifier(random_state = 1000)
	return runalgorithm(tree,trainfeatures,trainlabels,testfeatures)

def adaboost(trainfeatures,trainlabels,testfeatures):
	adaBoost = AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000),
                         algorithm="SAMME",
                         n_estimators=200)
	return runalgorithm(adaBoost,trainfeatures,trainlabels,testfeatures)

# Generic code for running any algorithm called from above algorithms
def runalgorithm(algorithm,trainfeatures,trainlabels,testfeatures):
	logname = runalgorithm.__name__
	algorithmName = algorithm.__class__.__name__
	
	log(logname,algorithmName + " Fitting train data")
        algorithm = algorithm.fit(trainfeatures,trainlabels)
	log(logname,algorithmName + " DONE Fitting train data")
	
	log(logname,algorithmName + " Scoring train data")
	scores = cross_val_score(algorithm, trainfeatures, trainlabels)
	score = scores.mean()
	score = str(score)
	log(logname,algorithmName + " Score : " + score)
	log(logname,algorithmName + " DONE Scoring train data")
	
	log(logname,algorithmName + " Predicting test data")
	Output = algorithm.predict(testfeatures)	
	log(logname,algorithmName + " DONE Predicting test data")
	writeFile = algorithmName + ".csv"
	log(logname,algorithmName + " Writing results to " + writeFile)
	np.savetxt(writeFile,Output,delimiter=",algorithmName + " ,fmt="%s")
	log(logname,algorithmName + " DONE Writing results to " + writeFile)
	return score

##################################################
# MAIN METHOD
##################################################
if __name__ == '__main__':	
	
	logname = "__main__"

	log(logname,"Reading Train Data")
	
	train = csv.reader(open('train.csv','rb'))
	header = train.next()
	
	######READING TRAIN DATA################	
	train_data=[]
	for row in train:
	        train_data.append(row)
	
	train_data = np.array(train_data)
	
	log(logname,"DONE Reading Train Data")
	
	log(logname,"Preprocessing Train Data")
	# replace categorical attributes
	for row in train_data:
		
		row[4] = convertGender(row[4])
		title = getTitle(row[3])
		row[3] = getTitleHash(title,row[4])
		row[6] = getFamily(row[6],row[7])
		row[8] = getTicketCode(row[8])
		row[9] = getFare(row[9],row[2])
		row[11] = convertEmbarked(row[11])
		

	trainfeatures = train_data[0::,[2,3,4,6,8,11]]
	trainlabels = train_data[0::,1]
	log(logname,"DONE Preprocessing Train Data")


	######READING TEST DATA################	
	log(logname,"Reading Test Data")
	test = csv.reader(open('test.csv','rb'))
	header = test.next()
	
	test_data=[]
	for row in test:
	        test_data.append(row)
	test_data = np.array(test_data)
	log(logname,"DONE Reading Test Data")
	
	# replace categorical attributes
	log(logname,"Preprocessing Test Data")
	for row in test_data:
		
		row[3] = convertGender(row[3])
		title = getTitle(row[2])
		row[2] = getTitleHash(title,row[3])
		row[5] = getFamily(row[5],row[6])
		row[7] = getTicketCode(row[7])
		row[8] = getFare(row[8],row[1])
		row[10] = convertEmbarked(row[10])
		

	testfeatures = test_data[0::,[1,2,3,5,7,10]]
	log(logname,"DONE Preprocessing Test Data")
	
	####################### TRAIN AND TEST ##########################

	scores = {}

	log(logname,"Calling Random Forest")
	score = randomforest(trainfeatures,trainlabels,testfeatures)
	scores['Random Forest'] = score
	log(logname,"DONE WITH Random Forest")

	log(logname,"Calling AdaBoost")
	score = adaboost(trainfeatures,trainlabels,testfeatures)
	scores['AdaBoost'] = score
	log(logname,"DONE WITH AdaBoost")
	
	log(logname,"Calling Decision Tree")
	score = decisiontree(trainfeatures,trainlabels,testfeatures)
	scores['Decision Tree'] = score
	log(logname,"DONE WITH Decision Tree")

	print "\nSCORES\n"
	for k, v in scores.iteritems():
		print k + "\t" + v
		
