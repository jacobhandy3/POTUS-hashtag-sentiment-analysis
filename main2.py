from IPython import embed
import numpy as np 
import pickle
import re
import csv
import utils
from twython import Twython
import random
import json
import tensorflow as tf
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from tensorflow.python.framework import ops
import pandas as pd
ops.reset_default_graph()

POSITIVE_WORDS_FILE = 'positive-words.txt'
NEGATIVE_WORDS_FILE = 'negative-words.txt'
API_KEYS_FILE = 'APIkeys.txt'

def file_to_wordset(filename):
    ''' Converts a file with a word per line to a Python set '''
    words = []
    with open(filename, 'r') as f:
        for line in f:
            words.append(line.strip())
    return set(words)

#split data between train and test(67% to 33%)
def splitDatasets(data, splitRatio):
    trainSize = int(len(data) * splitRatio)
    trainSet = []
    copy = list(data)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
        return[trainSet, copy]

#classify sentiment of tweet
def classify(tweet, **params):
    postive_words = file_to_wordset(params.pop('positive_words'))
    negative_words = file_to_wordset(params.pop('negative_words'))
    posCount, negCount = 0, 0
    for word in tweet.split():
        if word in postive_words:
            posCount += 1
        elif word in negative_words:
            negCount +=1
        else:
            posCount += 1
            negCount += 1
    if negCount > posCount:
        return 0
    else:
        return 1
def findClass(tweet, classWord):
    if classWord in tweet:
        return True
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vec = dataset[i]
        if vec[-1] not in separated:
            separated[vec[-1]] = []
        separated[vec[-1]].append(vec)
    return separated

def mean(numbers):
	return float(sum(numbers))/float(len(numbers))

def stdev(numbers):
    try:
        avg = float(mean(numbers))
        variance = float(sum([pow(x-avg,2) for x in numbers]))/float(len(numbers)-1)
        return math.sqrt(float(variance))
    except:
        avg = mean(numbers)
        variance = 1
        return math.sqrt(float(variance))

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (float(correct)/float(len(testSet))) * 100.0
#Twitter api key stuff
api_keys = []
with open(API_KEYS_FILE) as api:
    for line in api:
        api_keys.append([str(l) for l in line.strip().split('\n')])

#Get Twitter credentials
APP_KEY = api_keys[0]
APP_SECRET = api_keys[1]
OAUTH_TOKEN = api_keys[2]
OAUTH_SECRET_TOKEN = api_keys[3]
#Create Twython object with creds
twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_SECRET_TOKEN)
#Create query with search term, result_type, count, & lang
Q = {'q': '#POTUS', 'result_type': 'mixed', 'count': 100, 'lang': 'en'}
# Create dictionary with desired data
i = 0
D = {'id': [], 'Trump': [], 'obama': [], '#BackfireTrump': [], '2020': [], 'mueller': [], 'impeach': [], '#Democrats': [], 'Russia': [], 'sentiment': []}
# For each status in searching tweets
present = False

# V UNCOMMENT THIS TO CHANGE THE TWEET DATA FOR UPDATED RESULTS V
# f = open('tweetData1.txt', 'w', encoding="utf-8")
# for status in twitter.search(**Q)['statuses']:
#     s = status['text'].replace(',', '')
#     s = status['text'].replace('.', '')
#     s = status['text'].replace('#', '')
#     s = status['text'].replace('?', '')
#     s = status['text'].replace(':', '')
#     s = status['text'].replace('!', '')
#     s = status['text'].replace('rt' , '')
#     s = status['text'].replace('/', '')
#     s = status['text'].lower()
#     print(s)
#     f.write(s + '\n')
# f.close()
# t = open('tweetBin.txt', 'r', encoding="utf-8")
# for line in t:
#     D['id'].append(i)
#     i += 1
#     st = line.replace('.', '')
#     st = line.replace(':', '')
#     st = line.replace('rt' , '')
#     st = line.replace('/', '')
#     st = line.lower()
#     present = findClass(st, 'trump')
#     D['Trump'].append(1) if present == True else D['Trump'].append(0)
#     present = False
#     present = findClass(st, 'obama')
#     D['obama'].append(1) if present == True else D['obama'].append(0)
#     present = False
#     present = findClass(st, 'backfiretrump')
#     D['#BackfireTrump'].append(1) if present == True else D['#BackfireTrump'].append(0)
#     present = False
#     present = findClass(st, '2020')
#     D['2020'].append(1) if present == True else D['2020'].append(0)
#     present = False
#     present = findClass(st, 'mueller')
#     D['mueller'].append(1) if present == True else D['mueller'].append(0)
#     present = False
#     present = findClass(st, 'impeach')
#     D['impeach'].append(1) if present == True else D['impeach'].append(0)
#     present = False
#     present = findClass(st, 'democrat')
#     D['#Democrats'].append(1) if present == True else D['#Democrats'].append(0)
#     present = False
#     present = findClass(st, 'russi')
#     D['Russia'].append(1) if present == True else D['Russia'].append(0)
#     present = False
#     D['sentiment'].append(classify(st, positive_words=POSITIVE_WORDS_FILE, negative_words=NEGATIVE_WORDS_FILE))
# t.close()
# with open('dict_file.csv', 'w', encoding="utf-8") as f:
#     for key, value in D.items():
#         f.write('%s:%s\n' % (key, value))
# ^ UNCOMMENT THIS TO CHANGE THE TWEET DATA FOR UPDATED RESULTS ^

# NB Classifer stuff
# load csv into list
with open('dict_file.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
length = len(your_list[0])
tweetList = [[] for _ in range(length)]
i = 0
while i < length:
    tweetList[i].append(int(your_list[1][i]))
    tweetList[i].append(int(your_list[2][i]))
    tweetList[i].append(int(your_list[3][i]))
    tweetList[i].append(int(your_list[4][i]))
    tweetList[i].append(int(your_list[5][i]))
    tweetList[i].append(int(your_list[6][i]))
    tweetList[i].append(int(your_list[7][i]))
    tweetList[i].append(int(your_list[8][i]))
    tweetList[i].append(int(your_list[9][i]))
    i += 1
# # Create the trainset by separating 67% of the data
trainSet, testSet = splitDatasets(tweetList, 0.67)
summaries = summarizeByClass(trainSet)
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}%'.format(accuracy))