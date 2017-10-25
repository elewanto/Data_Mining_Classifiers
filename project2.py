# Eric Lewantowicz
# CSE 5243 Data Mining
# Project 2
# 10/14/2017
# File Dependences: amazon_cells_labelled.txt, imdb_labelled.txt, yelp_labelled.txt
# Module Dependences: nltk
# Execution Instruction: $ python ./project2.py

import fileinput
import re			# regular expressions
import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk import PorterStemmer
import random		# shuffle used from random module for randomizing data indices
from random import shuffle
import time
import math
from math import log

# define class objects
class Sentence:
	def __init__(self, sentiment):
		self.sentiment = sentiment
		self.upperWords = []

class BinTree():
	def __init__(self, value, sentiment):
		self.left = None
		self.right = None
		self.value = value
		self.sentiment = sentiment

	def getLeft(self):
		return self.left
	def getRight(self):
		return self.right
	def setValue(self, value):
		self.value = value
	def setSentiment(self, sentiment):
		self.sentiment = sentiment
	def getValue(self):
		return self.value
	def getSentiment(self):
		return self.sentiment
	def addRight(self, nodeVal):
		self.right = BinTree(nodeVal)
	def addLeft(self, nodeVal):
		self.left = BinTree(nodeVal)
	def printTree(self, level):
		print 'L', level, ' ', self.value, ': ',
		level += 1
		if  self.left != None:
			self.left.printTree(level)
		else:
			print 'leaf',
		if self.right != None:
			self.right.printTree(level)
		else:
			print 'leaf',


# global variables
sumOnes = 0
sumZeros = 0
numLemmatized = 0
numStemmed = 0
totalWords = 0
totalLemmaWords = 0
totalStemWords = 0
numTreeNodes = 0

printStem = 0
printLemma = 0

words = {}				# dictionary with word keys and list values, where the list is index of sentences and word occurrences
wordsNltkLemma = {}		# second dictionary of words processed using NLTK Lemmatization
wordsNltkStem = {}		# third dictionary of words proecessed using NLTK Stemming
sentences = []			# sentiment value for each sentence; 0: negative, 1: positive
sentWords = {}			# dictionary of words in each indexed sentence; key is index, value is list of sentence words
sentWordsStem = {}
sentWordsLemma = {}

# the sets contain the sentence indices from the global dataset
trainSet = []
validSet = []
testSet = []

trainNum = 1800			# 60% training
testNum = 600			# 20% testing
validNum = 600			# 20% validation
totalSentences = 3000

# process each line of text from input files
def processLine(line, sentIndex):
	global sumOnes
	global sumZeros
	global words
	global totalWords

	line = line.strip()											# remove leading and trailing whitespace from line
	if line.endswith('0'):
		sentences[sentIndex].sentiment = 0
		sumZeros += 1
	elif line.endswith('1'):
		sentences[sentIndex].sentiment = 1
		sumOnes += 1
	else:
		print 'Error finding sentiment sentence #', sentIndex
	line = line[:len(line) - 1]									# slice sentiment 0,1 digit from line input
	line = line.strip()											# strip trailing tab from line
	count = words.get('!')
	count[sentIndex] = line.count('!')
	words['!'] = count
	if count[sentIndex] > 0:
		wordList = sentWords.get(sentIndex)
		wordList.append('!')
		sentWords[sentIndex] = wordList	
	count = words.get('?')
	count[sentIndex] = line.count('?')
	words['?'] = count
	if count[sentIndex] > 0:
		wordList = sentWords.get(sentIndex)
		wordList.append('?')
		sentWords[sentIndex] = wordList	
	count = words.get('$')
	count[sentIndex] = line.count('$')
	words['$'] = count
	if count[sentIndex] > 0:
		wordList = sentWords.get(sentIndex)
		wordList.append('$')
		sentWords[sentIndex] = wordList	
	count = wordsNltkLemma.get('!')
	count[sentIndex] = line.count('!')
	wordsNltkLemma['!'] = count
	if count[sentIndex] > 0:
		wordList= sentWordsLemma.get(sentIndex)
		wordList.append('!')
		sentWordsLemma[sentIndex] = wordList	
	count = wordsNltkLemma.get('?')
	count[sentIndex] = line.count('?')
	wordsNltkLemma['?'] = count
	if count[sentIndex] > 0:
		wordList= sentWordsLemma.get(sentIndex)
		wordList.append('?')
		sentWordsLemma[sentIndex] = wordList	
	count = wordsNltkLemma.get('$')
	count[sentIndex] = line.count('$')
	wordsNltkLemma['$'] = count
	if count[sentIndex] > 0:
		wordList= sentWordsLemma.get(sentIndex)
		wordList.append('$')
		sentWordsLemma[sentIndex] = wordList	
	count = wordsNltkStem.get('!')
	count[sentIndex] = line.count('!')
	wordsNltkStem['!'] = count
	if count[sentIndex] > 0:
		wordList= sentWordsStem.get(sentIndex)
		wordList.append('!')
		sentWordsStem[sentIndex] = wordList	
	count = wordsNltkStem.get('?')
	count[sentIndex] = line.count('?')
	wordsNltkStem['?'] = count
	if count[sentIndex] > 0:
		wordList= sentWordsStem.get(sentIndex)
		wordList.append('?')
		sentWordsStem[sentIndex] = wordList	
	count = wordsNltkStem.get('$')
	count[sentIndex] = line.count('$')
	wordsNltkStem['$'] = count		
	if count[sentIndex] > 0:
		wordList= sentWordsStem.get(sentIndex)
		wordList.append('$')	
		sentWordsStem[sentIndex] = wordList		
	regex = re.compile('[|\[".,!():;*?#/\\]()+]')				# replace punctuation with spaces except n't and - adjacent to words after saving punctuation counts
	if regex.search(line):
		line = regex.sub(' ', line)
	regex = re.compile(r'&')									# replace '&' with 'and' in line
	if regex.search(line):
		line = regex.sub('and', line)

	processNltkLemma(line, sentIndex)							# function to process line using NLTK tokenizer and lemmatization
	processNltkStem(line, sentIndex)							# function to process line using NLTK tokenizer and stemming

	wordList = line.split()										# split sentence into list of words
	for w in wordList:
		totalWords += 1
		wp = processWord(w, sentIndex)							# process each word in list from the sentence
		wList = sentWords.get(sentIndex)
		wList.append(wp)
		sentWords[sentIndex] = wList
		if wp not in words:										# add new word to dictionary if word doesn't already exist
			wordCount = initializeList()
			wordCount[sentIndex] = 1
			words[wp] = wordCount
		else:													# update count for existing word if word already exists in dictionary
			wordCount = words.get(wp)
			wordCount[sentIndex] += 1
			words[wp] = wordCount

# function creates second dictionary of words using NLTK tokenizing and lemmatization
def processNltkLemma(line, sentIndex):
	global numLemmatized
	global totalLemmaWords
	global printLemma

	#tokens = nltk.word_tokenize(line)
	tokens = nltk.word_tokenize(line.decode('utf-8'))
	for t in tokens:
		totalLemmaWords += 1
		wt = t.lower()
		lt = WordNetLemmatizer().lemmatize(wt)					# use NLTK lemmatizer to lemmatize words
		if wt != lt:
			numLemmatized += 1									# increment if lemmatized word different from unlemmatized word
		lt = lt.encode('utf-8')
		wList = sentWordsLemma.get(sentIndex)
		wList.append(lt)
		sentWordsLemma[sentIndex] = wList
		if lt not in wordsNltkLemma:
			wordCount = initializeList()
			wordCount[sentIndex] = 1
			wordsNltkLemma[lt] = wordCount
		else:
			wordCount = wordsNltkLemma.get(lt)
			wordCount[sentIndex] += 1
			wordsNltkLemma[lt] = wordCount

# function creates second dictionary of words using NLTK tokenizing and stemming
def processNltkStem(line, sentIndex):
	global numStemmed
	global totalStemWords
	global printStem

	#tokens = nltk.word_tokenize(line)
	tokens = nltk.word_tokenize(line.decode('utf-8'))
	for t in tokens:
		totalStemWords += 1
		wt = t.lower()											# convert uppercase letters to lowercase
		lt = PorterStemmer().stem(wt)							# use NLTK Porter Stemmer to stem words
		if wt != lt:
			numStemmed += 1
		lt = lt.encode('utf-8')
		wList = sentWordsStem.get(sentIndex)
		wList.append(lt)
		sentWordsStem[sentIndex] = wList
		if lt not in wordsNltkStem:
			wordCount = initializeList()
			wordCount[sentIndex] = 1
			wordsNltkStem[lt] = wordCount
		else:
			wordCount = wordsNltkStem.get(lt)
			wordCount[sentIndex] += 1
			wordsNltkStem[lt] = wordCount


def processWord(word, sentIndex):
	if word.isupper():											# track and store words with all uppercase letters as special cases
		if word != 'I':
			count = words.get('UPPERCASE')
			count[sentIndex] += 1
			words['UPPERCASE'] = count
			count = wordsNltkStem.get('UPPERCASE')
			count[sentIndex] += 1
			wordsNltkStem['UPPERCASE'] = count
			count = wordsNltkLemma.get('UPPERCASE')
			count[sentIndex] += 1
			wordsNltkLemma['UPPERCASE'] = count			
			sentences[sentIndex].upperWords.append(word)
	word = word.lower()											# convert uppercase letters to lowercase
	return word

def pruneSingles(wordsVect, sWords):
	print 'Pruning singles: '
	print 'Original word count: ', len(wordsVect)
	deleteCount = 0
	# remove words that occur only one time due to lack of comparison
	for keyWord, value1 in wordsVect.items():					# value is 1-D array with word counts for that key word
		totalWordCount = 0
		for occurs in value1:									# occurs is each array index value
			totalWordCount += occurs
		if totalWordCount == 1:
			del wordsVect[keyWord]
			deleteCount += 1
			for keyIndex, value2 in sWords.items():
				if keyWord in value2:
					#value2.remove(keyWord)
					tempList = value2
					tempList = filter(lambda a: a != keyWord, tempList)
					sWords[keyIndex]=tempList
	print 'Number of single-occurrence words deleted: ', deleteCount
	print 'After delete word count: ', len(wordsVect)
	print

# remove words that have 50% sentiment split
def pruneHighEntropy(wordsVect, sWords):
	print 'Pruning high entropy: '
	print 'Original word count: ', len(wordsVect)
	deleteCount = 0
	for keyWord, value1 in wordsVect.items():
		posSentiment = 0
		negSentiment = 0
		for i in range(0, 3000):
			if value1[i] > 0:
				if sentences[i].sentiment == 0:
					negSentiment += 1
				else:
					posSentiment += 1
		if negSentiment == posSentiment:
			del wordsVect[keyWord]
			deleteCount += 1
			for keyIndex, value2 in sWords.items():
				if keyWord in value2:
					#value2.remove(keyWord)
					tempList = value2
					tempList = filter(lambda a: a != keyWord, tempList)
					sWords[keyIndex]=tempList
			#print key, posSentiment, negSentiment, ' : ',
	print 'Number high entropy words deleted: ', deleteCount
	print 'After delete word count: ', len(wordsVect)
	print

# remove words that are non-descriptive and have high occurrence across sentences
# use modified TF-IDF algorithm
def pruneTfIdf(wordsVect, sWords):
	print 'High occurrence words, low selective words deleted: '
	deleteCount = 0
	occurThreshold = 100	# delete words above threshold (smaller value prunes more words)
	posNegThreshold = 0.4	# delete words below threshold (larger value prunes more words)
	for keyWord, value1 in wordsVect.items():
		occurTotal = 0
		occurSentences = 0
		posNegRatio = 0
		posSentiment = 0
		negSentiment = 0
		for i in range(0, 3000):
			occurTotal += value1[i]
			if value1[i] > 0:
				occurSentences += 1
				if sentences[i].sentiment == 0:
					negSentiment += 1
				else:
					posSentiment += 1
		posNegRatio = abs(posSentiment - negSentiment) / float(posSentiment + negSentiment)
		if occurTotal >= occurThreshold and posNegRatio <= posNegThreshold:
			del wordsVect[keyWord]
			deleteCount += 1
			for keyIndex, value2 in sWords.items():
				if keyWord in value2:
					#value2.remove(keyWord)
					tempList = value2
					tempList = filter(lambda a: a != keyWord, tempList)
					sWords[keyIndex]=tempList
			#print key, occurSentences, occurTotal, posSentiment, negSentiment, '%.4f'% posNegRatio
	print 'Prune occurThreshold: ', occurThreshold
	print 'Prune posNegThreshold: ', posNegThreshold
	print 'Number high occurrence words deleted: ', deleteCount
	print 'After delete word count: ', len(wordsVect)
	print


def pruneLowFreq(wordsVect, sWords):
	print 'Low frequency words pruned: '
	print 'Before delete word count: ', len(wordsVect)
	deleteCont = 0
	occurThreshold = 10	# delete words below threshold (larger value prunes more words)
	for keyWord, value1 in wordsVect.items():
		occurTotal = 0
		for i in range(0, 3000):
			occurTotal += value1[i]
		if occurTotal <= occurThreshold:
			del wordsVect[keyWord]
			for keyIndex, value2 in sWords.items():
				if keyWord in value2:
					#value2.remove(keyWord)
					tempList = value2
					tempLIst = filter(lambda a: a != keyWord, tempList)
					sWords[keyIndex]=tempList
	print 'Prune lowFreq occurThreshold: ', occurThreshold
	print 'After delete word count: ', len(wordsVect)
	print

def pruneHighFreq(wordsVect, sWords):
	print 'High frequency words pruned: '
	print 'Before delete word count: ', len(wordsVect)
	deleteCont = 0
	occurThreshold = 100	# delete words below threshold (smaller value prunes more words)
	for keyWord, value1 in wordsVect.items():
		occurTotal = 0
		for i in range(0, 3000):
			occurTotal += value1[i]
		if occurTotal >= occurThreshold:
			del wordsVect[keyWord]
			for keyIndex, value2 in sWords.items():
				if keyWord in value2:
					#value2.remove(keyWord)
					tempList = value2
					tempList = filter(lambda a: a != keyWord, tempList)
					sWords[keyIndex]=tempList
	print 'Prune highFreq occurThreshold: ', occurThreshold
	print 'After delete word count: ', len(wordsVect)
	print
		

def assignSets():
	totalIndices = []
	for i in range(0, 3000):
		totalIndices.append(i)
	shuffle(totalIndices)
	for i in range(0, trainNum):
		trainSet.append(totalIndices[i])
	for i in range(trainNum, trainNum + validNum):
		validSet.append(totalIndices[i])
	for i in range(trainNum + validNum, trainNum + validNum + testNum): 
		testSet.append(totalIndices[i])

# find nearest neighbors with highest similarity values
# use majority voting of k most similar neighbors to predict test sentence sentiment
# can be run on Validation Set or Testing Set using Training Set as referencet
# param: wordsVec: dictionary of words (either raw parsed, NLTK Stem, NLTKLemma)
# param: k: number of similar sentences used to decide majority sentiment
# param: compareSet: the set of Test Sentences of Validation Sentences
# param: compareNum: the number of sentences in the Test of Validation set
# return: correct and incorrect numbers of predictions
def kNearestNeighborClassifier(wordsVec, k, compareSet, compareNum, sWords):
	print
	predictList = []
	baseCorrectCount = 0
	baseWrongCount = 0
	correctCount = 0
	wrongCount = 0
	for testIndex in range(0, compareNum):
		similarityList = []
		testWordList = sWords.get(compareSet[testIndex])
		for trainIndex in range(0, trainNum):
			similarityValue = 0
			if trainSet[trainIndex] != compareSet[testIndex]:						# check test isn't same as training (for training on training accuracy)
				for testWord in testWordList:
					if wordsVec[testWord][trainSet[trainIndex]] > 0:
						similarityValue += 1
				similarityList.append((trainSet[trainIndex], similarityValue))		# similarity list contains tuples (training_sentence_index, similarityValue)
		similarityList.sort(key=lambda tup: tup[1], reverse=True)					# sort similarity list tuples largest similarity to smallest
		basePosVotes = 0
		baseNegVotes = 0
		posVotes = 0
		negVotes = 0
		trainingContainsNot = False
		testContainsNot = containsNot(compareSet[testIndex])
		for kIter in range(0, k):
			sentIndex = similarityList[kIter][0]									# index of training sentence
			trainingContainsNot = containsNot(sentIndex)
			trainSentiment = sentences[sentIndex].sentiment							# sentiment of training sentence
			if trainSentiment == 0:
				baseNegVotes += 1;
			else:
				basePosVotes += 1;
			if trainingContainsNot and (not testContainsNot):						# use not modifier to flip predicted sentence sentiment
				if trainSentiment == 0:
					trainSentiment = 1
				else:
					trainSentiment = 0
			if trainSentiment == 0:													# sentiment of training sentence
				negVotes += 1
			else:
				posVotes += 1
		prediction = 0
		basePrediction = 0
		if basePosVotes >= baseNegVotes:
			basePrediction = 1
		if posVotes >= negVotes:													# use majority voting to make prediction
			prediction = 1
			if testContainsNot and (not trainingContainsNot):
				prediction = 0
		else:
			prediction = 0
			if testContainsNot and (not trainingContainsNot):
				prediction = 1
		predictList.append((compareSet[testIndex], prediction))
		if sentences[compareSet[testIndex]].sentiment == prediction:
			correctCount += 1
		else:
			wrongCount += 1
		if sentences[compareSet[testIndex]].sentiment == basePrediction:
			baseCorrectCount += 1
		else:
			baseWrongCount += 1		
	return (correctCount, wrongCount, baseCorrectCount, baseWrongCount)


def containsNot(sentIndex):
	found = 0
	wasFound = False
	for key, value in words.items():
		if (key == "not" or key.find("n't") != -1) and value[sentIndex] > 0:
			found += value[sentIndex]
			wasFound = True
	return wasFound

def printSentence(index):
	for key, value in words.items():
		for i in range(0, value[index]):
			print key,

def initializeList():
	newList = []
	for i in range(0, 3000):
		newList.append(0)
	return newList



def getInfo(sentSet):
	infoVal = 0
	numZero = 0
	numOne = 0
	total = 0
	for i in sentSet:
		total += 1
		if sentences[i].sentiment == 0:
			numZero += 1
		else:
			numOne += 1
	frac1 = numZero / float(total)
	frac2 = numOne / float(total)
	if frac1 > 0.0:
		frac1 = frac1 * math.log(frac1, 2)
	if frac2 > 0.0:
		frac2 = frac2 * math.log(frac2, 2)
	infoVal = -(frac1 + frac2)
	bestSentiment = -1
	if numZero > numOne:
		bestSentiment = 0
	else:
		bestSentiment = 1
	return (infoVal, bestSentiment)

def classifierTree(tree, testIndex, sentWords):
	if tree.right == None and tree.left==None:
		return tree.sentiment
	if tree.value in sentWords[testIndex]:
		if tree.right == None:
			return tree.sentiment
		else:
			return classifierTree(tree.right, testIndex, sentWords)
	else:
		if tree.left == None:
			return tree.sentiment
		else:
			return classifierTree(tree.left, testIndex, sentWords)



def buildDecisionTree(sentSet, wordSet, sentWords, depth, maxDepth, infoValThresh):
	global numTreeNodes
	root = None
	if len(sentSet) == 0:
		return root
	if len(wordSet) == 0:
		return root
	infoVal = getInfo(sentSet)
	numTreeNodes += 1
	root = BinTree('LEAF', -1)
	if infoVal[0] <= infoValThresh or depth >= maxDepth:			# all tuples of same class, or below info gain threshold
		root.sentiment = infoVal[1]
	else:
		bestWord = wordSet[0]
		bestInfoVal = 1.0
		bestSet0 = []
		bestSet1 = []
		sentSubSet0 = []
		sentSubSet1 = []
		for word in wordSet:										# find best info gain word 
			for i in sentSet:
				wordList = sentWords.get(i)
				if word in wordList:
					sentSubSet1.append(i)
				else:
					sentSubSet0.append(i)
			numZero = len(sentSubSet0)
			numOne = len(sentSubSet1)
			total = numZero + numOne
			frac0 = numZero / float(total)
			frac1 = numOne / float(total)
			info0 = 0
			info1 = 0
			if numZero > 0:
				info0 = getInfo(sentSubSet0)[0]
			if numOne > 0:
				info1 = getInfo(sentSubSet1)[0]
			infoNeededAfter = (frac0 * info0) + (frac1 * info1)		# lower info needed is better
			if infoNeededAfter < bestInfoVal:
				bestInfoVal = infoNeededAfter
				bestWord = word
				bestSet0 = list(sentSubSet0)
				bestSet1 = list(sentSubSet1)
			sentSubSet0 = []
			sentSubSet1 = []
		wordSet.remove(bestWord)
		root.value = bestWord
		depth += 1
		root.left = buildDecisionTree(bestSet0, wordSet, sentWords, depth, maxDepth, infoValThresh)
		root.right = buildDecisionTree(bestSet1, wordSet, sentWords, depth, maxDepth, infoValThresh)
	
	return root

  
if __name__ == '__main__':
	start = time.time()
	sentIndex = 0
	nltk.download('wordnet')										# download for lemmatizer
	for i in range(0, 3000):										# initialize sentences array with -1 default sentiment
		s = Sentence(-1)			
		sentences.append(s)
		sentWords[i]=[]
		sentWordsStem[i]=[]
		sentWordsLemma[i]=[]
	files = ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
	# add important punctuation to vector dictionary
	emptyList = initializeList()
	words['!'] = emptyList
	emptyList = initializeList()  	
	words['?'] = emptyList
	emptyList = initializeList() 
	words['$'] = emptyList
	emptyList = initializeList()
	words['UPPERCASE'] = emptyList
	emptyList = initializeList()
	wordsNltkLemma['!'] = emptyList
	emptyList = initializeList()  	
	wordsNltkLemma['?'] = emptyList
	emptyList = initializeList() 
	wordsNltkLemma['$'] = emptyList
	emptyList = initializeList()
	wordsNltkLemma['UPPERCASE'] = emptyList
	emptyList = initializeList()
	wordsNltkStem['!'] = emptyList
	emptyList = initializeList()  	
	wordsNltkStem['?'] = emptyList
	emptyList = initializeList() 
	wordsNltkStem['$'] = emptyList
	emptyList = initializeList()
	wordsNltkStem['UPPERCASE'] = emptyList
	for fileName in files:
		inputFile = open(fileName)
		for line in inputFile:
			processLine(line, sentIndex)
			sentIndex += 1
		inputFile.close()

	# project 1 output
	print('\n')
	print 'Number of sentences read: ', sentIndex
	print 'Number of sentiment 1s  : ', sumOnes
	print 'Number of sentiment 0s  : ', sumZeros, '\n'
	print 'Number of unique words in matrix using manual parsing:     ', len(words)
	print 'Number of unique words in matrix using NLTK Lemmatization:   ', len(wordsNltkLemma)
	print 'Number of unique words in matrix using NLTK Porter Stemming:          ', len(wordsNltkStem)
	end = time.time()
	print 'Time to build three dictionaries from dataset: ', '%.3f' % (end - start), ' seconds'

	# assign training, validation, testing sets
	assignSets()
	
	# run tree classifier on validation set before pruning
	print 'Decision Tree classifier: '
	correctPredictions = 0
	wrongPredictions = 0
	bestLevels = 0
	levels = 5
	bestAccuracy = 0
	infoValThresh = 0.2
	bestInfoValThresh = 0.2
	while levels < 21:
		start = time.time()
		wordSet = []
		for keyWord in words:
			wordSet.append(keyWord)
		tree = buildDecisionTree(trainSet, wordSet, sentWords, 0, levels, infoValThresh)
		print 'tree built, levels: ', levels
		end = time.time()
		print 'Time to build decision tree from unpruned dataset: ', '%.3f' % (end - start), ' seconds'	

		start = time.time()
		for i in validSet:
			prediction = classifierTree(tree, i, sentWords)
			if prediction == sentences[i].sentiment:
				correctPredictions += 1
			else:
				wrongPredictions += 1
		accuracy = (correctPredictions / float(correctPredictions + wrongPredictions) * 100) 
		end = time.time()
		print 'Number tree nodes: ', numTreeNodes
		print 'Time to classify ', len(validSet), ' tuples in tree with ', (levels+1), ' levels: ', '%.2f'%  (end - start), ' seconds'
		print 'Time to classify per tuple: ', '%.7f'% ((end - start) / float(validNum)), ' seconds'
		print 'accuracy: ', '%.2f' % accuracy
		print 'infoValThresh: ', '%.2f' % infoValThresh
		print 'levels: ', (levels+1)
		if accuracy > bestAccuracy:
			print 'BEST ACCURACY'
			bestAccuracy = accuracy
			bestInfoValThresh = infoValThresh
			bestLevels = levels
		print
		levels += 5
		correctPredictions = 0
		wrongPredictions = 0
		numTreeNodes = 0

	# build optimal level tree before pruning
	numTreeNodes = 0
	wordSet = []
	for keyWord in words:
		wordSet.append(keyWord)
	tree = buildDecisionTree(trainSet, wordSet, sentWords, 0, bestLevels, bestInfoValThresh)

	# run tree classifier on training set before prunint using optimal # tree levels and pruned tree
	print 'Decision tree on TRAINING set before pruning using optimal tree depth and info gain node threshold:'
	correctPredictions = 0
	wrongPredictions = 0
	start = time.time()
	for i in trainSet:
		prediction = classifierTree(tree, i, sentWords)
		if prediction == sentences[i].sentiment:
			correctPredictions += 1
		else:
			wrongPredictions += 1
	accuracy = (correctPredictions / float(correctPredictions + wrongPredictions) * 100) 
	end = time.time()
	print 'Number tree nodes: ', numTreeNodes
	print 'Time to classify ', len(trainSet), ' tuples in tree with ', (bestLevels+1), ' levels: ', '%.2f'%  (end - start), ' seconds'
	print 'Time to classify per tuple: ', '%.4f'% ((end - start) / float(trainNum)), ' seconds'
	print 'accuracy: ', '%.2f' % accuracy
	print 'infoValThresh: ', '%.2f' % bestInfoValThresh
	print

	# run tree classifier on test set before prunint using optimal # tree levels and pruned tree
	print 'Decision tree on test set before pruning using optimal tree depth and info gain node threshold:'
	correctPredictions = 0
	wrongPredictions = 0
	start = time.time()
	for i in testSet:
		prediction = classifierTree(tree, i, sentWords)
		if prediction == sentences[i].sentiment:
			correctPredictions += 1
		else:
			wrongPredictions += 1
	accuracy = (correctPredictions / float(correctPredictions + wrongPredictions) * 100) 
	end = time.time()
	print 'Number tree nodes: ', numTreeNodes
	print 'Time to classify ', len(testSet), ' tuples in tree with ', (bestLevels+1), ' levels: ', '%.2f'%  (end - start), ' seconds'
	print 'Time to classify per tuple: ', '%.4f'% ((end - start) / float(testNum)), ' seconds'
	print 'accuracy: ', '%.2f' % accuracy
	print 'infoValThresh: ', '%.2f' % bestInfoValThresh
	print
	

	# K-NN Classifier
	print 'kNN predictor Validation Set results before pruning: '
	optimalK = 1
	optimalWords = words
	optimalSWords = sentWords
	k = 1
	currentAccuracy = 0.0
	bestAccuracy = 0.0
	notCorrection = False

	while (k < 10):
		print 'Current k: ', k
		start = time.time()
		classifierTuple = kNearestNeighborClassifier(words, k, validSet, validNum, sentWords)
		end = time.time()
		print 'Time to classify ', validNum, ' validation tuples: ', '%.2f'% (end - start), ' seconds'
		print 'Time per tuple: ', '%.3f'% ((end - start) / float(validNum))
		currentAccuracy = ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100) 
		print 'Current Accuracy: ', '%.2f'% currentAccuracy
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = True
			optimalWords = words
			optimalSWords = sentWords
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using raw words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		currentAccuracy = ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = False
			optimalWords = words
			optimalSWords = sentWords
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using raw words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		start = time.time()
		classifierTuple = kNearestNeighborClassifier(wordsNltkStem, k, validSet, validNum, sentWordsStem)
		end = time.time()
		print 'Time to classify ', validNum, ' validation tuples: ', '%.2f'% (end - start), ' seconds'
		print 'Time per tuple: ', '%.2f'% ((end - start) / float(validNum)), ' seconds'
		currentAccuracy = ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100) 
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = True
			optimalWords = wordsNltkStem
			optimalSWords = sentWordsStem
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using NLTK Stemmed words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		currentAccuracy = ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = False
			optimalWords = wordsNltkStem
			optimalSWords = sentWordsStem
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using NLTK Stemmed words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		classifierTuple = kNearestNeighborClassifier(wordsNltkLemma, k, validSet, validNum, sentWordsLemma)
		currentAccuracy = ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100) 
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = True
			optimalWords = wordsNltkLemma
			optimalSWords = sentWordsLemma
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using NLTK Lemmatized words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		currentAccuracy = ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = False
			optimalWords = wordsNltkLemma
			optimalSWords = sentWordsLemma
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using NLTK Lemmatized words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		k += 2	# test k = 1, 3, 5
		#end while

	print 'optimalWords Test: '
	if optimalWords == words:
		print 'raw words optimal'
	elif optimalWords == wordsNltkStem:
		print 'Stem words optimal'
	elif optimalWords == wordsNltkLemma:
		print 'Lemma words optimal'
	else:
		print 'ERROR with optimal words variable'
	print

	# use k-Nearest Neighbor classifier on Training Set
	print 'kNN predictor TRAINING Set results BEFORE pruning using validation set optimizations for K, notCorrection, and dictionary:'
	print 'optimal k: ', optimalK, ' : not-correction: ', notCorrection 
	start = time.time()
	classifierTuple = kNearestNeighborClassifier(optimalWords, optimalK, trainSet, trainNum, optimalSWords)
	end = time.time()
	print 'Time to classify ', trainNum, ' training tuples: ', '%2f'% (end - start), ' seconds'
	print 'Time per training tuple: ', '%.2f'% ((end - start) / float(trainNum)), ' seconds'
	if notCorrection:
		print 'Correct predictions with NOT correction: ', classifierTuple[0]
		print 'Wrong predictions with NOT correction: ', classifierTuple[1]
		print 'Accuracy: ', '%.2f'% ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100)
	else:
		print 'Correct predictions w/out NOT correction: ', classifierTuple[2]
		print 'Wrong predictions w/out NOT correction: ', classifierTuple[3]
		print 'Accuracy: ', '%.2f'% ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)

	# use k-Nearest Neighbor classifier on Test Set
	print 'kNN predictor Test Set results BEFORE pruning using validation set optimizations for K, notCorrection, and dictionary:'
	print 'optimal k: ', optimalK, ' : not-correction: ', notCorrection 
	start = time.time()
	classifierTuple = kNearestNeighborClassifier(optimalWords, optimalK, testSet, testNum, optimalSWords)
	end = time.time()
	print 'Time to classify ', testNum, ' test tuples: ', '%2f'% (end - start), ' seconds'
	print 'Time per test tuple: ', '%.2f'% ((end - start) / float(testNum)), ' seconds'
	if notCorrection:
		print 'Correct predictions with NOT correction: ', classifierTuple[0]
		print 'Wrong predictions with NOT correction: ', classifierTuple[1]
		print 'Accuracy: ', '%.2f'% ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100)
	else:
		print 'Correct predictions w/out NOT correction: ', classifierTuple[2]
		print 'Wrong predictions w/out NOT correction: ', classifierTuple[3]
		print 'Accuracy: ', '%.2f'% ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)

	start = time.time()
	# prune single occurrence words
	pruneSingles(words, sentWords)
	pruneSingles(wordsNltkStem, sentWordsStem)
	pruneSingles(wordsNltkLemma, sentWordsLemma)

	# prune high entropy words (i.e. 50% split between positive, negative sentiment)
	pruneHighEntropy(words, sentWords)
	pruneHighEntropy(wordsNltkStem, sentWordsStem)
	pruneHighEntropy(wordsNltkLemma, sentWordsLemma)

	pruneTfIdf(words, sentWords)
	pruneTfIdf(wordsNltkStem, sentWordsStem)
	pruneTfIdf(wordsNltkLemma, sentWordsLemma)
	end = time.time()
	print 'Time to prune: ', '%.2f'% (end - start), ' seconds'

	
	# run tree classifier on validation set after pruning
	print 'Decision Tree classifier: '
	correctPredictions = 0
	wrongPredictions = 0
	bestLevels = 0
	bestInfoValThresh = 0
	infoValThresh = 0.1
	levels = 25
	bestAccuracy = 0
	while levels < 61:
		while infoValThresh <= 0.7:
			# build Decision Tree Classifier for Pruned training set
			start = time.time()
			wordSet = []
			for keyWord in words:
				wordSet.append(keyWord)
			tree = buildDecisionTree(trainSet, wordSet, sentWords, 0, levels, infoValThresh)
			print 'tree built, levels: ', levels
			end = time.time()
			print 'Time to build decision tree from unpruned dataset: ', '%.3f' % (end - start), ' seconds'	

			start = time.time()
			for i in validSet:
				prediction = classifierTree(tree, i, sentWords)
				if prediction == sentences[i].sentiment:
					correctPredictions += 1
				else:
					wrongPredictions += 1
			accuracy = (correctPredictions / float(correctPredictions + wrongPredictions) * 100) 
			end = time.time()
			print 'Number tree nodes: ', numTreeNodes
			print 'Time to classify ', len(validSet), ' tuples in tree with ', (levels+1), ' levels: ', '%.2f'%  (end - start), ' seconds'
			print 'Time to classify per tuple: ', '%.7f'% ((end - start) / float(validNum)), ' seconds'
			print 'accuracy: ', '%.2f' % accuracy
			print 'infoValThresh: ', '%.2f' % infoValThresh
			print 'levels: ', (levels+1)
			if accuracy > bestAccuracy:
				print 'BEST ACCURACY'
				bestAccuracy = accuracy
				bestLevels = levels
				bestInfoValThresh = infoValThresh
			infoValThresh += 0.2
			numTreeNodes = 0
			print
		levels += 5
		infoValThresh = 0.1
		correctPredictions = 0
		wrongPredictions = 0

	# build optimal level tree
	numTreeNodes = 0
	wordSet = []
	for keyWord in words:
		wordSet.append(keyWord)
	tree = buildDecisionTree(trainSet, wordSet, sentWords, 0, bestLevels, bestInfoValThresh)

	# run tree classifier on training set using optimal # tree levels and pruned tree
	print 'Decision tree classifier on TRAINING set after pruning using optimal tree depth and info gain node threshold:'
	correctPredictions = 0
	wrongPredictions = 0
	start = time.time()
	for i in trainSet:
		prediction = classifierTree(tree, i, sentWords)
		if prediction == sentences[i].sentiment:
			correctPredictions += 1
		else:
			wrongPredictions += 1
	accuracy = (correctPredictions / float(correctPredictions + wrongPredictions) * 100) 
	end = time.time()
	print 'Number tree nodes: ', numTreeNodes
	print 'Time to classify ', len(trainSet), ' tuples in tree with ', (bestLevels+1), ' levels: ', '%.2f'%  (end - start), ' seconds'
	print 'Time to classify per tuple: ', '%.4f'% ((end - start) / float(trainNum)), ' seconds'
	print 'accuracy: ', '%.2f' % accuracy
	print 'infoValThresh: ', '%.2f' % bestInfoValThresh

	# run tree classifier on test set using optimal # tree levels and pruned tree
	print 'Decision tree classifier on test set after pruning using optimal tree depth and info gain node threshold:'
	correctPredictions = 0
	wrongPredictions = 0
	start = time.time()
	for i in testSet:
		prediction = classifierTree(tree, i, sentWords)
		if prediction == sentences[i].sentiment:
			correctPredictions += 1
		else:
			wrongPredictions += 1
	accuracy = (correctPredictions / float(correctPredictions + wrongPredictions) * 100) 
	end = time.time()
	print 'Number tree nodes: ', numTreeNodes
	print 'Time to classify ', len(testSet), ' tuples in tree with ', (bestLevels+1), ' levels: ', '%.2f'%  (end - start), ' seconds'
	print 'Time to classify per tuple: ', '%.4f'% ((end - start) / float(testNum)), ' seconds'
	print 'accuracy: ', '%.2f' % accuracy
	print 'infoValThresh: ', '%.2f' % bestInfoValThresh
	
	
	# use k-Nearest Neighbor classifier on Validation Set to determine optimal K (test with k = 1, 3, 5)
	print 'kNN predictor Validation Set results after pruning: '
	optimalK = 1
	optimalWords = words
	optimalSWords = sentWords
	k = 1
	bestAccuracy = 0.0
	notCorrection = False
	while (k < 10):
		print 'Current k: ', k
		start = time.time()
		classifierTuple = kNearestNeighborClassifier(words, k, validSet, validNum, sentWords)
		currentAccuracy = ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100) 
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = True
			optimalWords = words
			optimalSWords = sentWords
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using raw words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		currentAccuracy = ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = False
			optimalWords = words
			optimalSWords = sentWords
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using raw words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		end = time.time()
		print 'Time to classify ', validNum, ' validation tuples: ', (end - start)
		print 'Time per tuple: ', '%.4f'% ((end - start) / float(validNum)), ' seconds'

		classifierTuple = kNearestNeighborClassifier(wordsNltkStem, k, validSet, validNum, sentWordsStem)
		currentAccuracy = ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100) 
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = True
			optimalWords = wordsNltkStem
			optimalSWords = sentWordsStem
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using NLTK Stemmed words dictionary'
		currentAccuracy = ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = False
			optimalWords = wordsNltkStem
			optimalSWords = sentWordsStem
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using NLTK Stemmed words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy

		classifierTuple = kNearestNeighborClassifier(wordsNltkLemma, k, validSet, validNum, sentWordsLemma)
		currentAccuracy = ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100) 
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = True
			optimalWords = wordsNltkLemma
			optimalSWords = sentWordsLemma
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using NLTK Lemmatized words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy
		currentAccuracy = ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)
		if currentAccuracy > bestAccuracy:
			bestAccuracy = currentAccuracy
			optimalK = k
			notCorrection = False
			optimalWords = wordsNltkLemma
			optimalSWords = sentWordsLemma
			print 'Best accuracy with k = ', k, '; NotCorrection: ', notCorrection, '; using NLTK Lemmatized words dictionary'
			print 'Accuracy: ', '%.2f'% bestAccuracy

		k += 2	# test k = 1, 3, 5
		#end while

	print 'optimalWords Test: '
	if optimalWords == words:
		print 'raw words optimal'
	elif optimalWords == wordsNltkStem:
		print 'Stem words optimal'
	elif optimalWords == wordsNltkLemma:
		print 'Lemma words optimal'
	else:
		print 'ERROR with optimal words variable'

	# use k-Nearest Neighbor classifier on Training Set
	start = time.time()
	classifierTuple = kNearestNeighborClassifier(optimalWords, optimalK, trainSet, trainNum, optimalSWords)
	end = time.time()
	print 'Time to classify ', trainNum, ' training tuples: ', '%.2f'% (end - start), ' seconds'
	print 'Time per tuple: ', '%.4f'% ((end - start) / float(trainNum)), ' seconds'
	print 'kNN predictor TRAINING Set results AFTER pruning using validation set optimizations for K, notCorrection, and dictionary:'
	print 'optimal k: ', optimalK, ' : not-correction: ', notCorrection 
	if notCorrection:
		print 'Correct predictions with NOT correction: ', classifierTuple[0]
		print 'Wrong predictions with NOT correction: ', classifierTuple[1]
		print 'Accuracy: ', '%.2f'% ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100)
	else:
		print 'Correct predictions w/out NOT correction: ', classifierTuple[2]
		print 'Wrong predictions w/out NOT correction: ', classifierTuple[3]
		print 'Accuracy: ', '%.2f'% ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)

	# use k-Nearest Neighbor classifier on Test Set
	start = time.time()
	classifierTuple = kNearestNeighborClassifier(optimalWords, optimalK, testSet, testNum, optimalSWords)
	end = time.time()
	print 'Time to classify ', testNum, ' test tuples: ', '%.2f'% (end - start), ' seconds'
	print 'Time per tuple: ', '%.4f'% ((end - start) / float(testNum)), ' seconds'
	print 'kNN predictor Test Set results AFTER pruning using validation set optimizations for K, notCorrection, and dictionary:'
	print 'optimal k: ', optimalK, ' : not-correction: ', notCorrection 
	if notCorrection:
		print 'Correct predictions with NOT correction: ', classifierTuple[0]
		print 'Wrong predictions with NOT correction: ', classifierTuple[1]
		print 'Accuracy: ', '%.2f'% ((classifierTuple[0] / float(classifierTuple[0] + classifierTuple[1])) * 100)
	else:
		print 'Correct predictions w/out NOT correction: ', classifierTuple[2]
		print 'Wrong predictions w/out NOT correction: ', classifierTuple[3]
		print 'Accuracy: ', '%.2f'% ((classifierTuple[2] / float(classifierTuple[2] + classifierTuple[3])) * 100)

