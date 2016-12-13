#play with model

from gensim.models import word2vec
import numpy as np  # Make sure that numpy is imported
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
# Download the punkt tokenizer for sentence splitting
import nltk.data
#nltk.download() 
import logging
from sklearn.cluster import KMeans
import time

from TFDataSet import *
import argparse
from softmax import train_soft

#print model.doesnt_match("friends community sherlock her".split())
#print model.most_similar("awful")

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

def getDataVecs(train, test, model, num_features):
	clean_train_reviews = []
	for review in train["review"]:
	    clean_train_reviews.append( review_to_wordlist( review, \
	        remove_stopwords=True ))

	trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

	print "Creating average feature vecs for test reviews"
	clean_test_reviews = []
	for review in test["review"]:
	    clean_test_reviews.append( review_to_wordlist( review, \
	        remove_stopwords=True ))

	testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

	return trainDataVecs, testDataVecs

def RFTraining(train, trainDataVecs, testDataVecs):
	# Fit a random forest to the training data, using 100 trees
	forest = RandomForestClassifier( n_estimators = 100 )

	print "Fitting a random forest to labeled training data..."
	forest = forest.fit( trainDataVecs, train["sentiment"] )

	# Test & extract results 
	result = forest.predict( testDataVecs )

	# Write the test results 
	output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	output.to_csv( "docs/random_forest_results.csv", index=False, quoting=3 )

	return output

def TFTraining(train_vals, train_labels, test_vals):
    train_labels = np.eye(2)[train_labels.as_matrix(columns=None)]
    data_set = TFDataSet(train_vals, train_labels, test_vals)
    #return train_soft(data_set)

    """
    print train_vals.shape
    print train_labels.shape
    print test_vals.shape
    """

    result = train_soft(data_set)

    # Write the test results 
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "docs/tensor_results.csv", index=False, quoting=3 )

    return output


    #return 'oi'

"""
#Train, test and model and number of features
train = pd.read_csv("docs/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

test = pd.read_csv("docs/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )

model = word2vec.Word2Vec.load('docs/300features_40minwords_10context')

num_features = len(model.syn0[1,:])

#get the data vectors
trainDataVecs, testDataVecs = getDataVecs(train, test, model, num_features)


#Random Forest Training 
output1 = RFTraining(train, trainDataVecs, testDataVecs)

#Tensor Training 
output2 = TFTraining(trainDataVecs, train["sentiment"], testDataVecs)

print output1
print output2
"""

model = word2vec.Word2Vec.load('docs/300features_40minwords_10context')

print 'model.doesnt_match("france germany england london ".split())'
print model.doesnt_match("france germany england london ".split())


print 'model.most_similar("depp")'
print model.most_similar("depp")



