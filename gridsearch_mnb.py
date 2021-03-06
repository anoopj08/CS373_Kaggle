import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd
from sys import argv
import csv

from time import time
import logging

from textblob import TextBlob

pipeline = Pipeline([
    #('vect', CountVectorizer()),
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])
#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0,stop_words='english',ngram_range = (1,4), strip_accents='unicode',preprocessor=proproc)

parameters = {
    'tfidf__use_idf': (True, False),
    'tfidf__sublinear_tf': (True, False),
    'tfidf__max_df': (0.1,0.3,0.5,0.7,0.9,1.0,1.2,1.4),
    'tfidf__norm': ('l1', 'l2'),
    'tfidf__ngram_range': ((1,1),(1,2),(1,3),(1,4)),
    'clf__alpha': (0.1,0.3,0.5,0.7,1.0),
    'clf__fit_prior': ('True','False'),
}

def proproc(w):
    w = w.lower()
    porter = PorterStemmer()
    w = porter.stem(w)
    return w

def main():
    trainingFile = argv[1]
    testFile = argv[2]
    trainPercentage = float(argv[3])
    validPercentage = float(argv[4])
    isValidation = 0

    if "-v" in argv:
        isValidation = 1

    if int(trainPercentage) + int(validPercentage) > 100:
        print("Percentages add up to more than 100. Try again you fucker.")
        exit()

    trainingData = pd.read_csv(trainingFile, sep=',', quotechar='"', header=0, engine='python')
    testData = pd.read_csv(testFile, sep=',', quotechar='"', header=0, engine='python')

    inOrder = trainingData.as_matrix()
    #lumpy.random.shuffle(inOrder)

    X = inOrder
    Xval = inOrder

    trainIndex = int((trainPercentage/100) * float(len(X)))
    X = X[:trainIndex]

    valIndex = int((trainPercentage/100) * float(len(Xval))) + trainIndex
    #Xval = Xval[:valIndex]       #  ADD THIS LINE BACK TO ENABLE VALIDATION SET

    Xtest = testData.as_matrix()

    param1 = X[:,0].tolist()
    param2 = X[:,1].tolist()
    param3 = Xtest[:,0].tolist()

    param4 = Xval[:,0].tolist()
    param5 = Xval[:,1].tolist()

    for dp in X:
        param1.append(dp[0])
        temp = lumpy.asarray([dp[1]])
        param2.append(temp)

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    t0 = time()
    grid_search.fit(param1, param2)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    main()

# Best score: 0.943
# Best parameters set:
# 	clf__alpha: 0.1
# 	clf__fit_prior: 'True'
# 	tfidf__max_df: 0.1
# 	tfidf__ngram_range: (1, 3)
# 	tfidf__norm: 'l2'
# 	tfidf__sublinear_tf: True
# 	tfidf__use_idf: True
