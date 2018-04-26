import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sys import argv
import csv
from nltk.stem import PorterStemmer
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from textblob import TextBlob
import itertools


def proproc(w):
    sid = SentimentIntensityAnalyzer()
    w = w.lower()
    porter = PorterStemmer()
    w = porter.stem(w)

    #print(w)
    retSentence = ""
    retSentiment = 0

    wList = w.split()
    wLen = len(wList)
    allComb = []
    for i in range(1,4):
        tempList = list(itertools.combinations(wList,i))
        for t in tempList:
            allComb.append(t)

    for fragment in allComb:
        sent = ' '.join(word for word in fragment)
        ss = sid.polarity_scores(sent)
        currSentiment = abs(ss['compound'])
        #print(sent,currSentiment)
        if currSentiment > retSentiment and len(sent) > len(retSentence):
            retSentiment = currSentiment
            retSentence = sent

    #print(retSentiment,retSentence)
    #exit()

    if retSentiment == 0:
        return w
    else:
        return retSentence
    return w

def main():
    trainingFile = argv[1]
    testFile = argv[2]
    trainPercentage = float(argv[3])
    validPercentage = float(argv[4])

    if int(trainPercentage) + int(validPercentage) > 100:
        print("Percentages add up to more than 100. Try again you fucker.")
        exit()

    trainingData = pd.read_csv(trainingFile, sep=',', quotechar='"', header=0, engine='python')
    testData = pd.read_csv(testFile, sep=',', quotechar='"', header=0, engine='python')

    inOrder = trainingData.as_matrix()

    X = inOrder
    Xval = inOrder

    trainIndex = int((trainPercentage/100) * float(len(X)))
    X = X[:trainIndex]

    valIndex = int((trainPercentage/100) * float(len(Xval))) + trainIndex
    Xval = Xval[:valIndex]       #  ADD THIS LINE BACK TO ENABLE VALIDATION SET

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



    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0,stop_words='english',ngram_range = (1,4), strip_accents='unicode',preprocessor=proproc)

    X_train = vectorizer.fit_transform(param1)
    X_test = vectorizer.transform(param3)

    Xval_predict = vectorizer.transform(param4)
    Xval_test = param5

    clf = MultinomialNB(alpha=0.49, class_prior=None, fit_prior=True)

    clf.fit(X_train,param2)

    valResults = clf.predict(Xval_predict)
    count = 0
    tot = len(valResults)
    for res in range(len(valResults)):
        if valResults[res] == Xval_test[res]:
            count += 1

    print(float(count)/float(tot))

    results = clf.predict(X_test)

    with open('submission.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id','sentiment'])
        count = 0
        for row in results:
            spamwriter.writerow([count,results[count]])
            count += 1


if __name__ == '__main__':
    main()
