import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sys import argv
import csv
import nltk
from nltk.stem import PorterStemmer


def proproc(w):
    w = w.lower()
    porter = PorterStemmer()
    w = porter.stem(w)
    #print(w)
    return w

def main():
    trainingFile = argv[1]
    testFile = argv[2]
    trainPercentage = float(argv[3])
    validPercentage = float(argv[4])

    if int(trainPercentage) + int(validPercentage) > 100:
        print("Percentages add up to more than 100. Please try again.")
        exit()

    trainingData = pd.read_csv(trainingFile, sep=',', quotechar='"', header=0, engine='python')
    testData = pd.read_csv(testFile, sep=',', quotechar='"', header=0, engine='python')

    inOrder = trainingData.as_matrix()
    lumpy.random.shuffle(inOrder)

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

    classifier_rbf = svm.LinearSVC()
    classifier_rbf.fit(X_train, param2)
    prediction_rbf = classifier_rbf.predict(Xval_predict)

    count = 0
    tot = len(prediction_rbf)
    for res in range(tot):
        if prediction_rbf[res] == Xval_test[res]:
            count += 1

    print(float(count)/float(tot))

    results = classifier_rbf.predict(X_test)
    with open('submission.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id','sentiment'])
        count = 0
        for row in results:
            spamwriter.writerow([count,results[count]])
            count += 1


if __name__ == '__main__':
    main()
