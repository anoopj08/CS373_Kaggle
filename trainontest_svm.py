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

    vectorizer = TfidfVectorizer(use_idf = True,norm = 'l2',sublinear_tf=True,stop_words='english',ngram_range = (1,2), strip_accents='unicode')#,preprocessor=proproc)

    X_train = vectorizer.fit_transform(param1)
    X_test = vectorizer.transform(param3)

    Xval_predict = vectorizer.transform(param4)
    Xval_test = param5

    classifier_rbf = svm.LinearSVC()#loss = 'squared_hinge', max_iter = 500, tol = 1e-07)
    classifier_rbf.fit(X_train, param2)
    prediction_rbf = classifier_rbf.predict(Xval_predict)

    count = 0
    tot = len(prediction_rbf)
    for res in range(tot):
        if prediction_rbf[res] == Xval_test[res]:
            count += 1

    print("SVM Score on Training Set: {}".format(float(count)/float(tot)))

    results = classifier_rbf.predict(X_test)

    Xtt_1 = trainingData.as_matrix()
    Xtt_2 = testData.as_matrix()

    Xtt_headlines = Xtt_1[:,0].tolist()
    Xtt_headlines += Xtt_2[:,0].tolist()

    Xtt_scores = Xtt_1[:,1].tolist()
    Xtt_scores += results.tolist()

    Xtt_train = vectorizer.fit_transform(Xtt_headlines)

    clf = svm.LinearSVC()
    clf.fit(Xtt_train,Xtt_scores)

    test_predictions = Xtt_2[:,0]
    val_predictions = Xtt_1[:,0]
    tr_val_predictions = vectorizer.transform(val_predictions)
    tr_test_predictions = vectorizer.transform(test_predictions)
    predictions = clf.predict(tr_val_predictions)

    count = 0
    tot = len(predictions)
    for res in range(tot):
        if predictions[res] == Xval_test[res]:
            count += 1

    print("SVM Score on Training+Test Set: {}".format(float(count)/float(tot)))

    results = clf.predict(tr_test_predictions)



    with open('submission_svm_traintest.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id','sentiment'])
        count = 0
        for row in results:
            spamwriter.writerow([count,results[count]])
            count += 1


if __name__ == '__main__':
    main()
