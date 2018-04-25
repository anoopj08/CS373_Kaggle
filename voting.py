import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm

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



    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0,stop_words='english',ngram_range = (1,4), strip_accents='unicode',preprocessor=proproc)

    X_train = vectorizer.fit_transform(param1)
    X_test = vectorizer.transform(param3)

    Xval_predict = vectorizer.transform(param4)
    Xval_test = param5

    if isValidation == 1:
        X_test = Xval_predict

    clf = MultinomialNB(alpha=0.49, class_prior=None, fit_prior=True)   #multinomial naive bayes
    clf.fit(X_train,param2)
    valResults = clf.predict(X_test)

    rf = RandomForestClassifier()   #Random Forest Classifier #bootstrap = True, max_depth = 50, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 10,  n_estimators = 200)
    rf.fit(X_train, param2)
    valResults_rf = rf.predict(X_test)

    svc = svm.LinearSVC()   #linear SVC
    svc.fit(X_train, param2)
    valResults_svc = svc.predict(X_test)

    vc = VotingClassifier(estimators=[('mnb',clf),('rf',rf),('svc',svc)],voting='hard', weights = [2,1,3])
    vc.fit(X_train,param2)
    valResults_vc = vc.predict(X_test)

    countVC = 0
    countNB = 0
    countSVC = 0
    numDiff = 0
    tot = len(valResults)
    finalResults = []
    for res in range(len(valResults)):
        if isValidation == 1:
            if valResults[res] == Xval_test[res]:
                countNB += 1
            if valResults_svc[res] == Xval_test[res]:
                countSVC += 1
            if valResults_vc[res] == Xval_test[res]:
                countVC += 1

        if valResults[res] != valResults_svc[res]:
            numDiff += 1
            #print(valResults[res],valResults_svc[res],valResults_rf[res], res)
            if valResults[res] == valResults_rf[res]:
                finalResults.append(valResults_rf[res])
            elif valResults_svc[res] == valResults_rf[res]:
                finalResults.append(valResults_rf[res])
            else:
                finalResults.append(valResults_svc[res])
        else:
            finalResults.append(valResults[res])

    if isValidation == 1:
        print("Multinomial Naive Bayes Accuracy: {}".format(float(countNB)/float(tot)))
        print("Linear SVC Accuracy: {}".format(float(countSVC)/float(tot)))
        print("Voting Classifier Accuracy: {}".format(float(countVC)/float(tot)))
    print("Number of Differences: {}".format(numDiff))


    with open('submission_avg.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id','sentiment'])
        count = 0
        for row in finalResults:
            spamwriter.writerow([count,finalResults[count]])
            count += 1

    with open('submission_mnb.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id','sentiment'])
        count = 0
        for row in finalResults:
            spamwriter.writerow([count,valResults[count]])
            count += 1

    with open('submission_svc.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id','sentiment'])
        count = 0
        for row in finalResults:
            spamwriter.writerow([count,valResults_svc[count]])
            count += 1


if __name__ == '__main__':
    main()
