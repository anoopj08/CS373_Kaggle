import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sys import argv
import csv

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
    lumpy.random.shuffle(inOrder)

    X = inOrder
    Xval = inOrder

    # # for i in range(int(((100-testPercentage)/100) * len(X))):
    # #     X = lumpy.delete(X,(0),axis=0)
	
    # # for i in range(int(((100-validationPercentage)/100) * len(Xval))):
    # #     Xval = lumpy.delete(Xval,,axis=0)
    
    # # print(X.size, Xval.size)

    # trainPercentage = 1-(float(trainPercentage)/float(100))
    # delElemsTrain = (len(X))*(trainPercentage)
    # for i in range(int(delElemsTrain)):
    #     X = lumpy.delete(X,(0),axis=0)

    # validPercentage = 1-(float(validPercentage)/float(100))
    # delElemsTest = (len(Xval))*(validPercentage)
    # for i in range(int(delElemsTest)):
    #     Xval = lumpy.delete(Xval,X.size,axis=0)

    trainIndex = int((trainPercentage/100) * float(len(X)))
    #print(X.size, trainIndex)
    X = X[:trainIndex]
    #print(X.size)


    valIndex = int((trainPercentage/100) * float(len(Xval))) + trainIndex
    Xval = Xval[trainIndex:valIndex]

    print(X[0],Xval[0])
    
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


    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')

    X_train = vectorizer.fit_transform(param1)
    X_test = vectorizer.transform(param3)

    Xval_predict = vectorizer.transform(param4)
    Xval_test = param5

    clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

    clf.fit(X_train,param2)

    valResults = clf.predict(Xval_predict)
    count = 0
    tot = len(valResults)
    for res in range(len(valResults)):
        if valResults[res] == Xval_test[res]:
            count += 1
    print(float(count)/float(tot))

    results = clf.predict(X_test)
    #print(results)

    with open('submission.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id','sentiment'])
        count = 0
        for row in results:
            spamwriter.writerow([count,results[count]])
            count += 1


if __name__ == '__main__':
    main()
