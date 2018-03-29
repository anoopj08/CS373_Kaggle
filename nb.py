import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sys import argv


def main():
    trainingFile = argv[1]
    testFile = argv[2]


    trainingData = pd.read_csv(trainingFile, sep=',', quotechar='"', header=0, engine='python')
    testData = pd.read_csv(testFile, sep=',', quotechar='"', header=0, engine='python')
    X = trainingData.as_matrix()
    Xtest = testData.as_matrix()

    param1 = X[:,0].tolist()
    param2 = X[:,1].tolist()

    param3 = Xtest[:,0].tolist()

    for dp in X:
        param1.append(dp[0])
        temp = lumpy.asarray([dp[1]])
        param2.append(temp)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')

    X_train = vectorizer.fit_transform(param1)
    X_test = vectorizer.transform(param3)

    clf = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)

    clf.fit(X_train,param2)
    print(X_test)
    results = clf.predict(X_test)

if __name__ == '__main__':
    main()
