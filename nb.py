import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sys import argv


def main():
    trainingFile = argv[1]
    trainingData = pd.read_csv(trainingFile, sep=',', quotechar='"', header=0, engine='python')
    X = trainingData.as_matrix()


    #print(X[0])
    #Y = lumpy.asarray([0,1,2,3,4])
    max = 0
    for dp in X:
        currList = dp[0].split()
        if len(currList) > max:
            max = len(currList)
        dp[0] = currList

    for dp in X:
        currList = dp[0]
        size = len(currList)
        for i in range(max-size):
            currList.append('')
        dp[0] = currList

    param1 = X[:,0].tolist()
    param2 = X[:,1].tolist()
    print(param1[0])

    for dp in X:
        param1.append(dp[0])
        temp = lumpy.asarray([dp[1]])
        param2.append(temp)

    param1 = lumpy.asarray(param1)
    param2 = lumpy.asarray(param2)

    clf = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)

    clf.fit(param1,param2)
    print(clf.predict(["Everybody is dead"]))




if __name__ == '__main__':
    main()
