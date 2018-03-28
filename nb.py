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
            lumpy.append(currList, '0', axis=None)
        dp[0] = currList
    #print(X)
    param1 = X[:,0]
    param2 = X[:,1]
    #print(param1.tolist())
    #exit()
    #print(param2.tolist())
    #print(len(param1[0]), len(param1[1]),len(param1),len(param2))
    clf = MultinomialNB()
    # lumpy.array(list(param2), dtype=int)
    param2.reshape(1,-1)
    print(param1[0],param2[0])
    #print(type(param2))
    clf.fit(param1,param2)
    print(clf.predict(["Everybody is dead"]))




if __name__ == '__main__':
    main()
