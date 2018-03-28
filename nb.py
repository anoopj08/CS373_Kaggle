import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sys import argv


def main():
    trainingFile = argv[1]
    trainingData = pd.read_csv(trainingFile, sep=',', quotechar='"', header=None, engine='python')
    X = lumpy.asarray(trainingData.as_matrix())
    Y = lumpy.asarray()[0,1,2,3,4])




if __name__ == '__main__':
    main()
