import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
import pandas as pd
from sys import argv
from sklearn import datasets, linear_model
import csv
import nltk
from sklearn import svm
#nltk.download('all')
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import AdaBoostRegressor #For Regression

from sklearn.ensemble import GradientBoostingClassifier #For Classification
from sklearn.ensemble import GradientBoostingRegressor #For Regression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor(random_state = 42)
from pprint import pprint


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
        print("Percentages add up to more than 100. Try again you fucker.")
        exit()


    trainingData = pd.read_csv(trainingFile, sep=',', quotechar='"', header=0, engine='python')
    testData = pd.read_csv(testFile, sep=',', quotechar='"', header=0, engine='python')

    inOrder = trainingData.as_matrix()

    lumpy.random.shuffle(inOrder)

    X = inOrder
    Xval = inOrder

    #X is training data
    #Xval is validation data

    trainIndex = int((trainPercentage/100) * float(len(X)))
    X = X[:trainIndex]

    #X is now the legit trianing set

    valIndex = int((trainPercentage/100) * float(len(Xval))) + trainIndex
    Xval = Xval[:valIndex]       #  ADD THIS LINE BACK TO ENABLE VALIDATION SET
    #X is now the legit validation set

    Xtest = testData.as_matrix()

        #param1 is text        param2 is sentiment from (0, 4)
    param1 = X[:,0].tolist()
    param2 = X[:,1].tolist()


    param3 = Xtest[:,0].tolist()

    param4 = Xval[:,0].tolist()
    param5 = Xval[:,1].tolist()


    for dp in X:
        param1.append(dp[0])
        temp = lumpy.asarray([dp[1]])
        param2.append(temp)

        #print(param1)
        #print(temp)
        #print(param2)


        """
        from sklearn.feature_extraction.text import CountVectorizer

        CountVectorizer.fit(data) #Learn the vocabulary of the training data

        CountVectorizer.transform(data) #Converts the training data into the Document Term Matrix

        CountVectorizer.transform(test) #Uses the fitted vocabulary (training) to build a document term matrix from the testing data.

        """

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0,stop_words='english',ngram_range = (1,4), strip_accents='unicode',preprocessor=proproc)


    X_train = vectorizer.fit_transform(param1) #converts training data into document term matrix
    X_test = vectorizer.transform(param3) #builds a document term matrix from testing data

    Xval_predict = vectorizer.transform(param4) #builds a document term matrix from validation data
    Xval_test = param5 #actual sentiment from validation set


    """
    Now that we have our features, we can train a classifier to try to predict the
    category of a post. Lets start with a naive Bayes classifier, which provides a nice
    baseline for this task scikit learn includes several variants of this classifier the
    one most suitable for word counts is the multinomial variant
    """




    """
    clf = MultinomialNB(alpha=0.49, class_prior=None, fit_prior=True)
    clf.fit(X_train,param2)
    valResults = clf.predict(Xval_predict)
    """


    #clf = RandomForestClassifier(max_features = 0.25, n_estimators = 500, n_jobs=2, random_state=0, criterion = 'gini', max_depth=None, min_weight_fraction_leaf=0.0, max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, verbose=0, warm_start=False,class_weight=None)

    #clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,class_weight=None)

    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in lumpy.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in lumpy.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    #pprint(random_grid)

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1, cv = 2, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, param2)

    pprint(rf_random.best_params_)

    exit()
    """

    #rf = RandomForestClassifier(bootstrap = True, max_depth = 50, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 10,  n_estimators = 200)
    dt = DecisionTreeClassifier()
    clf = AdaBoostClassifier(n_estimators=1000, base_estimator=dt,learning_rate=1.0)

    clf.fit(X_train, param2)
    valResults = clf.predict(Xval_predict)


    """
    clf = GaussianNB()
    # Train our classifier
    model = clf.fit(X_train.todense(), param2)
    valResults = clf.predict(Xval_predict.todense())
    """

    """
    clf = qda()
    clf.fit(X_train.todense(), param2)
    #qdabase = qda.fit(X_train.todense(), param2)
    valResults = clf.predict(Xval_predict.todense())
    """

    """
    clf = lda()
    clf.fit(X_train.todense(), param2)
    #qdabase = qda.fit(X_train.todense(), param2)
    valResults = clf.predict(Xval_predict.todense())
    """
    #model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=1, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)



    """
    model = svm.SVC(probability=True, kernel='linear', C=1, gamma=1)
    clf = AdaBoostClassifier(n_estimators=100, base_estimator=model,learning_rate=1)
    clf.fit(X_train, param2)
    clf.score(X_train, param2)
    valResults= clf.predict(Xval_predict)
    """

    """
    clf = AdaBoostClassifier(SGDClassifier(loss='log'), algorithm='SAMME')
    clf.fit(X_train, param2)
    clf.score(X_train, param2)
    valResults= clf.predict(Xval_predict)
    """

    count = 0
    tot = len(valResults)
    for res in range(len(valResults)):
        if valResults[res] == Xval_test[res]:
            count += 1

    print(float(count)/float(tot))

    #results = clf.predict(X_test.todense())
    results = clf.predict(X_test)

    with open('submission_randomforest.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id','sentiment'])
        count = 0
        for row in results:
            spamwriter.writerow([count,results[count]])
            count += 1


if __name__ == '__main__':
    main()
