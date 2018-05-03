import numpy as lumpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import multioutput
from sklearn import svm
import pandas as pd
from sys import argv
import csv
import nltk
#nltk.download('all')
from nltk.stem import PorterStemmer
from mlxtend.classifier import StackingClassifier



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
        print("Percentages add up to more than 100. Try again you fucker.")
        exit()

    trainingData = pd.read_csv(trainingFile, sep=',', quotechar='"', header=0, engine='python')
    trainingData = trainingData.drop_duplicates(subset=['text'],keep='first')
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



    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0,stop_words='english',ngram_range = (1,3), strip_accents='unicode')#,preprocessor=proproc)

    X_train = vectorizer.fit_transform(param1)
    X_test = vectorizer.transform(param3)

    Xval_predict = vectorizer.transform(param4)
    Xval_test = param5

    #clf1 = MultinomialNB(alpha=0.49, class_prior=None, fit_prior=True)
    clf1 = svm.LinearSVC(C=0.3)
    clf2 = RandomForestClassifier()
    #lr = multioutput.MultiOutputRegressor(GradientBoostingRegressor(), n_jobs=-1)
    lr = LogisticRegression()#RandomForestClassifier()

    stack_clf = StackingClassifier(classifiers=[clf1, clf2],
                          meta_classifier=lr)

    print('10-fold cross validation:\n')

    for clf, label in zip([clf1, clf2, stack_clf],
                            [#'Multinomial Naive Bayes',
                            'Linear SVM',
                            'Random Forest Classifier',
                            'StackingClassifier']):

        scores = model_selection.cross_val_score(clf,X_train,param2,cv=10,scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    clf2.fit(X_train,param2)
    valResults = clf2.predict(Xval_predict)
    count = 0
    tot = len(valResults)
    for res in range(len(valResults)):
        if valResults[res] == Xval_test[res]:
            count += 1
        else:
            print(param1[res],valResults[res],Xval_test[res])

    print("Stacking Ensembler Score: {}".format(float(count)/float(tot)))

    results = clf2.predict(X_test)

    # with open('submission_stacking.csv', 'w') as csvfile:
    #     spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     spamwriter.writerow(['id','sentiment'])
    #     count = 0
    #     for row in results:
    #         spamwriter.writerow([count,results[count]])
    #         count += 1


if __name__ == '__main__':
    main()
