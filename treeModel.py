import numpy as np
import dateutil.parser
import math
import util
from sklearn import tree
import graphviz

convert = lambda x: int(dateutil.parser.parse(x).hour)

def create_group():
    subject_files = ['Subject_1.csv', 'Subject_4.csv', 'Subject_6.csv', 'Subject_9.csv' ]

    samples = np.empty(shape=(0,64))

    for f in subject_files:
        sample = np.loadtxt('./data/'+f, converters={0:convert}, delimiter=',')
        s = util.reshape_sample(sample)
        samples = np.vstack((samples, s))

    return samples

def create_test_group():
    f = 'general_test_instances.csv'
    sample = np.loadtxt('./data/'+f, delimiter=',')
    s = util.reshape_test(sample)

    return s


def testRateRun(examples):
    dataSplit = 1500
    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    clf = clf.fit(examples[:dataSplit,:-1], examples[:dataSplit,-1])
    print('test data predictions:')
    successRate(clf.predict(examples[dataSplit:,:-1]), examples[dataSplit:,-1])


def successRate(expected, actual):
    success = 0
    for i in range(len(actual)):
        if expected[i] == actual[i]:
            success += 1
    print('correct: ' + str(success), 'total: ' + str(len(expected)), 'rate: ' + str(success/len(expected)))


def createTreeImage(clf):
    dotData = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dotData)
    graph.render('tree')

def createPredictionsCSV(clf, testData):
    predictions = clf.predict(testData)
    predProb = clf.predict_proba(testData)
    with open('tree_predictions.csv', 'w+') as f:
        for i in range(len(predictions)):
            prob = max(predProb[i])
            f.write(str(predictions[i]) + ',' + str(predProb[i]) + '\n')



examples = create_group()
clf = tree.DecisionTreeClassifier(class_weight='balanced')
clf = clf.fit(examples[:,:-1], examples[:,-1])
successRate(clf.predict(examples[:,:-1]), examples[:,-1])
testExamples = create_test_group()
createPredictionsCSV(clf, testExamples)

testRateRun(examples)
