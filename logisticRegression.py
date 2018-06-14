import numpy as np
import dateutil.parser
import math
import util
from sklearn import linear_model

convert = lambda x: int(dateutil.parser.parse(x).hour)

def create_group():
    #subject_files = ['Subject_1.csv', 'Subject_4.csv', 'Subject_6.csv', 'Subject_9.csv' ]
    #subject_files = ['Subject_2_part1.csv']
    subject_files = ['Subject_7_part1.csv']

    samples = np.empty(shape=(0,64))

    for f in subject_files:
        sample = np.loadtxt('./data/'+f, converters={0:convert}, delimiter=',')
        s = util.reshape_sample(sample)
        samples = np.vstack((samples, s))

    return samples

def create_test_group():
   #f = 'general_test_instances.csv'
   #f = 'subject2_instances.csv'
   f = 'subject7_instances.csv'
   sample = np.loadtxt('./test/'+f, delimiter=',')
   s = util.reshape_test(sample)

   return s

def testRateRun(examples):
   dataSplit = 1500
   clf = linear_model.LogisticRegression(class_weight='balanced', C=10.0)
   clf = clf.fit(examples[:dataSplit,:-1], examples[:dataSplit,-1])
   print('test data predictions:')
   util.successRate(clf.predict(examples[dataSplit:,:-1]), examples[dataSplit:,-1])

def createPredictionsCSV(clf, testData, fileName):
   predictions = clf.predict(testData)
   predProb = clf.predict_proba(testData)
   with open(fileName, 'w+') as f:
       for i in range(len(predictions)):
            prob = '%.5f'%(predProb[i][1])
            f.write(str(prob) + ',' + str(predictions[i]) + '\n')


examples = create_group()
clf = linear_model.LogisticRegression(class_weight='balanced', C=10.0)
clf = clf.fit(examples[:,:-1], examples[:,-1])
#util.successRate(clf.predict(examples[:,:-1]), examples[:,-1])
testExamples = create_test_group()
#createPredictionsCSV(clf, testExamples, 'logistic_regression_predictions.csv')
testRateRun(examples)
