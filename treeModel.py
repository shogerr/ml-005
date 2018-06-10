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

examples = create_group()
clf = tree.DecisionTreeClassifier(class_weight='balanced')
clf = clf.fit(examples[:,:-1], examples[:,-1])

dotData = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dotData)
graph.render('tree')
