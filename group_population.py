import numpy as np
import dateutil.parser
import math
import util
from sklearn import linear_model

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
logreg = linear_model.LogisticRegression(class_weight='balanced')
logreg.fit(examples[:,:63], examples[:,63])

print(examples.shape)
print(logreg.coef_.shape)
print(logreg.intercept_)
print(logreg.predict_proba(examples[:,:63]))
