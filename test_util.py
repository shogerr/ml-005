import util
import numpy as np

def test_reshape_test():
    s = np.loadtxt('./data/subject2_instances.csv', delimiter=',')
    print(s[0])
    print(s.shape)
    a = util.reshape_test(s)
    print(a)
    print(a[0])
    print(a.shape)

def test_reshape_sample():
    s = np.arange(70).reshape(10,7).T
    s = np.vstack((s,s+1))
    s[6][9] = 1
    s[13][9] = 0

    t = util.reshape_sample(s)
    print(s)
    print(t)
    print(t.shape)

test_reshape_sample()
