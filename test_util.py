import util
import numpy as np

s = np.loadtxt('./data/subject2_instances.csv', delimiter=',')
print(s[0])
print(s.shape)
a = util.reshape_test(s)
print(a)
print(a[0])
print(a.shape)
