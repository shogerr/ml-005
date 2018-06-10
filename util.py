import numpy as np
import math

def reshape_sample(s):
    # reverse samples and break up sample set where last column is true
    t = np.split(np.flip(s,0), np.where(np.flip(s,0)[:,9] == 1)[0][0:])

    # remove chunks with fewer than 63 elements
    t = [x for x in t if x.size > 62]

    # create set for all samples
    sample_set = np.empty(shape=(0,64))
    for i in range(len(t)):
        sample_local = np.empty(shape=(0,63))
        # Get the number of rows that are useful as examples
        num_rows = math.floor(t[i].shape[0]/7)

        for j in range(0, num_rows*7, 7):
            u = np.flip(t[i][j:j+7,:9], 0).reshape(1, 63)
            sample_local = np.vstack((sample_local, u))

        # create the labels
        labels = np.zeros(shape=(1,num_rows))
        if t[i][0,9] == 1:
            labels[0][0] = 1
        sample_local = np.hstack((sample_local, labels.T))
        sample_set = np.vstack((sample_set, sample_local))

    return sample_set

def reshape_test(s):
    seed = np.array((0,9,18,27,36,45,54))
    f = np.empty(shape=(0,0), dtype=np.int)
    for i in range(9):
        f = np.append(f, seed + i)

    s[:,np.arange(63)] = s[:,f]
    return s 

