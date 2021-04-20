from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-12
# fs = 512

# 512 = 1 second

def func(run):
    emg = run['emg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    indicies = indicies[3:]
    master_1 = np.zeros(shape=(10,3584,4))
    master_2 = np.zeros(shape=(10,3584,4))
    zscore = lambda x: (x - np.mean(x)) / (np.std(x) + EPSILON)
    emg = zscore(emg)
    i = 0
    j = 0
    for x in range(0, len(indicies)):
        trig_val= triggers[indicies[x]]
        trig_index = indicies[x]
        if trig_val == 102:
            segment = emg[trig_index-512*5:trig_index+512*2]
            master_1[i] = segment
            i+=1
        elif trig_val == 202:
            segment = emg[trig_index-512*5:trig_index+512*2]
            master_2[j] = segment
            j+=1
    return master_1,master_2

# Extract all MEP's and put them in a matrix for comparison
subj1_pre = loadmat('p2_subject1Pre.mat')
data = subj1_pre['subject1Pre']
data = data['MI'][0][0][0]
motion_1_emg = np.zeros(shape=(30,3584,4))
motion_2_emg = np.zeros(shape=(30, 3584, 4))
j = 0
for i in range(3):
    run = data[i]
    m1,m2 = func(run)
    for k in range(10):
        motion_1_emg[j] = m1[k]
        motion_2_emg[j] = m2[k]
        j+=1
print('hi')
