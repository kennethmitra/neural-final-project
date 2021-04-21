from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-12
# LENGTH = 512*7
# BEFORE = 512*5
# AFTER = 512*2
LENGTH = 102
BEFORE = 51
AFTER = 51
# fs = 512

# 512 = 1 second

def func(run):
    emg = run['emg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    indicies = indicies[3:]
    master_1 = np.zeros(shape=(10, LENGTH, 4))
    master_2 = np.zeros(shape=(10, LENGTH, 4))
    # emg = zscore(emg)
    i = 0
    j = 0
    for x in range(0, len(indicies)):
        trig_val = triggers[indicies[x]]
        trig_index = indicies[x]
        if trig_val == 102:
            segment = emg[trig_index - BEFORE:trig_index + AFTER]
            master_1[i] = segment
            i += 1
        elif trig_val == 202:
            segment = emg[trig_index - BEFORE:trig_index + AFTER]
            master_2[j] = segment
            j += 1
    return master_1, master_2


def getAVG(master):
    sensor1 = np.zeros(master.shape[1])
    sensor2 = np.zeros(master.shape[1])
    sensor3 = np.zeros(master.shape[1])
    sensor4 = np.zeros(master.shape[1])
    for i in range(master.shape[0]):
        current = master[i]
        s1 = current[:, 0]
        s1 = s1.flatten()
        sensor1 += s1

        s2 = current[:, 1]
        s2 = s2.flatten()
        sensor2 += s2

        s3 = current[:, 2]
        s3 = s3.flatten()
        sensor3 += s3

        s4 = current[:, 3]
        s4 = s4.flatten()
        sensor4 += s4
    sensor1 = sensor1 / master.shape[0]
    sensor2 = sensor2 / master.shape[0]
    sensor3 = sensor3 / master.shape[0]
    sensor4 = sensor4 / master.shape[0]
    zscore = lambda x: (x - np.mean(x))
    sensor1 = zscore(sensor1)
    sensor2 = zscore(sensor2)
    sensor3 = zscore(sensor3)
    sensor4 = zscore(sensor4)

    return sensor1, sensor2, sensor3, sensor4


def fourG(master, post, name):
    sensor1_pre, sensor2_pre, sensor3_pre, sensor4_pre = getAVG(master)
    sensor1_post, sensor2_post, sensor3_post, sensor4_post = getAVG(post)

    plt.plot(sensor1_pre, alpha = .7,color = 'blue')
    plt.plot(sensor1_post, alpha = .7, color = 'orange')
    plt.axvline(x=BEFORE, color='red', linestyle='--', alpha=.3)
    title = name + ' prox ext average across all runs with MI Session 1 vs 3'
    plt.title(title)
    plt.legend(['Session 1','Session 3'])
    plt.savefig(title+'.png')
    plt.show()

    plt.plot(sensor2_pre,alpha = .7,color = 'blue')
    plt.plot(sensor2_post,alpha = .7, color = 'orange')
    plt.axvline(x=BEFORE, color='red', linestyle='--', alpha=.3)
    title = name + ' dist ext average across all runs with MI Session 1 vs 3'
    plt.title(title)
    plt.legend(['Session 1', 'Session 3'])
    plt.savefig(title + '.png')
    plt.show()

    plt.plot(sensor3_pre,alpha = .7,color = 'blue')
    plt.plot(sensor3_post,alpha = .7, color = 'orange')
    plt.axvline(x=BEFORE, color='red', linestyle='--', alpha=.3)
    title = name + ' prox flx average across all runs with MI Session 1 vs 3'
    plt.title(title)
    plt.legend(['Session 1', 'Session 3'])
    plt.savefig(title + '.png')
    plt.show()

    plt.plot(sensor4_pre,alpha = .7,color = 'blue')
    plt.plot(sensor4_post,alpha = .7, color = 'orange')
    plt.axvline(x=BEFORE, color='red', linestyle='--', alpha=.3)
    title = name + ' dist flx average across all runs with MI Session 1 vs 3'
    plt.title(title)
    plt.legend(['Session 1', 'Session 3'])
    plt.savefig(title + '.png')
    plt.show()


# Extract all MEP's and put them in a matrix for comparison
subj1_pre = loadmat('p2_subject1Pre.mat')
data = subj1_pre['subject1Pre']
data = data['MI'][0][0][0]
motion_1_emg_pre = np.zeros(shape=(30, LENGTH, 4))
motion_2_emg_pre = np.zeros(shape=(30, LENGTH, 4))
j = 0
for i in range(3):
    run = data[i]
    m1, m2 = func(run)
    for k in range(10):
        motion_1_emg_pre[j] = m1[k]
        motion_2_emg_pre[j] = m2[k]
        j += 1

subj1_post = loadmat('p2_subject1Post.mat')
data = subj1_post['subject1Post']
data = data['MI'][0][0][0]
motion_1_emg_post = np.zeros(shape=(30, LENGTH, 4))
motion_2_emg_post = np.zeros(shape=(30, LENGTH, 4))
j = 0
for i in range(3):
    run = data[i]
    m1, m2 = func(run)
    for k in range(10):
        motion_1_emg_post[j] = m1[k]
        motion_2_emg_post[j] = m2[k]
        j += 1
fourG(motion_1_emg_pre, motion_1_emg_post, 'hand flexion')
fourG(motion_2_emg_pre, motion_2_emg_post, 'hand extension')
print('hi')
