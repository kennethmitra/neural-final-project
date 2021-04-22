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


def getShortest(run):
    emg = run['emg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    indicies = indicies[3:]
    shortest = np.inf

    for x in range(0, len(indicies)):
        trig_val = triggers[indicies[x]]
        trig_index = indicies[x]
        if trig_val == 101:
            trig_index_end = indicies[x + 1]
            segment = emg[trig_index:trig_index_end]
            if len(segment) < shortest:
                shortest = len(segment)
        elif trig_val == 202:
            trig_index_end = indicies[x + 1]
            segment = emg[trig_index:trig_index_end]
            if len(segment) < shortest:
                shortest = len(segment)
    return shortest


def getTaskExecutions(run, shortest):
    emg = run['emg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    indicies = indicies[3:]

    master_1 = np.zeros(shape=(10, shortest, 4))
    master_2 = np.zeros(shape=(10, shortest, 4))

    # for x in range(0, len(indicies)):
    #     print(triggers[indicies[x]])
    # print('hi-----------------')

    i = 0
    j = 0
    for x in range(0, len(indicies)):
        trig_val = triggers[indicies[x]]
        trig_index = indicies[x]
        if trig_val == 101:
            trig_index_end = indicies[x + 1]
            segment = emg[trig_index:trig_index_end]
            master_1[i] = segment[:shortest]
            i += 1
        elif trig_val == 201:
            trig_index_end = indicies[x + 1]
            segment = emg[trig_index:trig_index_end]
            master_2[j] = segment[:shortest]
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

    plt.plot(sensor1_pre, alpha=.7, color='blue')
    plt.plot(sensor1_post, alpha=.7, color='orange')
    plt.axvline(x=BEFORE, color='red', linestyle='--', alpha=.3)
    title = name + ' prox ext average across all runs with MI Session 1 vs 3'
    plt.title(title)
    plt.legend(['Session 1', 'Session 3'])
    plt.savefig(title + '.png')
    plt.show()

    plt.plot(sensor2_pre, alpha=.7, color='blue')
    plt.plot(sensor2_post, alpha=.7, color='orange')
    plt.axvline(x=BEFORE, color='red', linestyle='--', alpha=.3)
    title = name + ' dist ext average across all runs with MI Session 1 vs 3'
    plt.title(title)
    plt.legend(['Session 1', 'Session 3'])
    plt.savefig(title + '.png')
    plt.show()

    plt.plot(sensor3_pre, alpha=.7, color='blue')
    plt.plot(sensor3_post, alpha=.7, color='orange')
    plt.axvline(x=BEFORE, color='red', linestyle='--', alpha=.3)
    title = name + ' prox flx average across all runs with MI Session 1 vs 3'
    plt.title(title)
    plt.legend(['Session 1', 'Session 3'])
    plt.savefig(title + '.png')
    plt.show()

    plt.plot(sensor4_pre, alpha=.7, color='blue')
    plt.plot(sensor4_post, alpha=.7, color='orange')
    plt.axvline(x=BEFORE, color='red', linestyle='--', alpha=.3)
    title = name + ' dist flx average across all runs with MI Session 1 vs 3'
    plt.title(title)
    plt.legend(['Session 1', 'Session 3'])
    plt.savefig(title + '.png')
    plt.show()


def extractFeats(signal):
    WSize = .300
    Olap = .825
    fs = 512
    import math
    WSize = math.floor(WSize * fs)
    nOlap = math.floor(Olap * WSize)
    hop = WSize - nOlap
    nx = len(signal)
    length = int(np.fix((nx - (WSize - hop)) / hop))

    MAV_feature = np.zeros(shape=(length, 4))
    VAR_feature = np.zeros(shape=(length, 4))
    WL_feature = np.zeros(shape=(length, 4))
    for i in range(1, length + 1):
        s = ((i - 1) * hop + 1)
        e = min((i - 1) * hop + WSize, len(signal))
        segment = signal[s:e, :]
        MAV_feature[i - 1] = np.mean(np.abs(segment), axis=0)
        VAR_feature[i - 1] = (np.std(segment, axis=0)) ** 2
        WL_feature[i - 1] = np.sum(np.abs(np.diff(segment, axis=0)), axis=0)
    features = np.hstack((MAV_feature, VAR_feature, WL_feature))
    return features


# Extract all MEP's and put them in a matrix for comparison
# subj1_pre = loadmat('p2_subject1Pre.mat')
# data = subj1_pre['subject1Pre']
# data = data['MI'][0][0][0]
# motion_1_emg_pre = np.zeros(shape=(30, LENGTH, 4))
# motion_2_emg_pre = np.zeros(shape=(30, LENGTH, 4))
# j = 0
# for i in range(3):
#     run = data[i]
#     m1, m2 = func(run)
#     for k in range(10):
#         motion_1_emg_pre[j] = m1[k]
#         motion_2_emg_pre[j] = m2[k]
#         j += 1
#
# subj1_post = loadmat('p2_subject1Post.mat')
# data = subj1_post['subject1Post']
# data = data['MI'][0][0][0]
# motion_1_emg_post = np.zeros(shape=(30, LENGTH, 4))
# motion_2_emg_post = np.zeros(shape=(30, LENGTH, 4))
# j = 0
# for i in range(3):
#     run = data[i]
#     m1, m2 = func(run)
#     for k in range(10):
#         motion_1_emg_post[j] = m1[k]
#         motion_2_emg_post[j] = m2[k]
#         j += 1
# fourG(motion_1_emg_pre, motion_1_emg_post, 'hand flexion')
# fourG(motion_2_emg_pre, motion_2_emg_post, 'hand extension')
# print('hi')

# subj1_pre = loadmat('p2_subject1Pre.mat')
# data = subj1_pre['subject1Pre']
subj1_post = loadmat('p2_subject1Post.mat')
data = subj1_post['subject1Post']
data = data['MI'][0][0][0]
shortest_array = []
for i in range(3):
    run = data[i]
    shortest_i = getShortest(run)
    shortest_array.append([shortest_i])
shortest = min(shortest_array)[0]

motion_1_emg_pre = np.zeros(shape=(30, shortest, 4))
motion_2_emg_pre = np.zeros(shape=(30, shortest, 4))
j = 0
for i in range(3):
    run = data[i]
    m1, m2 = getTaskExecutions(run, shortest)
    for k in range(10):
        motion_1_emg_pre[j] = m1[k]
        motion_2_emg_pre[j] = m2[k]
        j += 1
features_motion1 = np.zeros(shape=(1, 12))
features_motion2 = np.zeros(shape=(1, 12))

for i in range(30):
    feature_i_1 = extractFeats(motion_1_emg_pre[i])
    feature_i_2 = extractFeats(motion_2_emg_pre[i])

    features_motion1 = np.vstack((features_motion1, feature_i_1))
    features_motion2 = np.vstack((features_motion2, feature_i_2))

features_motion1 = features_motion1[1:, :]
features_motion2 = features_motion2[1:, :]

# create feature scatter plots

# for i in range(4):
#     sensor_i_mav_1 = features_motion1[:, 0 + i]
#     sensor_i_var_1 = features_motion1[:, 8 + i]
#     sensor_i_mav_2 = features_motion2[:, 0 + i]
#     sensor_i_var_2 = features_motion2[:, 8 + i]
#     plt.scatter(sensor_i_mav_1,sensor_i_var_1)
#     plt.scatter(sensor_i_mav_2,sensor_i_var_2)
#     plt.xlabel("MAV")
#     plt.ylabel("VAR")
#     plt.legend(['hand flexion','hand extension'])
#     plt.title(f'sensor {i+1} MAV vs VAR')
#     plt.show()

targetValue = np.zeros(shape = (features_motion1.shape[0],1))
for i in range(len(targetValue)):
    targetValue[i] = int(1)
features_motion1 = np.hstack((features_motion1,targetValue))
for i in range(len(targetValue)):
    targetValue[i] = int(2)
features_motion2 = np.hstack((features_motion2,targetValue))
data = np.vstack((features_motion1,features_motion2))


# make classifier now
from sklearn.model_selection import train_test_split
X = data[:,:12]
y = data[:,12:]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

from sklearn.svm import SVC
from sklearn import metrics

model = SVC(kernel='poly', C=2500)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test,y_pred))
