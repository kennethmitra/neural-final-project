from scipy.io import loadmat
from scipy.signal import butter
from scipy.signal import filtfilt
import numpy as np
import matplotlib.pyplot as plt


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
    zscore = lambda x: (x - np.mean(x))
    # Filter here
    # fc1 = 36
    # fc2 = 172
    # Wp = [fc1,fc2]
    # Wp = [element * 2 for element in Wp]
    # Wp = [element / 512 for element in Wp]
    # signal = zscore(signal)
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


def getTaskExecutions(run, shortest):
    emg = run['emg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    indicies = indicies[3:]

    master_1 = np.zeros(shape=(10, shortest, 4))
    master_2 = np.zeros(shape=(10, shortest, 4))

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


def getShortest(run):
    emg = run['emg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    shortest = np.inf
    # for x in range(0, len(indicies)):
    #     print(triggers[indicies[x]])
    # print('hi-----------------')

    for x in range(0, len(indicies)):
        trig_val = triggers[indicies[x]]
        trig_index = indicies[x]
        if trig_val == 101:
            trig_index_end = indicies[x + 1]
            segment = emg[trig_index:trig_index_end]
            if len(segment) < shortest:
                shortest = len(segment)
        elif trig_val == 201:
            trig_index_end = indicies[x + 1]
            segment = emg[trig_index:trig_index_end]
            if len(segment) < shortest:
                shortest = len(segment)
    return shortest


def getPSD(data, label):
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
    data_motion1 = np.zeros(shape=(1, 4))
    data_motion2 = np.zeros(shape=(1, 4))

    for i in range(30):
        current = motion_1_emg_pre[i]
        data_motion1 = np.vstack((data_motion1,current))
        current = motion_2_emg_pre[i]
        data_motion2 = np.vstack((data_motion2,current))

    data_motion1 = data_motion1[1:,:]
    data_motion2 = data_motion2[1:,:]
    plt.psd(data_motion1[0] ** 2, 512)
    plt.psd(data_motion2[0] ** 2, 512)
    plt.legend(['hand flexion','hand extension'])

    plt.xlabel('Frequency')
    plt.ylabel('PSD(db)')
    plt.title(label + 'PSD plot')
    plt.show()



def createClassifier(data, label):
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

    targetValue = np.zeros(shape=(features_motion1.shape[0], 1))
    for i in range(len(targetValue)):
        targetValue[i] = int(1)
    features_motion1 = np.hstack((features_motion1, targetValue))
    for i in range(len(targetValue)):
        targetValue[i] = int(2)
    features_motion2 = np.hstack((features_motion2, targetValue))
    data = np.vstack((features_motion1, features_motion2))

    # make classifier now
    from sklearn.model_selection import train_test_split

    X = data[:, :12]
    y = data[:, 12:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    from sklearn.svm import SVC
    from sklearn import metrics

    model = SVC(kernel='poly', C=2500)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    print(label + " SVM Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))


allData = [('p2_subject1Pre.mat', 'subject1Pre'), ('p2_subject1Post.mat', 'subject1Post'),
           ('p2_subject2Pre.mat', 'subject2Pre'), ('p2_subject2Post.mat', 'subject2Post')]
for i in range(len(allData)):
    subj1_post = loadmat(allData[i][0])
    data = subj1_post[allData[i][1]]
    data = data['MI'][0][0][0]
    getPSD(data,allData[i][1])
    createClassifier(data, allData[i][1])
