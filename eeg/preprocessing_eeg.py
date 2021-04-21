from scipy.io import loadmat
from scipy import signal
from scipy.signal import butter
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-12
# fs = 512

# 1 second = 512 samples

def _process(run):
    eeg = run['eeg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    indicies = indicies[3:]
    master_1 = np.zeros(shape=(10, 3584, 32))
    master_2 = np.zeros(shape=(10, 3584, 32))
    zscore = lambda x: (x - np.mean(x)) / (np.std(x) + EPSILON)
    sos = butter(6, (0.1, 20), 'bandpass', fs=512, output='sos') 
    eeg = zscore(signal.sosfilt(sos, eeg))
    i = 0
    j = 0
    for x in range(len(indicies)):
        trig_val = triggers[indicies[x]]
        trig_index = indicies[x]
        if trig_val == 102:
            segment = eeg[trig_index - 512 * 5 : trig_index + 512 * 2]
            master_1[i] = segment
            i += 1
        elif trig_val == 202:
            segment = eeg[trig_index - 512 * 5 : trig_index + 512 * 2]
            master_2[j] = segment
            j += 1
    return master_1, master_2


# Extract all MEP's and put them in a matrix for comparison
def process(subject, session):
    filename = 'p2_subject{}{}'.format(str(subject), session)
    data = loadmat(filename + '.mat')
    data = data[filename.split('_')[-1]]
    data = data['MI'][0][0][0]
    motion_1_eeg = np.zeros(shape=(30, 3584, 32))
    motion_2_eeg = np.zeros(shape=(30, 3584, 32))
    j = 0
    for i in range(3):
        run = data[i]
        m1, m2 = _process(run)
        for k in range(10):
            motion_1_eeg[j] = m1[k]
            motion_2_eeg[j] = m2[k]
            j += 1
    np.save('subject{}{}_motion_1_eeg.npy'.format(str(subject), session), motion_1_eeg)
    np.save('subject{}{}_motion_2_eeg.npy'.format(str(subject), session), motion_2_eeg)


def main():
    process(1, 'Pre')
    process(2, 'Pre')
    process(1, 'Post')
    process(2, 'Post')


if __name__ == "__main__":
    main()