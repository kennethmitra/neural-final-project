from scipy.io import loadmat
from scipy import signal
from scipy.signal import butter
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-12
# fs = 512

# 1 second = 512 samples

def _process(run, apply_filter=False):
    eeg = run['eeg']
    if apply_filter:
        sos = butter(10, (0.1, 20), 'bandpass', fs=512, output='sos')
        for i in range(eeg.shape[1]):
            eeg[:, i] = signal.sosfilt(sos, eeg[:, i])
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    indicies = indicies[3:]
    master_1 = np.zeros(shape=(10, 3584, 32))
    master_2 = np.zeros(shape=(10, 3584, 32))
    zscore = lambda x: (x - np.mean(x)) / (np.std(x) + EPSILON)
    eeg = zscore(eeg)
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
def process(subject, session, apply_filter=False):
    filename = '../p2_subject{}{}'.format(str(subject), session)
    data = loadmat(filename + '.mat')
    data = data[filename.split('_')[-1]]
    data = data['MI'][0][0][0]
    motion_1_eeg = np.zeros(shape=(30, 3584, 32))
    motion_2_eeg = np.zeros(shape=(30, 3584, 32))
    j = 0
    for i in range(3):
        run = data[i]
        m1, m2 = _process(run, apply_filter)
        for k in range(10):
            motion_1_eeg[j] = m1[k]
            motion_2_eeg[j] = m2[k]
            j += 1

    plot_spectrum(motion_1_eeg, motion_2_eeg, subject, session, apply_filter)

    np.save('subject{}{}_motion_1_eeg.npy'.format(str(subject), session), motion_1_eeg)
    np.save('subject{}{}_motion_2_eeg.npy'.format(str(subject), session), motion_2_eeg)


def plot_spectrum(motion_1_eeg, motion_2_eeg, subject, session, apply_filter):
    if session == 'Pre':
        _session = 1
    elif session == 'Post':
        _session = 3
    
    for i in range(motion_1_eeg.shape[-1]):
        _motion_1_eeg = motion_1_eeg[:, :, i]
        _motion_2_eeg = motion_2_eeg[:, :, i]
        avg_motion_1_eeg = np.mean(_motion_1_eeg, axis=0)
        avg_motion_2_eeg = np.mean(_motion_2_eeg, axis=0)

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Subject {} Session {} EEG Channel {}'.format(str(subject), str(_session), str(i)))
        ax1.psd(avg_motion_1_eeg, Fs=512)
        ax2.psd(avg_motion_2_eeg, Fs=512)
        ax1.set_title('Hand Flexion')
        ax2.set_title('Hand Extension')
        fig.subplots_adjust(hspace=0.8)
        fig_name = 'subject_{}_session_{}_channel_{}_eeg_psd'.format(str(subject), str(_session), str(i))
        if apply_filter:
            fig_name += '_bandpass_filtered'
        plt.savefig(fig_name)
        plt.close()


def main():
    process(1, 'Pre')
    process(2, 'Pre')
    process(1, 'Post')
    process(2, 'Post')


if __name__ == "__main__":
    main()