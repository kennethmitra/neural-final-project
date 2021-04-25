from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-12


# fs = 512

# 512 = 1 second


def extractSegments_MI(run, t_minus=.5, t_plus=.5, do_zs=False):
    emg = run['emg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if x != 0])
    indicies = indicies[3:]
    master_1 = np.zeros(shape=(10, calcLen(t_minus, t_plus), 4))
    master_2 = np.zeros(shape=(10, calcLen(t_minus, t_plus), 4))
    zscore = lambda x: (x - np.mean(x)) / (np.std(x) + EPSILON)

    assert not do_zs

    i = 0
    j = 0
    for x in range(0, len(indicies)):
        trig_val = triggers[indicies[x]]
        trig_index = indicies[x]
        if trig_val == 102:
            segment = emg[trig_index - round(fs * t_minus):trig_index + round(fs * t_plus)]
            master_1[i] = segment
            i += 1
        elif trig_val == 202:
            segment = emg[trig_index - round(fs * t_minus):trig_index + round(fs * t_plus)]
            master_2[j] = segment
            j += 1
    return master_1, master_2


def extractSegments_noMI(run, t_minus=.5, t_plus=.5, do_zs=False):  # Edit this
    emg = run['emg']
    triggers = run['hdr']['triggers'][0][0]
    indicies = np.array([i for i, x in enumerate(triggers) if (x != 0)])
    indicies = indicies[3:]
    master = np.zeros(shape=(len(indicies), calcLen(t_minus, t_plus), 4))
    zscore = lambda x: (x - np.mean(x)) / (np.std(x) + EPSILON)
    assert not do_zs
    i = 0
    j = 0
    for x in range(0, len(indicies)):
        trig_val = triggers[indicies[x]]
        trig_index = indicies[x]
        if trig_val == 100:
            segment = emg[trig_index - round(fs * t_minus):trig_index + round(fs * t_plus)]
            master[i] = segment
            i += 1
    return master


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

def fourG(noMI_emg_segs, MI_flex_emg_segs, MI_ext_emg_segs, name, t_minus):
    MI_flex_grand_avgs = getAVG(MI_flex_emg_segs)
    MI_ext_grand_avgs = getAVG(MI_ext_emg_segs)
    noMI_grand_avgs = getAVG(noMI_emg_segs)

    sensor_names = ('prox ext', 'dist ext', 'prox flx', 'dist flx')
    for sens_idx in range(4):
        x = np.arange(0, len(noMI_grand_avgs[1])) / fs
        plt.plot(x, noMI_grand_avgs[sens_idx], alpha=.7, color='blue')
        plt.plot(x, MI_flex_grand_avgs[sens_idx], alpha=.7, color='green')
        plt.plot(x, MI_ext_grand_avgs[sens_idx], alpha=.7, color='orange')
        plt.axvline(x=t_minus, color='red', linestyle='--', alpha=.3)
        title = name + f' {sensor_names[sens_idx]} average across all runs'
        plt.title(title)
        plt.legend(['No MI', 'Wrist Flexion MI', 'Wrist Extension MI'])
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (µV)')
        plt.savefig(title + '.png')
        plt.show()

def custom_grapher(label_data, name, t_minus):

    labels = [x[0] for x in label_data]
    lines = [getAVG(x[1]) for x in label_data]

    sensor_names = ('prox ext', 'dist ext', 'prox flx', 'dist flx')
    for sens_idx in range(4):
        x = np.arange(0, len(lines[0][1])) / fs
        for line in lines:
            plt.plot(x, line[sens_idx], alpha=.7)
        plt.axvline(x=t_minus, color='red', linestyle='--', alpha=.3)
        title = name + f' {sensor_names[sens_idx]} grand averages'
        plt.title(title)
        plt.legend(labels)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (µV)')
        plt.savefig(title + '.png')
        plt.show()


################### Load Pre ####################
subj = 2
prepost = 'Pre'

# Extract all MEP's and put them in a matrix for comparison
subj1_pre = loadmat(f'p2_subject{subj}{prepost}.mat')
data = subj1_pre[f'subject{subj}{prepost}']
MI_data = data['MI'][0][0][0]

###################################
# Zoomed in Analysis of MEP region
###################################

# Extract Segments from the MI case (-0.5 sec to 0.5 sec after TMS)
t_minus = 0.5
t_plus = 0.5
fs = 512
calcLen = lambda a, b: round(a + b) * fs

MI_flex_emg_segs = np.zeros(shape=(30, calcLen(t_minus, t_plus), 4))
MI_ext_emg_segs = np.zeros(shape=(30, calcLen(t_minus, t_plus), 4))
segment_idx = 0
for run_idx in range(3):
    run = MI_data[run_idx]
    m1, m2 = extractSegments_MI(run, t_minus, t_plus)
    for k in range(10):
        MI_flex_emg_segs[segment_idx] = m1[k]
        MI_ext_emg_segs[segment_idx] = m2[k]
        segment_idx += 1

print(MI_ext_emg_segs.shape)

# Extract segments from the no MI case (-0.5 sec to 0.5 sec after TMS)
MI_emg_segs = np.zeros(shape=(80, calcLen(t_minus, t_plus), 4))

noMI_data = data['noMI'][0][0][0]
run = noMI_data[0]
noMI_emg_segs = extractSegments_noMI(run, t_minus, t_plus)
print(noMI_emg_segs.shape)

# Average them
print("----- Averages -----")
#fourG(noMI_emg_segs, MI_flex_emg_segs, MI_ext_emg_segs, name='S2-Post EMG MEPs No MI vs MI -', t_minus=t_minus)
noMI_emg_segs_pre = noMI_emg_segs
MI_flex_emg_segs_pre = MI_flex_emg_segs
MI_ext_emg_segs_pre = MI_ext_emg_segs


################### Load Post ####################
subj = 2
prepost = 'Post'

# Extract all MEP's and put them in a matrix for comparison
subj1_pre = loadmat(f'p2_subject{subj}{prepost}.mat')
data = subj1_pre[f'subject{subj}{prepost}']
MI_data = data['MI'][0][0][0]

###################################
# Zoomed in Analysis of MEP region
###################################

# Extract Segments from the MI case (-0.5 sec to 0.5 sec after TMS)
t_minus = 0.5
t_plus = 0.5
fs = 512
calcLen = lambda a, b: round(a + b) * fs

MI_flex_emg_segs = np.zeros(shape=(30, calcLen(t_minus, t_plus), 4))
MI_ext_emg_segs = np.zeros(shape=(30, calcLen(t_minus, t_plus), 4))
segment_idx = 0
for run_idx in range(3):
    run = MI_data[run_idx]
    m1, m2 = extractSegments_MI(run, t_minus, t_plus)
    for k in range(10):
        MI_flex_emg_segs[segment_idx] = m1[k]
        MI_ext_emg_segs[segment_idx] = m2[k]
        segment_idx += 1

print(MI_ext_emg_segs.shape)

# Extract segments from the no MI case (-0.5 sec to 0.5 sec after TMS)
MI_emg_segs = np.zeros(shape=(80, calcLen(t_minus, t_plus), 4))

noMI_data = data['noMI'][0][0][0]
run = noMI_data[0]
noMI_emg_segs = extractSegments_noMI(run, t_minus, t_plus)
print(noMI_emg_segs.shape)

# Average them
print("----- Averages -----")
#fourG(noMI_emg_segs, MI_flex_emg_segs, MI_ext_emg_segs, name='S2-Post EMG MEPs No MI vs MI -', t_minus=t_minus)
noMI_emg_segs_post = noMI_emg_segs
MI_flex_emg_segs_post = MI_flex_emg_segs
MI_ext_emg_segs_post = MI_ext_emg_segs

# 'No MI', 'Wrist Flexion MI', 'Wrist Extension MI'
#
#
custom_grapher([('No MI Pre', noMI_emg_segs_pre),
                ('No MI Post', noMI_emg_segs_post),
                ], name='S2-No MI case Pre vs Post EMG MEPs -', t_minus=t_minus)
custom_grapher([('Wrist Flexion MI Pre', MI_flex_emg_segs_pre),
                 ('Wrist Flexion MI Post', MI_flex_emg_segs_post),
                ], name='S2-Wrist Flexion Pre vs Post EMG MEPs -', t_minus=t_minus)
custom_grapher([('Wrist Extension MI Pre', MI_ext_emg_segs_pre),
                 ('Wrist Extension MI Post', MI_ext_emg_segs_post),
                ], name='S2-Wrist Extension Pre vs Post EMG MEPs -', t_minus=t_minus)