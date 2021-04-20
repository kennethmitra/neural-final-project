from scipy.io import loadmat
import numpy as np

EPSILON = 1e-12

# Extract all MEP's and put them in a matrix for comparison
data = loadmat('p2_subject1Pre.mat')['subject1Pre']
run1_emg = data['MI'][0, 0][0, 0]['emg']
run2_emg = data['MI'][0, 0][0, 0]['emg']
run3_emg = data['MI'][0, 0][0, 0]['emg']
print(run1_emg)

# Z score the EMG data
zscore = lambda x: (x - np.mean(x))/(np.std(x)+EPSILON)
runs = (run1_emg, run2_emg, run3_emg)
z_runs = [np.apply_along_axis(arr=run, axis=1, func1d=zscore) for run in runs]
print(z_runs[0])

# def func(run):
#     emg = run['emg']
#     triggers = run['hdr']['triggers'][0][0]
#     indices = np.array([i for i, x in enumerate(triggers) if x != 0])
#     for x in indices:
#         print(triggers[x])
#     print('hi')
#
#
# # Extract all MEP's and put them in a matrix for comparison
# subj1_pre = loadmat('p2_subject1Pre.mat')
# data = subj1_pre['subject1Pre']
# data = data['MI'][0][0][0]
# for i in range(3):
#     run = data[i]
#     func(run)