%% Plot the entire session
load('p2_subject1Pre.mat')

hold on
plot(subject1Pre.MI(1).emg)
plot(((subject1Pre.MI(1).hdr.triggers == 102) || (subject1Pre.MI(1).hdr.triggers == 202))*max(subject1Pre.MI(1).emg))

%% Extract all datapointsession
