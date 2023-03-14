import numpy as np
import pandas as pd
import os

def filter_spikes(spikes, clusters, quality=0.5, region='MO'):
    good_clusterIDs = clusters['cluster_id'][((clusters['label'] > quality) & ([True if region in s else False for s in clusters['acronym']]))]
    good_spikes_loc = np.isin(spikes['clusters'], good_clusterIDs) # Boolean array of whether a spike is in a good cluster
    
    good_spikes = {}
    for i in spikes:
        good_spikes[i] = spikes[i][good_spikes_loc]
    
    return good_spikes

def bin_spikes(spikes, trials, ROOT_DIR, exp_name, bin_size=50e-3, save=False):
    df = pd.DataFrame(data = {'clusters':spikes['clusters'], 'times':spikes['times']})
    df = df.groupby('clusters')['times'].apply(np.array) # Neurons x Spike Times
    
    # trials_spikes = [] # The Spike times and results of each trial
    bin_data = []
    goCue_times = []
    firstMove_times = []
    choices = []
    accuracy = []

    for i in range(len(trials['goCueTrigger_times'])):
        # Mouse must have a miniumum 50 ms reaction time for us to consider the trial.
        if (trials['firstMovement_times'][i] < trials['goCue_times'][i]+0.05):
            continue
        elif ((np.isnan(trials['goCue_times'][i])) | (np.isnan(trials['firstMovement_times'][i]))):
            continue
        
        # spike_range = {}
        
        # spike_range['goCue_times'] = trials['goCue_times'][i]
        # spike_range['firstMovement_times'] = trials['firstMovement_times'][i]
        # spike_range['choice'] = trials['choice'][i]
        # spike_range['feedbackType'] = trials['feedbackType'][i]
        
        goCue_times.append(trials['goCue_times'][i])
        firstMove_times.append(trials['firstMovement_times'][i])
        choices.append(trials['choice'][i])
        accuracy.append(trials['feedbackType'][i])
        
        x = []
        hist_bins = np.arange(trials['firstMovement_times'][i]-0.1, trials['firstMovement_times'][i]+1.00001, bin_size)
        
        
        for j in df: # Iterate through spike times of each cluster
            inds = ((j>(trials['firstMovement_times'][i]-0.1)) & (j<=(trials['firstMovement_times'][i]+1.00001)))
            x.append(np.histogram(j[inds], hist_bins)[0])
        
        spikes_df = pd.DataFrame(x, index=df.index)
        # spike_range['spikes_df'] = spikes_df
        bin_data.append(spikes_df)
        
        # trials_spikes.append(spike_range)
        
    if save:
        np.save(os.path.join(ROOT_DIR, 'data', 'out', exp_name, 'bin_data.npy'), bin_data)
        np.savetxt(os.path.join(ROOT_DIR, 'data', 'out', exp_name, 'goCue_times.csv'), goCue_times, delimiter=',')
        np.savetxt(os.path.join(ROOT_DIR, 'data', 'out', exp_name, 'firstMove_times.csv'), firstMove_times, delimiter=',')
        np.savetxt(os.path.join(ROOT_DIR, 'data', 'out', exp_name, 'choices.csv'), choices, delimiter=',')
        np.savetxt(os.path.join(ROOT_DIR, 'data', 'out', exp_name, 'accuracy.csv'), accuracy, delimiter=',')
    
    return bin_data, goCue_times, firstMove_times, choices, accuracy

def class_data(z, choices, accuracy, time_significance):
    X = z[:, time_significance, :]
    y = []
    
    for i in range(len(choices)):
        if ((choices[i]==1) & (accuracy[i]==1)):
            y.append(0)
        elif ((choices[i]==-1) & (accuracy[i]==1)):
            y.append(1)
        elif ((choices[i]==1) & (accuracy[i]==-1)):
            y.append(2)
        elif ((choices[i]==-1) & (accuracy[i]==-1)):
            y.append(3)
    
    
    return np.array(X), np.array(y)