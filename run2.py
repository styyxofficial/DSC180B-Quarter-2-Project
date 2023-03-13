from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
# from ibllib.atlas import AllenAtlas
# from brainbox.core import TimeSeries
from brainbox.processing import bin_spikes

# import jax.random
from vlgpax.model import Session
from vlgpax.kernel import RBF
from vlgpax import vi
from einops import rearrange
# from sklearn.preprocessing import StandardScaler

from src.data.data_prep import filter_spikes, bin_spikes


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import random
import sys
import json
import os
import shutil

ROOT_DIR = os.path.dirname(__file__)

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise Exception('Config file not provided as an argument')
    elif len(sys.argv) > 2:
        raise Exception('Too many arguments. Only provide the config file as an argument.')
    
    with open('config/' + sys.argv[1]) as json_file:
        config = json.load(json_file)
    
    # make necessary directories
    shutil.rmtree(os.path.join(ROOT_DIR, 'data', 'out', config['exp_name']))
    shutil.rmtree(os.path.join(ROOT_DIR, 'output', config['exp_name'], 'imgs'))
    os.makedirs(os.path.join(ROOT_DIR, 'data', 'out', config['exp_name']))
    os.makedirs(os.path.join(ROOT_DIR, 'output', config['exp_name'], 'imgs'))
    
    # Connect to ONE Database
    # mode=local uses locally stored data. If the mouse data you want to analyze is not in your cache_dir, remove the mode=local parameter so the data can be downloaded to your machine
    pw = 'international'
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', password=pw, silent=True,
              cache_dir='./data/raw/ONE', mode='local')
    
    # Load data
    sl = SpikeSortingLoader(one=one, eid=config['eid'], pname=config['probe_name'])
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    trials = one.load_object(config['eid'], 'trials', collection='alf')
    
    # Save a raster plot before data cleaning
    if config['raster']:
        sl.raster(spikes, channels, save_dir='./output/' + config['exp_name'] + '/imgs/raster1.png')
    
    # Clean data
    # Here we filter channels based on quality (>0.5) and brain region (Primary Motor Cortex [MO])
    spikes = filter_spikes(spikes, clusters)
    
    # Save a raster plot after data cleaning
    if config['raster']:
        sl.raster(spikes, channels, save_dir='./output/' + config['exp_name'] + '/imgs/raster2.png')
    
    bin_data, goCue_times, firstMove_times, choices, accuracy = bin_spikes(spikes, trials, ROOT_DIR, config['exp_name'], bin_size=config['bin_size'], save=True)
    n_train = 250 # number of training data
    
    
    sessionTrain = Session(config['bin_size'])
    for i in range(n_train):
        sessionTrain.add_trial(i, y=bin_data[i].T)
        
    sessionTest = Session(config['bin_size'])
    for i in range(n_train, len(bin_data)):
        sessionTest.add_trial(i, y=bin_data[i].T)
        
    kernel = RBF(scale=1, lengthscale=config['length_scale'])
    sessionTrain, params = vi.fit(sessionTrain, n_factors=config['latent_dims'], kernel=kernel, seed=10, max_iter=config['max_iter'], trial_length=config['min_trial_length'], GPFA=True)
    sessionTest = vi.infer(sessionTest, params=params)
    
    print(spikes)
    
    
    
    