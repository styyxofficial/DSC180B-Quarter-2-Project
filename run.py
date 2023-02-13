import sys
import json
from src.models.FactorAnalysis import FactorAnalysisModel
from src.models.GaussianProcess import GP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from src.data.RBFKernel import RBFKernel
import shutil

ROOT_STATS_DIR = './output'

def record_fa_stats(config, weights):
    with open(os.path.join(ROOT_STATS_DIR, config['experiment_name'], 'weights.txt'), "w") as outfile:
        outfile.write(np.array2string(weights))
        
def plot_fa_stats(config, X, y):
    plt.figure()
    plt.plot(X, y, label="Negative Log Likelihood")
    plt.xlabel("Epochs")
    plt.ylabel("Negative Log Likelihood")
    plt.title(config['experiment_name']+" Stats Plot")
    plt.savefig(os.path.join(ROOT_STATS_DIR, config['experiment_name'], 'likelihood.png'))
    plt.close()
         
def get_factor_loadings(config):
    # If the model has not been run, run it
    if os.path.exists(os.path.join(ROOT_STATS_DIR, config['experiment_name'])) ==False:
        model = FactorAnalysisModel(config)
        X_train = pd.read_csv(config['training_data_path']).to_numpy()   # Read data
        factor_loadings, likelihoods = model.train(X_train)
        
        os.makedirs(os.path.join(ROOT_STATS_DIR, config['experiment_name']))
        plot_fa_stats(config, np.arange(config['epochs']), likelihoods)
        record_fa_stats(config, factor_loadings)

def plot_gpfa(config, X_train, y_train, X_test, mu, sigma):
    fig, ax1 = plt.subplots(
    nrows = 1, ncols=1, figsize=(6, 6))

    # Plot the correct distribution
    ax1.plot(X_train, y_train, 'ko', linewidth=2, label='$(x_{train}, y_{train})$')

    # Plot the posterior
    ax1.plot(X_test, mu, 'r-', lw=2, label='$\mu$')
    ax1.fill_between(X_test, mu-2*np.sqrt(np.diag(sigma)), mu+2*np.sqrt(np.diag(sigma)), color='red', alpha=0.15, label='$2 \sigma$')


    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data.')
    ax1.axis([X_train.min()-2, X_train.max()+2, -3, 3])
    ax1.legend()
    fig.savefig(os.path.join(ROOT_STATS_DIR, config['experiment_name'], 'gp_regression.png'))

def record_gpfa_params(config, mu, sigma, sv, ls, nv, cond_num):
    dictionary = {'mean' : mu.tolist(),
                  'variance' : sigma.tolist(),
                  'signal_variance' : sv,
                  'length_scale' : ls,
                  'noise_variance' : nv,
                  'condition_number' : cond_num}
    
    json_object = json.dumps(dictionary, indent=4)

    with open(os.path.join(ROOT_STATS_DIR, config['experiment_name'], 'params.json'), "w") as outfile:
        outfile.write(json_object)
    return

def get_gp_regression(config):
    # get data
    # train gp
    # get plots and parameters. Save them
    if os.path.exists(os.path.join(ROOT_STATS_DIR, config['experiment_name'])) ==False:
        # Get data
        df = pd.read_csv(config['training_data_path'])   # Read data
        model = GP(RBFKernel())
        X_test = np.linspace(start=df['X_train'].min()-2, stop=df['X_train'].max()+2, num=config['num_test_locations'])
        
        # Train model
        mu, sigma, sv, ls, nv, cond_num = model.train(df['X_train'].to_numpy(), df['y_train'].to_numpy(), X_test)

        # Save output
        os.makedirs(os.path.join(ROOT_STATS_DIR, config['experiment_name']))
        plot_gpfa(config, df['X_train'], df['y_train'], X_test, mu, sigma)
        record_gpfa_params(config, mu, sigma, sv, ls, nv, cond_num)
    return

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            config = json.load(open('config/testdata.json'))
            get_factor_loadings(config)
        elif sys.argv[1] == 'gp':
            config = json.load(open('config/gp.json'))
            get_gp_regression(config)
        elif sys.argv[1] == 'fa':
            config = json.load(open('config/fa.json'))
            get_factor_loadings(config)
        elif sys.argv[1] == 'gpfa':
            # Run both GP and FA
            config = json.load(open('config/fa.json'))
            get_factor_loadings(config)
            
            config = json.load(open('config/gp.json'))
            get_gp_regression(config)
        elif sys.argv[1] == 'all':
            # Run both GP and FA
            config = json.load(open('config/fa.json'))
            get_factor_loadings(config)
            
            config = json.load(open('config/gp.json'))
            get_gp_regression(config)
        elif sys.argv[1] == 'clean':
            shutil.rmtree(os.path.join(ROOT_STATS_DIR), ignore_errors=True)
            os.makedirs(os.path.join(ROOT_STATS_DIR))
    else:
        # run gpfa
        config = json.load(open('config/fa.json'))
        get_factor_loadings(config)
        
        config = json.load(open('config/gp.json'))
        get_gp_regression(config)
        
