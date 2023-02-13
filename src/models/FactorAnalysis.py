import numpy as np
from numpy.linalg import inv, det
import pandas as pd

class FactorAnalysisModel:
    def __init__(self, config):
        self.latent_factors = config['latent_factors']
        self.epochs = config['epochs'] # Number of iterations to run EM
        return
        
    def train(self, X):
        
        M = self.latent_factors  # Number of latent variables
        
        N = len(X)
        mu_x = np.mean(X, axis=0).reshape(-1, 1)
        W_L = np.random.uniform(low=-5, high=5, size=(X.shape[1], M))
        psi_L = np.diag(np.random.uniform(low=0, high=1, size=X.shape[1]))
        likelihoods = np.zeros(shape=self.epochs)

        for i in range(self.epochs):
            # Expectation Step
            Ez_L = np.empty(X.shape[0], dtype=object)
            Ezzt_L = np.empty(X.shape[0], dtype=object)
            G = inv(np.eye(M) + W_L.T @ inv(psi_L) @ W_L)
            
            for j in range(0, X.shape[0]):
                Ez_L[j] = G @ W_L.T @ inv(psi_L) @ (X[j].reshape(-1, 1) - mu_x)
                Ezzt_L[j] = G + Ez_L[j] @ Ez_L[j].T
                
                
            # Compute new W

            w_sum1 = np.zeros(shape=(X.shape[1], M))

            for j in range(X.shape[0]):
                w_sum1 += (X[j].reshape(-1, 1) - mu_x) @ Ez_L[j].T

            W_L = w_sum1 @ inv(Ezzt_L.sum(axis=0))
            
            
            # Compute new psi. Assuming S is the sample covariance.

            psi_sum1 = np.zeros(shape=(M, X.shape[1]))

            for j in range(X.shape[0]):
                psi_sum1 += Ez_L[j] @ (X[j].reshape(-1, 1) - mu_x).T
                
            psi_L = np.diag(np.diag(np.cov(X.T) - (1/N) * W_L @ psi_sum1))
            
            likelihoods[i] = self.likelihood(X, W_L, psi_L)
    
        return W_L, likelihoods
    
    def likelihood(self, X, W_L, psi_L):
        C = W_L @ W_L.T + psi_L
        return -(len(W_L)/2) * (np.log(det(C)) + np.trace(np.cov(X.T) * inv(C)))