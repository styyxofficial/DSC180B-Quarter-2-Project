import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import inv, det
from scipy.optimize import minimize
from data.data_prep import RBFKernel


class GP:
    def __init__(self, kernel) -> None:
        self.kernel = kernel
        pass
    
    def train(self, X_train, y_train, X_test):
        noise_variance = 0.1
        
        
        def nll(theta):
            kern = RBFKernel(theta[0], theta[1])
            noise_var = theta[2]
            cov_X_X = self.covariance(X_train, X_train, kern)
            noise_cov = noise_var*np.identity(n=len(cov_X_X))
            # cov_Xt_X = covariance(X_test, X_train, kern)
            # cov_Xt_Xt = covariance(X_test, X_test, kern)
            k_inv = inv(cov_X_X + noise_cov)

            # args(K + noise_cov, Y_train, k_inv, X_train)
            return 0.5 * np.log(det(cov_X_X+noise_cov)) + \
                   0.5 * (y_train - self.mean(X_train)).T @ k_inv @ (y_train-self.mean(X_train)) + \
                   0.5 * len(X_train) * np.log(2*np.pi)

        res = minimize(nll, [1, 1, 0.1],
               bounds=((1e-5, None), (1e-5, None), (1e-5, None)),
               method='L-BFGS-B')
        
        sv, ls, nv = res.x


        self.kernel = RBFKernel(sv, ls)
        cov_X_X = self.covariance(X_train, X_train, self.kernel)
        cov_Xt_X = self.covariance(X_test, X_train, self.kernel)
        cov_Xt_Xt = self.covariance(X_test, X_test, self.kernel)
        k_inv = inv(cov_X_X + nv*np.identity(n=len(cov_X_X)))
        
        mu =  self.mean(X_test) + cov_Xt_X @ k_inv @ (y_train-self.mean(X_train))
        sigma = cov_Xt_Xt - cov_Xt_X @ k_inv @ cov_Xt_X.T
        cond_num = np.linalg.cond(cov_X_X+ noise_variance*np.identity(n=len(cov_X_X))) # lower is better
        
        return mu, sigma, sv, ls, nv, cond_num # best_signal_variance, best_length_scale, best_noise_variance, condition number
    

    def prob(self, y_train, mean_x, noise_cov):
        y = multivariate_normal.rvs(mean=mean_x, cov=noise_cov)
        jitter = np.eye(len(noise_cov)) * 1e-6
        prob = multivariate_normal.pdf(y_train, mean=mean_x, cov=noise_cov + jitter)
        return prob
    
    def mean(self, X):
        # We assume 0 mean
        return np.zeros(shape=len(X))

    def covariance(self, X1, X2, kernel):
        cov = np.zeros(shape=(len(X1), len(X2)))

        for i in range(len(X1)):
            for j in range(len(X2)):
                cov[i][j] = kernel.transform(X1[i], X2[j])
                
        return cov