import numpy as np

class RBFKernel:
    def __init__(self, signal_variance = 1, length_scale = 2) -> None:
        self.signal_variance = signal_variance
        self.length_scale = length_scale

    def transform(self, x1, x2):
        return self.signal_variance**2 * np.exp(-((x1-x2)**2)/2*self.length_scale**2)
    
    def set_params(self, signal_variance, length_scale):
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        return self