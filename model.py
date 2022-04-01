import numpy as np
import matplotlib.pyplot as plt

class tipping_point():
    def __init__(self, N=0.75, t_max=250):
        self.replicates = 0
        self.t_max = t_max
        self.mu = -5e-4
        self.sigma = 1e-2
        self.r = -0.0625
        self.N_init = N
        self.N = N
        self.offset = 0.5
        self.reps = 0

    def step(self):
        self.eta = np.random.normal(self.mu, self.sigma)
        self.N += (0.75 - self.N)**2 * (0.25 - self.N) + self.eta
        self.N = np.clip(self.N, 0, 1)
        return self.N

    def draw_replicate(self):
        self.reset()
        return np.array([self.step() for t in range(self.t_max)])
    
    def collect_samples(self, reps):
        self.reps = reps
        self.samples = np.array([self.draw_replicate() for rep in range(reps)])
        return self.samples
    
    def reset(self):
        self.N = self.N_init
    
    def plot(self, file_name):
        for idx, sample in enumerate(self.samples):
            plt.plot(np.linspace(1, self.t_max, self.t_max), sample, alpha=0.1, color="b")
        plt.savefig(file_name)
