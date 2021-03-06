import numpy as np
import matplotlib.pyplot as plt
import copy

class stochastic_tp():
    def __init__(self, N=0.75, t_max=250):
        self.t_max = t_max
        self.mu = 0
        self.sigma = 1e-2
        self.N_init = N
        self.N = N
        self.reps = 0
        self.x1 = 0.75
        self.x2 = 0.25

    def step(self):
        self.eta = np.random.normal(self.mu, self.sigma)
        self.N += (self.x1 - self.N)**2 * (self.x2 - self.N) + self.eta
        self.N = np.clip(self.N, 0, 1)
        return self.N

    def draw_replicate(self):
        self.reset()
        return np.array([self.step() for t in range(self.t_max)])
    
    def collect_samples(self, reps):
        self.reps += reps
        self.samples = np.array([self.draw_replicate() for rep in range(reps)])
        return self.samples
    
    def reset(self):
        self.N = self.N_init
    
    def plot(self, file_name):
        for idx, sample in enumerate(self.samples):
            plt.plot(np.linspace(1, self.t_max, self.t_max), sample, alpha=0.1, color="b")
        plt.savefig(file_name)

class saddle_node_tp():
    def __init__(self, N=0.75, t_max=250, alpha=0.0015, h=0.15):
        self.t_max = t_max
        self.mu = 0
        self.sigma = 0.02
        self.N_init = N
        self.N = N
        self.reps = 0
        self.r = 1
        self.K = 1
        self.alpha = alpha
        self.h = h
        self.h_init = h
        self.s = 0.1

    def step(self):
        self.eta = np.random.normal(self.mu, self.sigma)
        self.N += self.r * self.N * (1 - self.N / self.K) - self.h * (self.N**2 / (self.s**2 + self.N**2)) + self.eta
        self.h += self.alpha
        self.h = np.clip(self.h, 0, 0.27)
        self.N = np.clip(self.N, 0, 1)
        
        return self.N

    def draw_replicate(self, random_alpha):
        self.reset()
        if random_alpha:
            self.alpha = np.random.uniform(0.0004, 0.0015)
        return np.array([self.step() for t in range(self.t_max)])
    
    def collect_samples(self, reps, random_alpha=False):
        self.reps = reps
        self.samples = np.array([self.draw_replicate(random_alpha) for rep in range(reps)])
        return self.samples
    
    def reset(self):
        self.N = self.N_init
        self.h = self.h_init
    
    def plot(self, file_name):
        for idx, sample in enumerate(self.samples):
            plt.plot(np.linspace(1, self.t_max, self.t_max), sample, alpha=0.1, color="b")
        plt.savefig(file_name)

class hopf_tp():
    def __init__(self, N=[9, 1], t_max=100, delta=-0.01, K_init=14):
        self.t_max = t_max
        self.mu = 0
        self.sigma = .02
        self.N_init = N
        self.N = N
        self.reps = 0
        self.r = 0.75
        self.K_init = K_init
        self.K = K_init
        self.delta = delta
        self.c = 0.1

    def step(self):
        self.K += self.delta
        self.eta = np.random.normal(self.mu, self.sigma, 2)
        g = self.N[0] * np.exp(self.r * (1 - self.N[0] / self.K))
        self.N[0] = g * np.exp(- self.c * self.N[1] + self.eta[0])
        self.N[1] = g * (1 - np.exp(-self.c * self.N[1] + self.eta[1]))
        
        self.N = np.clip(self.N, 0 , 100)
        return self.N

    def draw_replicate(self):
        self.reset()
        return np.array([self.step() for t in range(self.t_max)])
    
    def collect_samples(self, reps):
        self.reps += reps
        self.samples = np.array([self.draw_replicate() for rep in range(reps)])
        
        return self.samples
    
    def reset(self):
        self.N = self.N_init
        self.K = self.K_init
        
    
    def plot(self, file_name):
        for idx, sample in enumerate(self.samples):
            plt.plot(np.linspace(1, self.t_max, self.t_max), sample[:,0], alpha=0.1, color="b")
            plt.plot(np.linspace(1, self.t_max, self.t_max), sample[:,1], alpha=0.1, color="r")
        plt.savefig(file_name)
