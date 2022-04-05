import numpy as np
import matplotlib.pyplot as plt

class saddlenode():
    def __init__(self, N=0.75, t_max=250):
        params={
                "r": 0.7,
                "K": 1.5,
                "sigma": 0.01,
                "q": 3,
                "b": 0.15,
                "a": 0.19,
                "M": 1.2,
                "x0": 0.8,
                "alpha": 0.001,
                "beta": 1.0,
            },
        self.replicates = 0
        self.t_max = t_max
        self.mu = -5e-4
        self.sigma = params["sigma"]
        self.r = -0.0625
        self.N_init = params["x0"]
        self.N = N
        self.offset = 0.5
        self.reps = 0

    def may(x, params, size=1):
        with np.errstate(divide="ignore"):
            r = params["r"]
            M = params["M"]
            a = params["a"]
            q = params["q"]
            b = params["b"]
            exp_mu = (
                x
                + x * r * (1 - x / M)
                - a * np.power(x, q) / (np.power(x, q) + np.power(b, q))
            )
            mu = np.log(np.clip(exp_mu, 0, np.inf))
        state = np.random.lognormal(mu, params["sigma"], size)
        return np.maximum(0, state)
vim 
    def step(self):
        self.eta = np.random.normal(self.mu, self.sigma)
        self.N = self.may(self.N, params)
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
