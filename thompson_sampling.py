import numpy as np


class ThompsonSampling:
    def __init__(self, dim):
        self.dim = dim
        self.a = np.ones(dim)
        self.b = np.ones(dim)

    def act(self):
        samples = np.random.beta(self.a, self.b, self.dim)
        return np.argmax(samples)

    def update(self, u, r):
        self.a[u] += r == 1
        self.b[u] += r == 0

    def total_reward(self):
        return np.sum(self.a) - self.dim
