import numpy as np


class EpsilonGreedy:
    def __init__(self, dim):
        self.dim = dim
        self.y = np.zeros(dim)
        self.n = np.zeros(dim)

    def act(self, eps=0):
        if np.random.uniform() < eps:
            u = np.random.randint(0, self.dim)
        else:
            u = np.argmax(self.y / (self.n + np.finfo(float).eps))

        return u

    def update(self, u, r):
        self.y[u] += r
        self.n[u] += 1

    def total_reward(self):
        return np.sum(self.y)
