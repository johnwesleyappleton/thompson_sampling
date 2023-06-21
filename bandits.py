import numpy as np


class Bandits:
    def __init__(self, n):
        self.bandits = np.random.uniform(size=n)

    def reward(self, u):
        p = self.bandits[u]
        return np.random.binomial(1, p)

    def optimal_bandit(self):
        return np.argmax(self.bandits)

    def expected_reward(self):
        return np.max(self.bandits)
