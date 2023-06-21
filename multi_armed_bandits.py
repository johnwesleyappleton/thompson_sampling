import numpy as np
import matplotlib.pyplot as plt
from bandits import Bandits
from thompson_sampling import ThompsonSampling
from epsilon_greedy import EpsilonGreedy


# initialize agents
num_bandits = 5
num_agents = 3
eps = 0.01
bandits = Bandits(num_bandits)
thompson_sampling_agents = [ThompsonSampling(num_bandits) for _ in range(num_agents)]
epsilon_greedy_agents = [EpsilonGreedy(num_bandits) for _ in range(num_agents)]

# run the simulation
num_iter = 10000
thompson_sampling_avg_rewards = np.zeros(num_iter)
thompson_sampling_avg_optimal = np.zeros(num_iter)
thompson_sampling_avg_regret = np.zeros(num_iter)
epsilon_greedy_avg_rewards = np.zeros(num_iter)
epsilon_greedy_avg_optimal = np.zeros(num_iter)
epsilon_greedy_avg_regret = np.zeros(num_iter)
for i in range(num_iter):
    # print iteration
    if (i+1) % 1000 == 0:
        print('Iteration', i+1)

    # get actions
    thompson_sampling_actions = [agent.act() for agent in thompson_sampling_agents]
    epsilon_greedy_actions = [agent.act(eps) for agent in epsilon_greedy_agents]

    # get optimal choices
    optimal = bandits.optimal_bandit()
    thompson_sampling_is_optimal = [action == optimal for action in thompson_sampling_actions]
    epsilon_greedy_is_optimal = [action == optimal for action in epsilon_greedy_actions]

    # track average optimal choices
    idx = max(0, i-1)
    thompson_sampling_avg_optimal[i] = thompson_sampling_avg_optimal[idx] + sum(thompson_sampling_is_optimal) / num_agents
    epsilon_greedy_avg_optimal[i] = epsilon_greedy_avg_rewards[idx] + sum(epsilon_greedy_is_optimal) / num_agents

    # get results
    thompson_sampling_samples = [bandits.reward(action) for action in thompson_sampling_actions]
    epsilon_greedy_samples = [bandits.reward(action) for action in epsilon_greedy_actions]

    # update results
    for j in range(num_agents):
        thompson_sampling_agents[j].update(thompson_sampling_actions[j], thompson_sampling_samples[j])
        epsilon_greedy_agents[j].update(epsilon_greedy_actions[j], epsilon_greedy_samples[j])

    # get total rewards
    thompson_sampling_rewards = [agent.total_reward() for agent in thompson_sampling_agents]
    epsilon_greedy_rewards = [agent.total_reward() for agent in epsilon_greedy_agents]

    # track average total rewards
    thompson_sampling_avg_rewards[i] = sum(thompson_sampling_rewards) / num_agents
    epsilon_greedy_avg_rewards[i] = sum(epsilon_greedy_rewards) / num_agents

    # get regret
    expected_reward = bandits.expected_reward()
    thompson_sampling_regret = [expected_reward - sample for sample in thompson_sampling_samples]
    epsilon_greedy_regret = [expected_reward - sample for sample in epsilon_greedy_samples]

    # track average total regret
    idx = max(0, i-1)
    thompson_sampling_avg_regret[i] = thompson_sampling_avg_regret[idx] + sum(thompson_sampling_regret) / num_agents
    epsilon_greedy_avg_regret[i] = epsilon_greedy_avg_regret[idx] + sum(epsilon_greedy_regret) / num_agents


# plot average rewards
plt.plot(thompson_sampling_avg_rewards / (np.arange(num_iter)+1), label='Thompson Sampling')
plt.plot(epsilon_greedy_avg_rewards / (np.arange(num_iter)+1), label='Epsilon Greedy')
plt.title('Comparison 3: Average Reward')
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.legend()
plt.show()

# plot percent optimal choices
plt.plot(thompson_sampling_avg_optimal / (np.arange(num_iter)+1), label='Thompson Sampling')
plt.plot(epsilon_greedy_avg_optimal / (np.arange(num_iter)+1), label='Epsilon Greedy')
plt.title('Comparison 2: Percentage of Optimal Actions')
plt.xlabel('Iteration')
plt.ylabel('Percent Optimal Actions')
plt.legend()
plt.show()

# plot cumulative regret
plt.plot(thompson_sampling_avg_regret, label='Thompson Sampling')
plt.plot(epsilon_greedy_avg_regret, label='Epsilon Greedy')
plt.title('Comparison 1: Cumulative Regret')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.show()
