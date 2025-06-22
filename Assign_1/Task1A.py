#Tabular Q-learning with Epsillon Greedy Exploration
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', is_slippery=False)

EPISODES = 10000
lr = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.999

Q = np.zeros([env.observation_space.n, env.action_space.n])
rewards = []

for episode in range(EPISODES):
    state = env.reset()[0]
    total_reward = 0
    done = False
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done, truncated, info = env.step(action)
        
        Q[state, action] += lr * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        total_reward += reward
        state = next_state

    epsilon = max(epsilon * epsilon_decay, 0.01)
    rewards.append(total_reward)

plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
plt.title('Q-Learning Training Progress')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
