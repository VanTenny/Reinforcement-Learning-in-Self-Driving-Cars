# Optimized Monte Carlo Control for FrozenLake
import gym
import numpy as np
import matplotlib.pyplot as plt 

env = gym.make('FrozenLake-v1', is_slippery=False)

EPISODES = 2000 
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.999

# Pre-allocated numpy arrays instead of defaultdict
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
returns_sum = np.zeros((num_states, num_actions))
returns_count = np.zeros((num_states, num_actions))
policy = np.random.randint(num_actions, size=num_states)  # Array-based policy

rewards = []

for episode in range(EPISODES):
    episode_history = []
    state = env.reset()[0]
    done = False
    
    # Epsilon-greedy action selection
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = policy[state]
        
        next_state, reward, done, truncated, info = env.step(action)
        episode_history.append((state, action, reward))
        state = next_state

    # Monte Carlo update
    G = 0
    visited = set()
    for t in reversed(range(len(episode_history))):
        state, action, reward = episode_history[t]
        G = gamma * G + reward
        
        if (state, action) not in visited:
            returns_sum[state, action] += G
            returns_count[state, action] += 1
            Q[state, action] = returns_sum[state, action] / returns_count[state, action]
            policy[state] = np.argmax(Q[state])
            visited.add((state, action))
    
    rewards.append(sum([step[2] for step in episode_history]))
    epsilon = max(epsilon * epsilon_decay, 0.01)  # Decay epsilon

plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
plt.title('Optimized MC Training')
plt.xlabel('Episode')
plt.ylabel('Avg Reward')
plt.savefig('monte_carlo_rewards.png')
plt.show()
