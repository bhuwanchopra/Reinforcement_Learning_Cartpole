
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_dqn(env, episodes=500, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, lr=1e-3, visualization_start=0, visualization_end=10):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)
    epsilon = epsilon_start
    rewards_history = []

    def select_action(state, epsilon, ep_env):
        if random.random() < epsilon:
            return ep_env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state)
            return q_values.argmax().item()


    import gymnasium as gym
    for episode in range(episodes):
        # Use rendering env for episodes in visualization range, else use non-rendering env
        if visualization_start <= episode < visualization_end:
            ep_env = gym.make('CartPole-v1', render_mode='human')
        else:
            ep_env = gym.make('CartPole-v1')
        state, _ = ep_env.reset()
        total_reward = 0
        for t in range(200):
            action = select_action(state, epsilon, ep_env)
            next_state, reward, terminated, truncated, _ = ep_env.step(action)
            done = terminated or truncated
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += float(reward)
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones)

                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break
        ep_env.close()
        rewards_history.append(total_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Visualization
    plt.figure(figsize=(12,6))
    plt.plot(rewards_history, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Rewards')
    plt.legend()
    plt.show()

