import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

def train_policy_gradient(episodes=500, gamma=0.99, lr=1e-3, visualization_start=0, visualization_end=10):
    from setup_env import make_env
    temp_env = make_env()
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()
    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    rewards_history = []

    for episode in range(episodes):
        # Visualization window
        if visualization_start <= episode < visualization_end:
            ep_env = gym.make('CartPole-v1', render_mode='human')
        else:
            ep_env = gym.make('CartPole-v1')
        state, _ = ep_env.reset()
        log_probs = []
        rewards = []
    total_reward = 0.0
        for t in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state_tensor)
            action = torch.multinomial(probs, num_samples=1).item()
            log_prob = torch.log(probs.squeeze(0)[action])
            next_state, reward, terminated, truncated, _ = ep_env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            total_reward += float(reward)
            if terminated or truncated:
                break
        ep_env.close()
        rewards_history.append(total_reward)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
    loss = torch.tensor(0.0)
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    # Visualization
    plt.figure(figsize=(12,6))
    plt.plot(rewards_history, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Policy Gradient Training Rewards')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_policy_gradient()
