
import gymnasium as gym

def run_cartpole():
    env = gym.make('CartPole-v1', render_mode="human")
    observation, info = env.reset()
    total_reward = 0
    for t in range(200):
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished after {t+1} timesteps. Total reward: {total_reward}")
            break
    env.close()

if __name__ == "__main__":
    run_cartpole()
