import gymnasium as gym

def make_env():
    """Create and return the CartPole environment with human render mode."""
    return gym.make('CartPole-v1', render_mode='human')

if __name__ == "__main__":
    env = make_env()
    print("Environment created successfully.")
    env.close()
