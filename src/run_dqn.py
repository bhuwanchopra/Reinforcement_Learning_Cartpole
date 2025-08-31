from setup_env import make_env
from dqn import train_dqn

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DQN training for CartPole with visualization window.")
    parser.add_argument('--visualization_start', type=int, default=0, help='Episode to start visualization (inclusive)')
    parser.add_argument('--visualization_end', type=int, default=10, help='Episode to end visualization (exclusive)')
    args = parser.parse_args()

    env = make_env()
    train_dqn(env, visualization_start=args.visualization_start, visualization_end=args.visualization_end)
    env.close()
