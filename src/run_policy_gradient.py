import argparse
from policy_gradient import train_policy_gradient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Policy Gradient training for CartPole with visualization window.")
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--visualization_start', type=int, default=0, help='Episode to start visualization (inclusive)')
    parser.add_argument('--visualization_end', type=int, default=10, help='Episode to end visualization (exclusive)')
    args = parser.parse_args()

    train_policy_gradient(
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        visualization_start=args.visualization_start,
        visualization_end=args.visualization_end
    )
