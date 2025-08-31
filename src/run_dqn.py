from setup_env import make_env
from dqn import train_dqn

if __name__ == "__main__":
    env = make_env()
    train_dqn(env)
    env.close()
