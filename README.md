# Reinforcement Learning CartPole

This project demonstrates reinforcement learning on the CartPole environment using Deep Q-Networks (DQN).

## Setup

1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

2. (Optional) Create and activate a Python virtual environment:
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

## Usage

To run DQN training and visualize CartPole for selected episodes:

```bash
python src/run_dqn.py --visualization_start 5 --visualization_end 15
```

This will show CartPole visualization for episodes 5 to 14. Adjust the parameters as needed.

## Project Structure

- `src/dqn.py`: DQN agent and training logic
- `src/run_dqn.py`: Main script, accepts visualization window as arguments
- `src/setup_env.py`: Environment setup utility
- `requirements.txt`: Python dependencies

## Visualization

The CartPole window will appear for episodes in the specified visualization range. Training reward progress is also plotted at the end of training.

## License

See `LICENSE` for details.