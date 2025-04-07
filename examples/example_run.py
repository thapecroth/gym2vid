"""Example script demonstrating the usage of gym2vid library.

This script shows how to:
1. Set up a gym environment
2. Initialize the runner
3. Record multiple episodes
4. Handle the recording process properly

Example:
    $ python example_run.py
"""

import gymnasium as gym
from gym2vid import Runner


def main():
    """Run the example recording process."""
    env_name = 'CartPole-v1'
    
    # Initialize the runner with the environment name
    runner = Runner(
        env_name=env_name,
        config={
            'output_dir': 'recordings/cartpole_example',
            'fps': 30,
            'render_width': 640,
            'render_height': 480
        }
    )

    try:
        # Train and record episodes
        runner.train_and_record(
            num_episodes=3,
            output_dir='recordings/cartpole_example',
            train_timesteps=10000,  # Number of timesteps to train for
            n_train_envs=2  # Number of parallel environments for training
        )
    except Exception as e:
        print(f"Error: {e}")

    print("\nRecording completed! Check the 'recordings/cartpole_example' directory for the output videos.")


if __name__ == "__main__":
    main() 