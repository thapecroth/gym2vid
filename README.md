# gym2vid

A Python package for training reinforcement learning agents and recording their gameplay videos with state and action annotations.

## Installation

```bash
pip install gym2vid
```

## Features

- Train RL agents on Gymnasium environments
- Record gameplay videos
- Annotate videos with state and action information
- Slow down videos for better visualization
- Support for custom environments

## Quick Start

```python
from gym2vid import Runner

# Initialize the runner with an environment
runner = Runner("FrozenLake-v1")

# Train and record episodes
runner.train_and_record(num_episodes=10, output_dir="./videos")

# Create annotated video
runner.create_annotated_video(slow_factor=2.0)
```

## Documentation

### Runner Class

The main interface for training and recording:

- `train_and_record(num_episodes, output_dir)`: Train the agent and record episodes
- `create_annotated_video(slow_factor)`: Create an annotated video with state/action information

### Configuration

You can customize the training parameters:

```python
config = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 0.1
}
runner = Runner("FrozenLake-v1", config=config)
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
