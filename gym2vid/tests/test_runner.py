import os
import pytest
import shutil
from gym2vid import Runner
import tempfile
from typing import Dict, Any
import pickle


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def basic_runner():
    """Fixture for a basic Runner instance."""
    return Runner("CartPole-v1")


@pytest.fixture
def trained_runner(temp_output_dir, basic_runner):
    """Fixture for a Runner that has been trained."""
    basic_runner.train_and_record(
        num_episodes=1, output_dir=temp_output_dir, train_timesteps=100, n_train_envs=1
    )
    return basic_runner


@pytest.mark.parametrize(
    "env_name,config",
    [
        ("CartPole-v1", {}),
        ("CartPole-v1", {"learning_rate": 0.001, "gamma": 0.99}),
        ("FrozenLake-v1", {}),
    ],
)
def test_runner_initialization(env_name: str, config: Dict[str, Any]):
    """Test Runner initialization with different environments and configs."""
    runner = Runner(env_name, config=config)

    assert runner.env_name == env_name
    assert runner.config == config
    assert runner.model_path is None
    assert runner.output_dir is None


@pytest.mark.parametrize(
    "train_params",
    [
        {"num_episodes": 1, "train_timesteps": 100, "n_train_envs": 1},
        {"num_episodes": 2, "train_timesteps": 200, "n_train_envs": 2},
    ],
)
def test_train_and_record(temp_output_dir, basic_runner, train_params):
    """Test training and recording with different parameters."""
    basic_runner.train_and_record(output_dir=temp_output_dir, **train_params)

    # Check model creation
    model_dir = os.path.join(temp_output_dir, "model")
    assert os.path.exists(model_dir)
    model_file = os.path.join(model_dir, "trpo_CartPole-v1.zip")
    assert os.path.exists(model_file)

    # Check video creation
    video_dir = os.path.join(temp_output_dir, "videos", "CartPole-v1")
    assert os.path.exists(video_dir)

    # Check that there's one MP4 and one PKL file per episode
    video_files = os.listdir(video_dir)
    mp4_files = [f for f in video_files if f.endswith(".mp4") and "annotated" not in f]
    pkl_files = [f for f in video_files if f.endswith(".pkl")]
    
    assert len(mp4_files) == train_params["num_episodes"]
    assert len(pkl_files) == train_params["num_episodes"]
    
    # Check that each MP4 file has a corresponding PKL file with matching timesteps
    for episode_id in range(train_params["num_episodes"]):
        mp4_path = os.path.join(
            video_dir, f"CartPole-v1_episodes_{episode_id}.mp4"
        )
        pkl_path = os.path.join(
            video_dir, f"CartPole-v1_episodes_{episode_id}.pkl"
        )
        
        assert os.path.exists(mp4_path), f"MP4 file for episode {episode_id} not found"
        assert os.path.exists(pkl_path), f"PKL file for episode {episode_id} not found"
        
        # Verify that the PKL file contains action and state data
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            
        assert "action_ls" in data, "PKL file missing action list"
        assert "states_ls" in data, "PKL file missing states list"
        assert "n_timesteps" in data, "PKL file missing timestep count"
        assert len(data["action_ls"]) == data["n_timesteps"], "Action list length doesn't match timestep count"
        assert len(data["states_ls"]) == data["n_timesteps"], "States list length doesn't match timestep count"


@pytest.mark.parametrize("slow_factor", [1.0, 2.0])
def test_create_annotated_video(temp_output_dir, trained_runner, slow_factor):
    """Test video annotation with different slow factors."""
    trained_runner.create_annotated_video(slow_factor=slow_factor)

    video_dir = os.path.join(temp_output_dir, "videos", "CartPole-v1")
    video_files = os.listdir(video_dir)

    # Check for annotated videos
    annotated_files = [f for f in video_files if "annotated" in f]
    assert len(annotated_files) > 0, "No annotated videos were created"

    # Check for slowed videos if slow_factor != 1.0
    if slow_factor != 1.0:
        slowed_files = [f for f in video_files if f"slowed_{slow_factor}x" in f]
        assert len(slowed_files) > 0, f"No slowed videos with factor {slow_factor}x were created"


@pytest.mark.parametrize(
    "config",
    [
        {"learning_rate": 0.001},
        {"gamma": 0.99},
        {"n_steps": 128},
        {"learning_rate": 0.001, "gamma": 0.99, "n_steps": 128},
    ],
)
def test_custom_config(temp_output_dir, config):
    """Test different custom configurations."""
    runner = Runner("CartPole-v1", config=config)
    runner.train_and_record(
        num_episodes=1, output_dir=temp_output_dir, train_timesteps=100, n_train_envs=1
    )

    assert os.path.exists(os.path.join(temp_output_dir, "model"))
    assert runner.config == config


class TestErrorHandling:
    """Group error handling test cases."""

    def test_create_annotated_video_before_training(self, basic_runner):
        """Test error when creating annotated video before training."""
        with pytest.raises(ValueError, match="No output directory set"):
            basic_runner.create_annotated_video()

    @pytest.mark.parametrize(
        "invalid_env",
        [
            "InvalidEnv-v1",
            "NonexistentEnv-v0",
            "Unknown-v1",
        ],
    )
    def test_invalid_environment(self, invalid_env):
        """Test error handling for invalid environments."""
        with pytest.raises(Exception):
            runner = Runner(invalid_env)
            runner.train_and_record(num_episodes=1, output_dir="./temp")


if __name__ == "__main__":
    pytest.main([__file__])
