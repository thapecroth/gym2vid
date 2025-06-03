import gymnasium as gym
import cv2
import pickle
import os
from tqdm import tqdm
from typing import List, Literal, get_args, Dict, Any, Optional
import fire
import numpy as np
import multiprocessing

# Handle optional ML dependencies gracefully for documentation generation
try:
    from stable_baselines3.common.evaluation import evaluate_policy
    from sb3_contrib import TRPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from multiprocessing import Pool
    import torch

    ML_DEPENDENCIES_AVAILABLE = True
except ImportError:
    # Create mock classes/functions for documentation generation
    class MockTRPO:
        @staticmethod
        def load(*args, **kwargs):
            raise RuntimeError(
                "ML dependencies not available. Install with: pip install torch stable-baselines3 sb3-contrib"
            )

    class MockPool:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ML dependencies not available. Install with: pip install torch stable-baselines3 sb3-contrib"
            )

    TRPO = MockTRPO
    Pool = MockPool
    torch = None
    ML_DEPENDENCIES_AVAILABLE = False

import traceback
from .annotate import create_annotated_video, create_slowed_video

multiprocessing.set_start_method("spawn", force=True)
# N_GPUS = torch.cuda.device_count() if torch else 0
N_GPUS = 3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch else "cpu"


# python list of famous openai gym environments
list_of_gym_environments = Literal[
    "ALL",
    "CartPole-v1",
    "Acrobot-v1",
    "FrozenLake-v1",
    # "ALE/Adventure-v5",
    # "ALE/AirRaid-v5",
    # "ALE/Alien-v5",
    # "ALE/Amidar-v5",
    # "ALE/Assault-v5",
    # "ALE/Asterix-v5",
    # "ALE/Asteroids-v5",
    # "ALE/Atlantis-v5",
    # "ALE/BankHeist-v5",
    # "ALE/BattleZone-v5",
    # "ALE/BeamRider-v5",
    # "ALE/Berzerk-v5",
    # "ALE/Bowling-v5",
    # "ALE/Boxing-v5",
    # "ALE/Breakout-v5",
    # "ALE/Carnival-v5",
    # "ALE/Centipede-v5",
    # "ALE/ChopperCommand-v5",
    # "ALE/CrazyClimber-v5",
    # "ALE/Defender-v5",
    # "ALE/DemonAttack-v5",
    # "ALE/DoubleDunk-v5",
    # "ALE/ElevatorAction-v5",
    # "ALE/Enduro-v5",
    # "ALE/FishingDerby-v5",
    # "ALE/Freeway-v5",
    # "ALE/Frostbite-v5",
    # "ALE/Gopher-v5",
    # "ALE/Gravitar-v5",
    # "ALE/Hero-v5",
    # "ALE/IceHockey-v5",
    # "ALE/Jamesbond-v5",
    # "ALE/JourneyEscape-v5",
    # "ALE/Kangaroo-v5",
    # "ALE/Krull-v5",
    # "ALE/KungFuMaster-v5",
    # "ALE/MontezumaRevenge-v5",
    # "ALE/MsPacman-v5",
    # "ALE/NameThisGame-v5",
    # "ALE/Phoenix-v5",
    # "ALE/Pitfall-v5",
    # "ALE/Pong-v5",
    # "ALE/Pooyan-v5",
    # "ALE/PrivateEye-v5",
    # "ALE/Qbert-v5",
    # "ALE/Riverraid-v5",
    # "ALE/RoadRunner-v5",
    # "ALE/Robotank-v5",
    # "ALE/Seaquest-v5",
    # "ALE/Skiing-v5",
    # "ALE/Solaris-v5",
    # "ALE/SpaceInvaders-v5",
    # "ALE/StarGunner-v5",
    # "ALE/Tennis-v5",
    # "ALE/TimePilot-v5",
    # "ALE/Tutankham-v5",
    # "ALE/UpNDown-v5",
    # "ALE/Venture-v5",
    # "ALE/VideoPinball-v5",
    # "ALE/WizardOfWor-v5",
    # "ALE/YarsRevenge-v5",
    # "ALE/Zaxxon-v5",
]


def simulate(
    env_name: list_of_gym_environments,
    model_path: str,
    episode: int,
    cuda_idx: int,
    output_dir: str = None,
):
    import gymnasium as gym
    import cv2

    # Force CPU for TRPO as it's not optimized for GPU
    model = TRPO.load(model_path, device="cpu")

    def add_text_to_frame(
        frame: np.ndarray, obs: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX  # type: ignore
        font_scale = 0.5
        font_color = (0, 0, 0)  # Black color
        line_type = 2

        # Convert observation and action to strings
        obs_text = f"Obs: {obs.round(2)}"
        action_text = f"Action: {action.tolist()}"

        # Add observation text
        cv2.putText(frame, obs_text, (10, 30), font, font_scale, font_color, line_type)  # type: ignore

        # Add action text
        cv2.putText(frame, action_text, (10, 50), font, font_scale, font_color, line_type)  # type: ignore

        return frame

    try:
        video_dir = (
            os.path.join(output_dir, "videos", env_name)
            if output_dir
            else os.path.join("videos", env_name)
        )
        os.makedirs(video_dir, exist_ok=True)
        env = gym.make(env_name, render_mode="rgb_array")
        env.metadata["render_fps"] = 24
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        episode_name = f"{env_name.replace('/', '_')}_episodes_{episode}"
        action_ls, states_ls = [], []

        # Initialize video writer
        frame = env.render()
        assert isinstance(frame, np.ndarray), "Rendered frame must be a numpy array"
        height, width, layers = frame.shape  # type: ignore
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        video_path = os.path.join(video_dir, f"{episode_name}.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, 24, (width, height))  # type: ignore
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

        frame_counter = 0

        while not done and frame_counter < 240:
            action, _states = model.predict(obs)
            frame = env.render()
            assert isinstance(frame, np.ndarray), "Rendered frame must be a numpy array"
            # Write frame to video file
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # type: ignore
            video_writer.write(frame_rgb)

            action_ls.append(action)
            states_ls.append(obs)

            if env_name == "FrozenLake-v1":
                action = action.item()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            frame_counter += 1

        # Release the video writer
        video_writer.release()

        # Verify video was created successfully
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError(f"Failed to create video at {video_path}")

        assert len(action_ls) == len(
            states_ls
        ), "Action and state lists must have same length"

        pkl_path = os.path.join(video_dir, f"{episode_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "action_ls": (
                        action_ls.tolist()
                        if isinstance(action_ls, np.ndarray)
                        else action_ls
                    ),
                    "states_ls": (
                        states_ls.tolist()
                        if isinstance(states_ls, np.ndarray)
                        else states_ls
                    ),
                    "n_timesteps": (
                        action_ls.shape[0]
                        if isinstance(action_ls, np.ndarray)
                        else len(action_ls)
                    ),
                },
                f,
            )
        env.close()
        return True
    except Exception as e:
        print(f"Error in {env_name}: {e}")
        traceback.print_exc()  # Print full traceback for better debugging
        return False


def train(
    env_name: list_of_gym_environments, train_timesteps: int, n_envs: int = 8
) -> str:
    if not ML_DEPENDENCIES_AVAILABLE:
        raise RuntimeError(
            "ML dependencies not available. Install with: pip install torch stable-baselines3 sb3-contrib"
        )

    os.makedirs("model", exist_ok=True)
    save_model_name = f"model/trpo_{env_name.replace('/', '_')}"
    save_model_name_zip = save_model_name + ".zip"
    print(f"save_model_name {save_model_name}")

    # check if model exists
    if os.path.exists(save_model_name_zip):
        print(f"Model {save_model_name_zip} already exists. Skipping training.")
    else:
        print(f"Training model {save_model_name} with n_envs={n_envs}")
        env = make_vec_env(env_name, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        env.metadata["render_fps"] = [24 for _ in range(n_envs)]
        model = TRPO("MlpPolicy", env, verbose=1, device="cuda")
        # Train the model
        total_timesteps = train_timesteps
        model.learn(total_timesteps=total_timesteps)

        # Save the trained model
        model.save(save_model_name)
        print(f"saving model {save_model_name}")

        # Evaluate the trained agent
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        env.close()
        del model
    return save_model_name_zip


def main(
    env_name: list_of_gym_environments,
    train_timesteps: int,
    infer_episodes: int,
    n_train_envs: int = 2,
    n_infer_envs: int = 2,
):
    if not ML_DEPENDENCIES_AVAILABLE:
        raise RuntimeError(
            "ML dependencies not available. Install with: pip install torch stable-baselines3 sb3-contrib"
        )

    to_run_envs = [env_name]
    if env_name == "ALL":
        to_run_envs: List[list_of_gym_environments] = list(
            get_args(list_of_gym_environments)
        )[1:]

    for _env_name in to_run_envs:
        model_path = train(_env_name, train_timesteps, n_train_envs)
        assert os.path.exists(model_path), f"Model {model_path} does not exist"
        os.makedirs("videos", exist_ok=True)
        with Pool(processes=n_infer_envs) as pool:
            results = []
            with tqdm(total=infer_episodes) as pbar:

                def update_pbar(*args):
                    pbar.update()

                for episode_id in range(infer_episodes):
                    results.append(
                        pool.apply_async(
                            simulate,
                            tuple(
                                (_env_name, model_path, episode_id, episode_id % N_GPUS)
                            ),
                            callback=update_pbar,
                            error_callback=lambda x: print(f"Error in {x}"),
                        )
                    )

                for result in results:
                    result.get()


class Runner:
    """Main interface for training and recording gym environments."""

    def __init__(self, env_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the Runner.

        Args:
            env_name: Name of the Gymnasium environment
            config: Optional configuration dictionary for training parameters
        """
        if not ML_DEPENDENCIES_AVAILABLE:
            print(
                "Warning: ML dependencies (torch, stable-baselines3, sb3-contrib) not available."
            )
            print(
                "Some functionality will be limited. Install with: pip install torch stable-baselines3 sb3-contrib"
            )

        self.env_name = env_name
        self.config = config or {}
        self.model_path = None
        self.output_dir = None

    def train_and_record(
        self,
        num_episodes: int,
        output_dir: str,
        train_timesteps: int = 10000,
        n_train_envs: int = 2,
    ) -> None:
        """Train the agent and record episodes.

        Args:
            num_episodes: Number of episodes to record
            output_dir: Directory to save videos and data
            train_timesteps: Number of timesteps to train for
            n_train_envs: Number of parallel environments for training
        """
        if not ML_DEPENDENCIES_AVAILABLE:
            raise RuntimeError(
                "ML dependencies not available. Install with: pip install torch stable-baselines3 sb3-contrib"
            )

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Train the model
        model_dir = os.path.join(output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        save_model_name = os.path.join(
            model_dir, f"trpo_{self.env_name.replace('/', '_')}"
        )
        save_model_name_zip = save_model_name + ".zip"

        if not os.path.exists(save_model_name_zip):
            env = make_vec_env(
                self.env_name, n_envs=n_train_envs, vec_env_cls=SubprocVecEnv
            )
            env.metadata["render_fps"] = [24 for _ in range(n_train_envs)]

            # Apply any custom config and force CPU for TRPO
            model_kwargs = {
                "verbose": 1,
                "device": "cpu",  # Force CPU for TRPO as it's not optimized for GPU
            }

            model = TRPO("MlpPolicy", env, **model_kwargs)
            model.learn(total_timesteps=train_timesteps)
            model.save(save_model_name)
            env.close()

        self.model_path = save_model_name_zip

        # Record episodes
        video_dir = os.path.join(output_dir, "videos", self.env_name)
        os.makedirs(video_dir, exist_ok=True)

        with Pool(processes=min(num_episodes, torch.cuda.device_count() or 1)) as pool:
            results = []
            with tqdm(total=num_episodes) as pbar:

                def update_pbar(*args):
                    pbar.update()

                for episode_id in range(num_episodes):
                    results.append(
                        pool.apply_async(
                            simulate,
                            (
                                self.env_name,
                                self.model_path,
                                episode_id,
                                episode_id % (torch.cuda.device_count() or 1),
                                output_dir,
                            ),
                            callback=update_pbar,
                            error_callback=lambda x: print(f"Error: {x}"),
                        )
                    )

                # Wait for all episodes to complete and check for failures
                failed_episodes = []
                for episode_id, result in enumerate(results):
                    try:
                        if not result.get():
                            failed_episodes.append(episode_id)
                    except Exception as e:
                        print(f"Episode {episode_id} failed with error: {e}")
                        failed_episodes.append(episode_id)

                if failed_episodes:
                    print(
                        f"Warning: Episodes {failed_episodes} failed to record properly"
                    )

    def create_annotated_video(self, slow_factor: float = 1.0) -> None:
        """Create an annotated video with state/action information.

        Creates annotated videos for all episodes by overlaying state and action
        information on the recorded videos. Optionally creates slowed-down
        versions of these annotated videos.

        Args:
            slow_factor: Factor to slow down the video by. Default is 1.0 (no slowing).
                Values > 1.0 will slow the video (e.g., 2.0 means half speed).

        Raises:
            ValueError: If no output directory is set (train_and_record must be called first).
            RuntimeError: If source video files are missing or annotation fails.
        """
        if not self.output_dir:
            raise ValueError("No output directory set. Run train_and_record first.")

        video_dir = os.path.join(self.output_dir, "videos", self.env_name)
        if not os.path.exists(video_dir):
            raise RuntimeError(f"Video directory not found: {video_dir}")

        # Get list of episode files
        video_files = [
            f
            for f in os.listdir(video_dir)
            if f.endswith(".mp4") and "annotated" not in f
        ]
        if not video_files:
            raise RuntimeError(f"No video files found in {video_dir}")

        successful_annotations = 0
        for video_file in video_files:
            try:
                episode_id = int(video_file.split("_episodes_")[1].split(".")[0])
                mp4_path = os.path.join(video_dir, video_file)
                pkl_path = os.path.join(
                    video_dir,
                    f"{self.env_name.replace('/', '_')}_episodes_{episode_id}.pkl",
                )
                output_path = os.path.join(
                    video_dir,
                    f"{self.env_name.replace('/', '_')}_episodes_{episode_id}_annotated.mp4",
                )

                if not os.path.exists(mp4_path):
                    print(f"Warning: Source video not found: {mp4_path}")
                    continue
                if not os.path.exists(pkl_path):
                    print(f"Warning: Pickle file not found: {pkl_path}")
                    continue

                create_annotated_video(mp4_path, pkl_path, output_path)
                successful_annotations += 1

                if slow_factor != 1.0:
                    slower_output_path = os.path.join(
                        video_dir,
                        f"{self.env_name.replace('/', '_')}_episodes_{episode_id}_annotated_slowed_{slow_factor}x.mp4",
                    )
                    create_slowed_video(output_path, slower_output_path, slow_factor)

            except Exception as e:
                print(f"Failed to annotate episode {episode_id}: {e}")
                continue

        if successful_annotations == 0:
            raise RuntimeError("Failed to create any annotated videos")


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        print(f"Error: {e}")
