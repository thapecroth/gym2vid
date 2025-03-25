import gymnasium as gym

# from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO
import cv2
import pickle
import os
from tqdm import tqdm
from typing import List, Literal, get_args
import fire
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import Pool
import torch
import traceback

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)
# N_GPUS = torch.cuda.device_count()
N_GPUS = 3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    env_name: list_of_gym_environments, model_path: str, episode: int, cuda_idx: int
):
    import gymnasium as gym
    import cv2

    model = TRPO.load(model_path, device=f"cuda:{cuda_idx}")

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
        os.makedirs("videos/" + env_name, exist_ok=True)
        env = gym.make(env_name, render_mode="rgb_array")
        env.metadata["render_fps"] = 24
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        episode_name = f"{env_name.replace('/', '_')}_episodes_{episode}"
        action_ls, states_ls = [], []

        # Initialize video writer
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        height, width, layers = frame.shape  # type: ignore
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        video_writer = cv2.VideoWriter(f"videos/{env_name}/{episode_name}.mp4", fourcc, 24, (width, height))  # type: ignore
        frame_counter = 0

        while not done and frame_counter < 240:
            action, _states = model.predict(obs)
            frame = env.render()
            assert isinstance(frame, np.ndarray)
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

        assert len(action_ls) == len(states_ls)

        with open(f"videos/{env_name}/{episode_name}.pkl", "wb") as f:
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
        traceback.print_stack()
        return False


def train(
    env_name: list_of_gym_environments, train_timesteps: int, n_envs: int = 8
) -> str:
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


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        print(f"Error: {e}")
