import pickle
import pathlib
import click
from datasets import Dataset
import numpy as np


N_ACTIONS = {tuple([0]): 0, tuple([1]): 0}  # 0:left, right
ACTION_SEP = 3


def process_pickle(file: pathlib.Path):
    with open(file, "rb") as f:
        data = pickle.load(f)
        action_ls = data["action_ls"]
        states_ls = data["states_ls"]
        n_timesteps = data["n_timesteps"]
        assert len(action_ls) == len(states_ls) == n_timesteps
        action_ls = [tuple(action) for action in action_ls]
        possible_actions = set(action_ls)
        assert len(possible_actions) == 2 == len(N_ACTIONS)
        assert (
            len(possible_actions - set(N_ACTIONS.keys())) == 0
        )  # we have mapping for all actions
        tokens = []
        for actions, states in zip(action_ls, states_ls):
            for action in actions:
                tokens.append(N_ACTIONS[action])
            tokens.append(ACTION_SEP)
        return tokens


@click.command()
@click.option("--input_glob", help="input glob")
@click.option("--output_dataset_name", help="output dataset name")
@click.option("--to_hub", is_flag=True, help="whether to upload to huggingface")
def main(input_glob: str, output_dataset_name: str, to_hub: bool):
    files = pathlib.Path(input_glob).glob("*.pkl")
    ds = Dataset.from_dict(
        {
            "files": files,
        }
    )
    ds.map(process_pickle)
    ds.save_to_disk(output_dataset_name)
    if to_hub:
        ds.push_to_hub(output_dataset_name, private=True)


if __name__ == "__main__":
    main()
